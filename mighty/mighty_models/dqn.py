"""Policy networks for DQN and extensions."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from mighty.mighty_models.networks import MLP, make_feature_extractor
from torch import jit, nn


class DQN(nn.Module):
    """DQN network."""

    def __init__(self, num_actions, obs_size, dueling=False, **kwargs):
        """Initialize the network."""
        super().__init__()
        head_kwargs = {"hidden_sizes": [32, 32]}
        feature_extractor_kwargs = {"obs_shape": obs_size}
        if "head_kwargs" in kwargs:
            head_kwargs.update(kwargs["head_kwargs"])
        if "feature_extractor_kwargs" in kwargs:
            feature_extractor_kwargs.update(kwargs["feature_extractor_kwargs"])

        # Make feature extractor
        self.feature_extractor, self.output_size = make_feature_extractor(
            **feature_extractor_kwargs
        )
        self.dueling = dueling
        self.num_actions = int(num_actions)
        self.obs_size = obs_size
        self.hidden_sizes = head_kwargs["hidden_sizes"]

        # Make policy head
        self.head, self.value, self.advantage = make_q_head(
            self.output_size,
            self.num_actions,
            **head_kwargs,
        )

    def forward(self, x):
        """Forward pass."""
        x = self.feature_extractor(x)
        x = self.head(x)
        advantage = self.advantage(x)
        if self.dueling:
            value = self.value(x)
            x = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            x = advantage
        return x

    def reset_head(self, hidden_sizes=None):
        """Reset the head of the network."""
        if hidden_sizes is None:
            hidden_sizes = self.hidden_sizes
        self.head, self.value, self.advantage = make_q_head(
            self.output_size,
            self.num_actions,
            hidden_sizes,
        )
        self.hidden_sizes = hidden_sizes

    def shrink_weights(self, shrinkage, noise_weight):
        """Shrink weights of the network."""
        params_old = deepcopy(list(self.head.parameters()))
        value_params_old = deepcopy(list(self.value.parameters()))
        adv_params_old = deepcopy(list(self.advantage.parameters()))
        self.reset_head(hidden_sizes=self.hidden_sizes)
        for p_old, p_rand in zip(*[params_old, self.head.parameters()], strict=False):
            p_rand.data = deepcopy(shrinkage * p_old.data + noise_weight * p_rand.data)
        for p_old, p_rand in zip(
            *[adv_params_old, self.advantage.parameters()], strict=False
        ):
            p_rand.data = deepcopy(shrinkage * p_old.data + noise_weight * p_rand.data)
        if self.dueling:
            for p_old, p_rand in zip(
                *[value_params_old, self.value.parameters()], strict=False
            ):
                p_rand.data = deepcopy(
                    shrinkage * p_old.data + noise_weight * p_rand.data
                )

    def __getstate__(self):
        return (
            self.feature_extractor,
            self.head,
            self.advantage,
            self.value,
            self.dueling,
            self.num_actions,
        )

    def __setstate__(self, state):
        self.feature_extractor = state[0]
        self.head = state[1]
        self.advantage = state[2]
        self.value = state[3]
        self.dueling = state[4]
        self.num_actions = state[5]


class IQN(DQN):
    """IQN network, based on https://github.com/BY571/IQN-and-Extensions/."""

    def __init__(
        self,
        num_actions,
        obs_size,
        num_taus=16,
        n_cos=64,
        feature_extractor="mlp",
        dueling=False,
        **feature_extractor_kwargs,
    ):
        """Initialize the network."""
        super().__init__(
            num_actions=num_actions,
            obs_size=obs_size,
            feature_extractor=feature_extractor,
            dueling=dueling,
            **feature_extractor_kwargs,
        )
        self.num_taus = num_taus
        self.last_taus = None
        self.n_cos = n_cos
        self.pis = torch.FloatTensor(
            [np.pi * i for i in range(1, self.n_cos + 1)]
        ).view(1, 1, self.n_cos)
        self.cos_embedding = nn.Linear(self.n_cos, self.output_size)

    def forward(self, x, num_taus=None):
        """Forward pass."""
        x = self.feature_extractor(x)
        x = self.head(x)
        if num_taus is None:
            num_taus = self.num_taus
        batch_size = x.shape[0]

        self.last_taus = torch.rand(batch_size, num_taus).unsqueeze(-1)
        cos = torch.cos(self.last_taus * self.pis)
        cos = cos.view(batch_size * num_taus, self.n_cos)
        cos_embedding = self.cos_embedding(cos)
        cos_x = torch.relu(cos_embedding).view(
            batch_size, num_taus, self.output_size
        )  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) -> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_taus, self.output_size)

        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            x = value + advantage - advantage.mean(dim=1, keepdim=True)
        return x.mean(dim=1)

    def get_taus(self):
        """Get last taus."""
        return self.last_taus


class SPRQN(nn.Module):
    """SPRQN network."""

    def __init__(self, num_actions, *head_kwargs, **feature_extractor_kwargs):
        """Initialize the network."""
        super().__init__(num_actions, *head_kwargs, **feature_extractor_kwargs)
        # TODO: init
        self.projection = MLP()
        self.predictor = MLP()
        # TODO: not an MLP
        self.transition_model = MLP()

    @jit.script_method
    def forward(self, x, actions=None):
        """Forward pass."""
        encoding = self.feature_extractor(x)
        x = encoding.reshape(-1)
        projection = self.projection(x)
        x = nn.relu(projection)
        x = self.head(x)
        if actions is not None:
            spr_pred = self.transition_model(encoding, actions)
            spr_pred = self.projection(x)
            spr_pred = self.predictor(x)
            output = (x, spr_pred)
        else:
            output = x
        return output

    @jit.script_method
    def project(self, x):
        """Feature projection."""
        encoding = self.feature_extractor(x)
        x = encoding.reshape(-1)
        return self.projection(x)


def make_q_head(in_size, num_actions, hidden_sizes=None):
    """Make Q head network."""
    # Make fully connected layers
    if hidden_sizes is None:
        hidden_sizes = []
    layers = []
    last_size = in_size
    if isinstance(last_size, list):
        last_size = last_size[0]

    for size in hidden_sizes:
        layers.append(nn.Linear(last_size, size))
        last_size = size

    # Make value layer
    value_layer = nn.Linear(last_size, 1)
    # Make advantage layer
    advantage_layer = nn.Linear(last_size, num_actions)
    return nn.Sequential(*layers), value_layer, advantage_layer
