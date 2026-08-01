from typing import Tuple

import torch
from torch import nn

from mighty.mighty_models.networks import ACTIVATIONS, make_feature_extractor


class SACModel(nn.Module):
    """SAC Model with squashed Gaussian policy and twin Q-networks."""

    output_style = (
        "squashed_gaussian"  # For continuous actions, we use squashed Gaussian output
    )
    tanh_squash: bool = True  # SAC always uses tanh squashing

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        log_std_min: float = -5,
        log_std_max: float = 2,
        action_low: float = -1,
        action_high: float = +1,
        **kwargs,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # This model is continuous only
        self.continuous_action = True

        # Register the per-dim scale and bias so we can rescale [-1,1]→[low,high].
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

        head_kwargs = {"hidden_sizes": [256, 256], "activation": "relu"}
        feature_extractor_kwargs = {
            "obs_shape": self.obs_size,
            "activation": "relu",
            "hidden_sizes": [256, 256],
            "n_layers": 2,
        }

        # Allow direct specification of hidden_sizes and activation at top level
        if "hidden_sizes" in kwargs:
            feature_extractor_kwargs["hidden_sizes"] = kwargs["hidden_sizes"]
            head_kwargs["hidden_sizes"] = kwargs["hidden_sizes"]
        if "activation" in kwargs:
            feature_extractor_kwargs["activation"] = kwargs["activation"]
            head_kwargs["activation"] = kwargs["activation"]

        if "head_kwargs" in kwargs:
            head_kwargs.update(kwargs["head_kwargs"])
        if "feature_extractor_kwargs" in kwargs:
            feature_extractor_kwargs.update(kwargs["feature_extractor_kwargs"])

        # Store for Q-network creation
        self.hidden_sizes = feature_extractor_kwargs.get("hidden_sizes", [256, 256])
        self.activation = feature_extractor_kwargs.get("activation", "relu")

        # Policy feature extractor and head
        self.policy_feature_extractor, policy_feat_dim = make_feature_extractor(
            **feature_extractor_kwargs
        )

        # Policy head: just the final output layer
        self.policy_head = make_policy_head(
            in_size=policy_feat_dim,
            out_size=self.action_size * 2,  # mean and log_std
            hidden_sizes=[],  # No hidden layers, just final linear layer
            activation=head_kwargs["activation"],
        )

        # Create policy_net for backward compatibility
        self.policy_net = nn.Sequential(self.policy_feature_extractor, self.policy_head)

        # Q-networks: feature extractors + heads
        q_feature_extractor_kwargs = feature_extractor_kwargs.copy()
        q_feature_extractor_kwargs["obs_shape"] = self.obs_size + self.action_size

        # Q-network 1
        self.q_feature_extractor1, q_feat_dim = make_feature_extractor(
            **q_feature_extractor_kwargs
        )
        self.q_head1 = make_q_head(
            in_size=q_feat_dim,
            hidden_sizes=[],  # No hidden layers, just final linear layer
            activation=head_kwargs["activation"],
        )
        self.q_net1 = nn.Sequential(self.q_feature_extractor1, self.q_head1)

        # Q-network 2
        self.q_feature_extractor2, _ = make_feature_extractor(
            **q_feature_extractor_kwargs
        )
        self.q_head2 = make_q_head(
            in_size=q_feat_dim,
            hidden_sizes=[],  # No hidden layers, just final linear layer
            activation=head_kwargs["activation"],
        )
        self.q_net2 = nn.Sequential(self.q_feature_extractor2, self.q_head2)

        # Target Q-networks
        self.target_q_feature_extractor1, _ = make_feature_extractor(
            **q_feature_extractor_kwargs
        )
        self.target_q_head1 = make_q_head(
            in_size=q_feat_dim,
            hidden_sizes=[],  # No hidden layers, just final linear layer
            activation=head_kwargs["activation"],
        )
        self.target_q_net1 = nn.Sequential(
            self.target_q_feature_extractor1, self.target_q_head1
        )

        self.target_q_feature_extractor2, _ = make_feature_extractor(
            **q_feature_extractor_kwargs
        )
        self.target_q_head2 = make_q_head(
            in_size=q_feat_dim,
            hidden_sizes=[],  # No hidden layers, just final linear layer
            activation=head_kwargs["activation"],
        )
        self.target_q_net2 = nn.Sequential(
            self.target_q_feature_extractor2, self.target_q_head2
        )

        # Copy weights from live to target networks
        self.target_q_feature_extractor1.load_state_dict(
            self.q_feature_extractor1.state_dict()
        )
        self.target_q_head1.load_state_dict(self.q_head1.state_dict())
        self.target_q_feature_extractor2.load_state_dict(
            self.q_feature_extractor2.state_dict()
        )
        self.target_q_head2.load_state_dict(self.q_head2.state_dict())

        # Freeze target networks
        for p in self.target_q_feature_extractor1.parameters():
            p.requires_grad = False
        for p in self.target_q_head1.parameters():
            p.requires_grad = False
        for p in self.target_q_feature_extractor2.parameters():
            p.requires_grad = False
        for p in self.target_q_head2.parameters():
            p.requires_grad = False

        # Create a value function wrapper for compatibility
        class ValueFunctionWrapper(nn.Module):
            def __init__(self, parent_model):
                super().__init__()
                self.parent_model = parent_model

            def forward(self, x):
                # SAC doesn't have a separate value function, but for compatibility
                # we can return the minimum of the two Q-values with a zero action
                # This is mainly for interface compatibility
                batch_size = x.shape[0]
                zero_action = torch.zeros(
                    batch_size, self.parent_model.action_size, device=x.device
                )
                state_action = torch.cat([x, zero_action], dim=-1)
                q1 = self.parent_model.forward_q1(state_action)
                q2 = self.parent_model.forward_q2(state_action)
                return torch.min(q1, q2)

        self.value_function_module = ValueFunctionWrapper(self)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy sampling.

        Returns:
          action: torch.Tensor in rescaled range [action_low, action_high]
          z: raw Gaussian sample before tanh
          mean: Gaussian mean
          log_std: Gaussian log std
        """
        x = self.policy_net(state)
        mean, log_std = x.chunk(2, dim=-1)

        # Soft clamping
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        std = torch.exp(log_std)

        if deterministic:
            z = mean
        else:
            z = mean + std * torch.randn_like(mean)

        # tanh→[-1,1]
        raw_action = torch.tanh(z)

        # Rescale into [action_low, action_high]
        action = raw_action * self.action_scale + self.action_bias

        return action, z, mean, log_std

    def policy_log_prob(
        self, z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-prob of action a = tanh(z), correcting for tanh transform.
        """
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_pz = dist.log_prob(z).sum(dim=-1, keepdim=True)
        eps = 1e-6  # small constant to avoid numerical issues
        log_correction = (torch.log(1 - torch.tanh(z).pow(2) + eps)).sum(
            dim=-1, keepdim=True
        )
        log_pa = log_pz - log_correction
        return log_pa

    def forward_q1(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.q_net1(state_action)

    def forward_q2(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.q_net2(state_action)

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value function for compatibility."""
        return self.value_function_module(x)


def make_q_head(in_size, hidden_sizes=None, activation="relu"):
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
        layers.append(ACTIVATIONS[activation]())
        last_size = size

    layers.append(nn.Linear(last_size, 1))
    return nn.Sequential(*layers)


def make_policy_head(in_size, out_size, hidden_sizes=None, activation="relu"):
    """Make policy head network (actor)."""
    if hidden_sizes is None:
        hidden_sizes = []

    layers = []
    last_size = in_size
    if isinstance(last_size, list):
        last_size = last_size[0]

    for size in hidden_sizes:
        layers.append(nn.Linear(last_size, size))
        layers.append(ACTIVATIONS[activation]())
        last_size = size

    layers.append(nn.Linear(last_size, out_size))
    return nn.Sequential(*layers)
