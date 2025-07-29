from typing import Tuple

import torch
from torch import nn

from mighty.mighty_models.networks import make_feature_extractor


class SACModel(nn.Module):
    """SAC Model with squashed Gaussian policy and twin Q-networks."""

    output_style = (
        "squashed_gaussian"  # For continuous actions, we use squashed Gaussian output
    )

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # This model is continuous only
        self.continuous_action = True

        # Shared feature extractor for policy and Q-networks
        extractor, out_dim = make_feature_extractor(
            architecture="mlp",
            obs_shape=obs_size,
            n_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        # Policy network outputs mean and log_std
        self.policy_net = nn.Sequential(
            extractor,
            nn.Linear(out_dim, action_size * 2),
        )

        # Twin Q-networks
        # — live Q-nets —
        self.q_net1 = self._make_q_net()
        self.q_net2 = self._make_q_net()

        self.target_q_net1 = self._make_q_net()
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2 = self._make_q_net()
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        for p in self.target_q_net1.parameters():
            p.requires_grad = False
        for p in self.target_q_net2.parameters():
            p.requires_grad = False

    def _make_q_net(self) -> nn.Sequential:
        q_in = self.obs_size + self.action_size
        q_extractor, _ = make_feature_extractor(
            architecture="mlp",
            obs_shape=q_in,
            n_layers=len(self.hidden_sizes),
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        )
        return nn.Sequential(q_extractor, nn.Linear(self.hidden_sizes[-1], 1))

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy sampling.

        Returns:
          action: torch.Tensor in [-1,1]
          z: raw Gaussian sample before tanh
          mean: Gaussian mean
          log_std: Gaussian log std
        """
        x = self.policy_net(state)
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            z = mean
        else:
            z = mean + std * torch.randn_like(mean)
        action = torch.tanh(z)
        return action, z, mean, log_std

    def forward_q1(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.q_net1(state_action)

    def forward_q2(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.q_net2(state_action)
