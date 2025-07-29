import math
from typing import Tuple

import torch
import torch.nn as nn

from mighty.mighty_models.networks import make_feature_extractor


class PPOModel(nn.Module):
    """PPO Model with policy and value networks."""

    output_style = (
        "squashed_gaussian"  # For continuous actions, we use squashed Gaussian output
    )

    def __init__(
        self,
        obs_shape: int,
        action_size: int,
        hidden_sizes: list[int] = [64, 64],
        activation: str = "tanh",
        continuous_action: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """Initialize the PPO model."""
        super().__init__()

        self.obs_size = int(obs_shape)
        self.action_size = int(action_size)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.continuous_action = continuous_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Make feature extractor
        self.feature_extractor_policy, feat_dim = make_feature_extractor(
            architecture="mlp",
            obs_shape=obs_shape,
            n_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        self.feature_extractor_value, _ = make_feature_extractor(
            architecture="mlp",
            obs_shape=obs_shape,
            n_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        if self.continuous_action:
            # Output size must be 2 * action_size (mean + log_std)
            final_out_dim = action_size * 2
        else:
            # For discrete actions, output logits of size = action_size
            final_out_dim = action_size

        # (Architecture based on
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py)

        # Policy network
        self.policy_head = nn.Sequential(
            self.feature_extractor_policy,  # [batch, feat_dim]
            nn.Linear(feat_dim, hidden_sizes[0]),  # [batch, hidden_sizes[0]]
            nn.LayerNorm(hidden_sizes[0]),  # (optional normalization)
            getattr(nn, activation.capitalize())(),  # e.g. tanh or ReLU
            nn.Linear(hidden_sizes[0], final_out_dim),  # [batch, final_out_dim]
        )

        # Value network
        self.value_head = nn.Sequential(
            self.feature_extractor_value,  # [batch, feat_dim]
            nn.Linear(feat_dim, hidden_sizes[0]),  # [batch, hidden_sizes[0]]
            nn.LayerNorm(hidden_sizes[0]),
            getattr(nn, activation.capitalize())(),
            nn.Linear(hidden_sizes[0], 1),  # [batch, 1]
        )

        # Orthogonal initialization
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                out_dim = m.out_features
                if self.continuous_action and out_dim == final_out_dim:
                    # This is the final policy‐output layer (mean & log_std):
                    gain = 0.01
                elif (not self.continuous_action) and out_dim == action_size:
                    # Final policy‐output layer (discrete‐logits):
                    gain = 0.01
                elif out_dim == 1:
                    # Final value‐output layer:
                    gain = 1.0
                else:
                    # Any intermediate hidden layer:
                    gain = math.sqrt(2)
                nn.init.orthogonal_(m.weight, gain)
                nn.init.constant_(m.bias, 0.0)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the policy network."""

        if self.continuous_action:
            raw = self.policy_head(x)  # [batch, 2 * action_size]
            mean, log_std = raw.chunk(2, dim=-1)  # each [batch, action_size]
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)  # [batch, action_size]

            # Sample a raw Gaussian z; during inference/training this is 'reparameterized'
            # (If you need a deterministic‐eval mode, you can add a flag argument here.)
            eps = torch.randn_like(mean)
            z = mean + std * eps  # [batch, action_size]
            action = torch.tanh(z)  # squash to [−1, +1]

            return action, z, mean, log_std

        else:
            logits = self.policy_head(x)  # [batch, action_size]
            return logits

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network."""
        return self.value_head(x)
