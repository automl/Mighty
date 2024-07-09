import torch
import torch.nn as nn
from mighty.mighty_models.networks import make_feature_extractor


from typing import Tuple


class PPOModel(nn.Module):
    """PPO Model with policy and value networks."""

    def __init__(
        self,
        obs_size,
        action_size,
        hidden_sizes=[64, 64],
        activation="tanh",
        continuous_action=False,
    ):
        """Initialize the PPO model."""
        super().__init__()

        self.obs_size = int(obs_size)
        self.action_size = int(action_size)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.continuous_action = continuous_action

        # Make feature extractor
        self.feature_extractor, self.output_size = make_feature_extractor(
            architecture="mlp",
            obs_shape=obs_size,
            n_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        # Policy network
        if self.continuous_action:
            self.policy_net = nn.Sequential(
                self.feature_extractor, nn.Linear(self.output_size, 2)
            )
        else:
            self.policy_net = nn.Sequential(
                self.feature_extractor, nn.Linear(self.output_size, self.action_size)
            )

        # Value network
        self.value_net = nn.Sequential(
            self.feature_extractor, nn.Linear(self.output_size, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default Forward as the forward pass through the policy network."""

        return self.forward_policy(x)

    def forward_policy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the policy network."""

        # pdb.set_trace()

        x = self.policy_net(x)

        if self.continuous_action:
            mean, log_std = x.chunk(2, dim=-1)
            mean = mean.squeeze(-1)  # Remove the extra dimension
            log_std = log_std.clamp(-20, 2).squeeze(-1)  # Remove the extra dimension
            return mean, log_std.exp()
        else:
            return x  # return logits

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network."""
        return self.value_net(x)
