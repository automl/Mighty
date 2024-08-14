from torch import nn
from mighty.mighty_models.networks import make_feature_extractor


class SACModel(nn.Module):
    """SAC Model with policy and Q-networks."""

    def __init__(self, obs_size, action_size, hidden_sizes=[64, 64], activation="relu"):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size

        # Policy network mapping observations to actions
        self.policy_net = nn.Sequential(
            make_feature_extractor(
                architecture="mlp",
                obs_shape=obs_size,
                n_layers=len(hidden_sizes),
                hidden_sizes=hidden_sizes,
                activation=activation,
            )[0],
            nn.Linear(hidden_sizes[-1], 2),
        )

        # Q-networks mapping observation and actions to Q-values
        self.q_net1 = nn.Sequential(
            make_feature_extractor(
                architecture="mlp",
                obs_shape=obs_size + action_size,
                n_layers=len(hidden_sizes),
                hidden_sizes=hidden_sizes,
                activation=activation,
            )[0],
            nn.Linear(hidden_sizes[-1], 1),
        )
        self.q_net2 = nn.Sequential(
            make_feature_extractor(
                architecture="mlp",
                obs_shape=obs_size + action_size,
                n_layers=len(hidden_sizes),
                hidden_sizes=hidden_sizes,
                activation=activation,
            )[0],
            nn.Linear(hidden_sizes[-1], 1),
        )

        # Value network
        self.value_net = nn.Sequential(
            make_feature_extractor(
                architecture="mlp",
                obs_shape=obs_size,
                n_layers=len(hidden_sizes),
                hidden_sizes=hidden_sizes,
                activation=activation,
            )[0],
            nn.Linear(hidden_sizes[-1], 1),
        )

    def forward(self, state):
        x = self.policy_net(state)
        mean, log_std = x.chunk(2, dim=-1)
        # FIXME: this should probably be a hyperparameter
        log_std = log_std.clamp(-20, 2)
        return mean, log_std.exp()

    # FIXME: do all of these really need to be separate functions?
    def forward_q1(self, state_action):
        return self.q_net1(state_action)

    def forward_q2(self, state_action):
        return self.q_net2(state_action)

    def forward_value(self, state):
        return self.value_net(state)
