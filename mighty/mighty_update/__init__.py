from mighty.mighty_update.q_learning import (
    ClippedDoubleQLearning,
    DoubleQLearning,
    QLearning,
    SPRQLearning,
)

from mighty.mighty_update.sac_update import SACUpdate
from mighty.mighty_update.ppo_update import PPOUpdate

__all__ = ["QLearning", "DoubleQLearning", "ClippedDoubleQLearning", "SPRQLearning", "SACUpdate", "PPOUpdate"]
