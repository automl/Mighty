from .base_agent import MightyAgent
from .dqn import MightyDQNAgent
from .ppo import MightyPPOAgent
from .sac import MightySACAgent

# FIXME: does it make sense to also split them in on- and off-policy agents? I mean for ifs in the base class?
# Then we wouldn't have to test for PPO, just for on-policy
VALID_AGENT_TYPES = ["DQN", "PPO", "SAC", "DDQN"]
AGENT_CLASSES = {
    "DQN": MightyDQNAgent,
    "PPO": MightyPPOAgent,
    "SAC": MightySACAgent,
    "DDQN": MightyDQNAgent,
}

from .factory import get_agent_class  # noqa: E402

__all__ = [
    "MightyAgent",
    "get_agent_class",
    "MightyDQNAgent",
    "MightyPPOAgent",
    "MightySACAgent",
]
