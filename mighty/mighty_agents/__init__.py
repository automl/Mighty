from .base_agent import MightyAgent
from .dqn import MightyDQNAgent
from .ppo import MightyPPOAgent
from .sac import MightySACAgent


VALID_AGENT_TYPES = ["DQN", "PPO", "SAC", "DDQN"]
AGENT_CLASSES = {"DQN": MightyDQNAgent, "PPO": MightyPPOAgent, "SAC": MightySACAgent, "DDQN": MightyDQNAgent}

from .factory import get_agent_class
__all__ = ["MightyAgent", "get_agent_class", "MightyDQNAgent", "MightyPPOAgent", "MightySACAgent"]