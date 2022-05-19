from mighty.agent.base_agent import MightyAgent
from mighty.agent.sac import MightySACAgent
from mighty.agent.dqn import MightyDQNAgent
from mighty.agent.ppo import MightyPPOAgent


def get_agent_class(agent_type: str) -> type(MightyAgent):
    """Transforms config keyword for agents to class"""
    agent_class = None
    if agent_type == "DDQN" or agent_type == "DQN":
        agent_class = MightyDQNAgent
    elif agent_type == "SAC":
        agent_class = MightySACAgent
    elif agent_type == "PPO":
        agent_class = MightyPPOAgent
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class
