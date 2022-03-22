from mighty.agent.archive.base import AbstractAgent
# from mighty.agent.ddqn import DDQNAgent
from mighty.agent.dqn import DDQNAgent


def get_agent_class(agent_type: str) -> type(AbstractAgent):
    agent_class = None
    if agent_type == "DDQN":
        agent_class = DDQNAgent
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class
