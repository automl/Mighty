from mighty.agent.base import AbstractAgent
# from mighty.agent.ddqn import DDQNAgent
from mighty.agent.coax_ddqn import DDQNAgent


def get_agent_class(agent_type: str) -> type(AbstractAgent):
    agent_class = None
    if agent_type == "ddqn":
        agent_class = DDQNAgent
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class
