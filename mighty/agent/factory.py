from mighty.agent.base import AbstractAgent
from mighty.agent.ddqn import DDQNAgent
from mighty.agent.td3 import TD3Agent


def get_agent_class(agent_type: str) -> type(AbstractAgent):
    agent_class = None
    if agent_type == "DDQN":
        agent_class = DDQNAgent
    elif agent_type == "TD3":
        agent_class = TD3Agent
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class
