from mighty.agent.base_agent import MightyAgent
from mighty.agent.sac import SACAgent
from mighty.agent.dqn import DQNAgent
from mighty.agent.ppo import PPOAgent


def get_agent_class(agent_type: str) -> type(MightyAgent):
    agent_class = None
    if agent_type == "DDQN" or agent_type == "DQN":
        agent_class = DQNAgent
    elif agent_type == "SAC":
        agent_class = SACAgent
    elif agent_type == "PPO":
        agent_class = PPOAgent
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class
