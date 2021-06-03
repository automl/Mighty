import numpy as np


class AbstractAgent:
    """
    Any mighty agent should implement this class
    """

    def __init__(self, env: dacbench.AbstractEnv, gamma: float, logger):
        """
        Initialize an Agent
        :param gamma: discount factor
        :param env: environment to train on
        :param logger: data logger
        """
        self.gamma = gamma
        self.env = env
        self.logger = logger
        #TODO: make util function that can detect this correctly for all kinds of gym spaces and use it here
        self.action_dim = self.env.action.space.n
        self.state_shape = self.env.observation_space.shape

    def get_action(self, state: np.ndarray, epsilon:float = 0.0):
        """
        Return action given a state and policy epsilon (NOTE: epsilon only makes sense in certain cases)
        :param state: environment state
        :param epsilon: epsilon for action selection
        :return: action for the next environment step
        """
        raise NotImplementedError

    def train(self, episodes: int = 1):
        """
        Trains the agent for a given amount of episodes
        :param episodes: Training episodes
        """
        raise NotImplementedError

    def eval(self, env: dacbench.AbstractEnv, episodes: int = 1):
        """
        Evaluates the agent on a given environment
        :param env: evaluation environment
        :param episodes: number of evaluation episodes
        """
        raise NotImplementedError

    def checkpoint(self, filepath: str):
        """
        Save agent policy
        :param filepath: path to save point
        """
        raise NotImplementedError

    def update_policy(self, deltas: np.ndarray):
        """
        Perform weight update with given deltas
        :param deltas: weight updates
        """
        raise NotImplementedError
