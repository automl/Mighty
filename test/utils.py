import gymnasium as gym


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}
