import numpy as np
from gym.spaces import Discrete, Box


class MockEnvDiscreteActions:
    def __init__(self):
        self.action_space = Discrete(5)
        self.observation_space = Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.__count = 0
        self.instance_set = {0: [0]}

    def step(self, action):
        self.__count += 1
        return np.array([self.__count, 0]), 1, self.__count >= 10, None

    def reset(self):
        self.__count = 0
        return np.array([self.__count, 0])


class MockEnvContinuousActions:
    def __init__(self):
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]))
        self.observation_space = Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.__count = 0
        self.instance_set = {0: [0]}

    def step(self, action):
        self.__count += 1
        return np.array([self.__count, 0]), 1, self.__count >= 10, None

    def reset(self):
        self.__count = 0
        return np.array([self.__count, 0])
