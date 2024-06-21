import gymnasium as gym
import torch
import numpy as np
import os


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}


class DummyModel:
    def __init__(self, action=1):
        self.action = action

    def __call__(self, s):
        fake_qs = np.zeros((len(s), 5))
        fake_qs[:, self.action] = 1
        return torch.tensor(fake_qs)


def clean(logger):
        logger.close()
        os.remove(logger.log_file.name)
        if (logger.log_dir / "rewards.jsonl").exists():
            os.remove(logger.log_dir / "rewards.jsonl")
        if (logger.log_dir / "eval.jsonl").exists():
            os.remove(logger.log_dir / "eval.jsonl")
        os.removedirs(logger.log_dir)