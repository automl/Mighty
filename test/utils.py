import gymnasium as gym
import torch
import numpy as np
import shutil


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.action_space = gym.spaces.Discrete(4)
        self.inst_id = None
        self.instance_set = [42]

    @property
    def instance_id_list(self):
        return [self.inst_id]

    def set_inst_id(self, inst_id):
        self.inst_id = inst_id

    def set_instance_set(self, instance_set):
        self.instance_set = instance_set

    def reset(self, options={}):
        if self.inst_id is None:
            self.inst_id = np.random.default_rng().integers(0, 100)
        return self.observation_space.sample(), {}

    def step(self, action):
        tr = np.random.default_rng().choice([0, 1], p=[0.9, 0.1])
        return self.observation_space.sample(), 0, False, tr, {}


class DummyModel:
    def __init__(self, action=1):
        self.action = action

    def __call__(self, s):
        fake_qs = np.zeros((len(s), 5))
        fake_qs[:, self.action] = 1
        return torch.tensor(fake_qs)


def clean(logger):
    logger.close()
    shutil.rmtree(logger.log_dir.parents[0], ignore_errors=False, onerror=None)