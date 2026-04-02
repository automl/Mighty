import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


class DummyEnv(gym.Env):
    """Simple dummy discrete environment for testing."""

    def __init__(self):
        super().__init__()
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

    def reset(self, options={}, seed=None):
        super().reset(seed=seed if seed is not None else 0)
        if self.inst_id is None:
            self.inst_id = self._np_random.integers(0, 100)
        return self.observation_space.sample(), {}

    def step(self, action):
        tr = self._np_random.choice([0, 1], p=[0.95, 0.05])
        return self.observation_space.sample(), self._np_random.random(), False, tr, {}


class DummyContinuousEnv(gym.Env):
    """Simple dummy continuous environment for testing."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.inst_id = None
        self.instance_set = [42]

    @property
    def instance_id_list(self):
        return [self.inst_id]

    def set_inst_id(self, inst_id):
        self.inst_id = inst_id

    def set_instance_set(self, instance_set):
        self.instance_set = instance_set

    def reset(self, options={}, seed=None):
        super().reset(seed=seed if seed is not None else 0)
        if self.inst_id is None:
            self.inst_id = self._np_random.integers(0, 100)
        return self.observation_space.sample(), {}

    def step(self, action):
        tr = self._np_random.choice([0, 1], p=[0.9, 0.1])
        return self.observation_space.sample(), self._np_random.random(), False, tr, {}


class DummyModel:
    def __init__(self, action=1):
        self.action = action

    def __call__(self, s):
        fake_qs = np.zeros((len(s), 5))
        fake_qs[:, self.action] = 1
        return torch.tensor(fake_qs)


def clean(path):
    """Helper function to clean up test directories."""
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=False, onerror=None)
