"""Utility wrappers for environments."""
from __future__ import annotations

from functools import partial

import gymnasium as gym
import numpy as np


class PufferlibToGymAdapter(gym.Wrapper):
    """Adapter for Pufferlib environments to be used with OpenAI Gym."""

    def __init__(self, env):
        """Adapter for Pufferlib environments to be used with OpenAI Gym."""
        super().__init__(env)
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 60,
        }

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation."""
        if "options" in kwargs:
            del kwargs["options"]
        obs, info = self.env.reset(**kwargs)
        return obs, info
    

class FlattenVecObs(gym.Wrapper):
    """Flatten observation space of a vectorized environment."""

    def __init__(self, env):
        """Flatten observation space of a vectorized environment."""
        super().__init__(env)
        self.single_observation_space = gym.spaces.flatten_space(
            self.env.single_observation_space
        )

    def reset(self, options=None):
        """Reset the environment and return the initial observation."""
        if options is None:
            options = {}
        obs, info = self.env.reset(options=options)
        obs = np.array(
            list(map(partial(gym.spaces.flatten, self.single_observation_space), obs))
        )
        return obs, info

    def step(self, action):
        """Take a step in the environment."""
        obs, reward, te, tr, info = self.env.step(action)
        obs = np.array(
            list(map(partial(gym.spaces.flatten, self.single_observation_space), obs))
        )
        return obs, reward, te, tr, info


class MinigridImgVecObs(gym.Wrapper):
    """Change observation space of a vectorized environment to be an image."""

    def __init__(self, env):
        """Change observation space of a vectorized environment to be an image."""
        super().__init__(env)
        self.single_observation_space = gym.spaces.Box(
            shape=self.env.observation_space.shape[1:], low=0, high=255
        )
