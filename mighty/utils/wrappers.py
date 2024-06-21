"""Utility wrappers for environments."""

from __future__ import annotations

from functools import partial
import itertools
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


class MultiDiscreteActionWrapper(gym.Wrapper):
    """Wrapper to cast MultiDiscrete action spaces to Discrete. This should improve usability with standard RL libraries."""

    def __init__(self, env):
        """
        Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap

        """
        super().__init__(env)
        self.n_actions = len(self.env.single_action_space.nvec)
        self.single_action_space = gym.spaces.Discrete(
            np.prod(self.env.single_action_space.nvec)
        )
        self.action_mapper = {}
        for idx, prod_idx in zip(
            range(np.prod(self.env.single_action_space.nvec)),
            itertools.product(
                *[np.arange(val) for val in self.env.single_action_space.nvec]
            ),
        ):
            self.action_mapper[idx] = prod_idx

    def step(self, action):
        """Maps discrete action value to array."""
        action = [self.action_mapper[a] for a in action]
        return self.env.step(action)


class CARLVectorEnvSimulator(gym.vector.VectorEnv):
    def __init__(self, env, **kwargs) -> None:
        self.env = env
        self.single_action_space = env.action_space
        self.single_observation_space = env.observation_space
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        else:
            self.num_envs = 1

        if hasattr(env, "envs"):
            self.envs = env.envs
        else:
            self.envs = [env]

    def close(self):
        self.env.close()

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    @property
    def instance_id_list(self):
        return list(self.env.contexts.keys())

    @property
    def inst_ids(self):
        return self.env.context_id

    @property
    def instances(self):
        return self.env.context
    
    @property
    def instance_set(self):
        return self.env.contexts

    @inst_ids.setter
    def inst_ids(self, inst_id):
        self.env.context_id = inst_id

    @instances.setter
    def instances(self, instance):
        self.env.context = instance

    @instance_set.setter
    def instance_set(self, instance_set):
        self.env.contexts = instance_set


class ContextualVecEnv(gym.vector.SyncVectorEnv):
    @property
    def instance_id_list(self):
        return self.envs[0].instance_id_list

    @property
    def inst_ids(self):
        return [self.envs[i].inst_id for i in range(self.num_envs)]

    @property
    def instances(self):
        return [self.envs[i].instance for i in range(self.num_envs)]
    
    @property
    def instance_set(self):
        return [self.envs[i].instance_set for i in range(self.num_envs)]

    @inst_ids.setter
    def inst_ids(self, inst_ids):
        for i in range(self.num_envs):
            self.envs[i].set_inst_id(inst_ids[i])

    @instance_set.setter
    def instance_set(self, instance_set):
        # Split instance set into parallel sets
        parallel_sets = [instance_set[i:i + self.num_envs] for i in range(0, len(instance_set), self.num_envs)]
        addons = 0
        while len(parallel_sets[-1]) != len(parallel_sets[0]):
            parallel_sets[-1].append(parallel_sets[0][addons])
            addons += 1

        for i, ps in enumerate(parallel_sets):
            self.envs[i].set_instance_set(ps)
