"""Utility wrappers for environments."""

from __future__ import annotations

import itertools

import gymnasium as gym
import numpy as np

class PufferWrapperEnv(gym.Env):
    """A dummy environment for testing purposes."""

    def __init__(self, env):
        """Initialize the dummy environment."""
        super().__init__()
        self.puffer_env = env

    def __getattr__(self, name):
        """Delegate attribute access to the underlying Pufferlib environment."""
        return getattr(self.puffer_env, name)

    def reset(self, *, seed = None, options = None):
        return self.puffer_env.reset(seed=seed)
    
    def step(self, action):
        return self.puffer_env.step(action)

class MinigridImgVecObs(gym.Wrapper):
    """Change observation space of a vectorized environment to be an image."""

    def __init__(self, env):
        """Change observation space of a vectorized environment to be an image."""
        super().__init__(env)
        self.single_observation_space = gym.spaces.Box(
            shape=self.env.observation_space.shape[1:], low=0, high=255
        )


class DictToVecActions(gym.Wrapper):
    """Flatten observation space of a vectorized environment."""

    def __init__(self, env):
        """Flatten observation space of a vectorized environment."""
        super().__init__(env)
        self.og_single_action_space = self.env.single_action_space
        self.single_action_space = gym.spaces.flatten_space(
            self.env.single_action_space
        )

        self.action_keys = list(self.og_single_action_space.spaces.keys())

    def step(self, action):
        """Take a step in the environment."""
        action = {key: action[i] for i, key in enumerate(self.action_keys)}
        return self.env.step(action)


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
        self.n_actions = len(self.env.action_space.nvec)

        self.action_mapper = {}
        for idx, prod_idx in zip(
            range(np.prod(self.env.action_space.nvec)),
            itertools.product(*[np.arange(val) for val in self.env.action_space.nvec]),
        ):
            self.action_mapper[idx] = prod_idx

        self.action_space = gym.spaces.Discrete(
            int(np.prod(self.env.action_space.nvec))
        )

    def step(self, action):
        """Maps discrete action value to array."""
        action = [self.action_mapper[a] for a in action]
        return self.env.step(action)


class CARLVectorEnvSimulator(gym.vector.VectorEnv):
    def __init__(self, env, **kwargs) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
            self.single_action_space = env.single_action_space
            self.single_observation_space = env.observation_space
        else:
            self.num_envs = 1
            self.single_action_space = env.action_space
            self.single_observation_space = env.observation_space

        if hasattr(env, "envs"):
            self.envs = env.envs
        else:
            self.envs = [env]

    def close(self):
        self.env.close()

    def reset(self, **kwargs):
        if self.num_envs > 1:
            return self.env.reset(**kwargs)
        else:
            if "seed" in kwargs and not isinstance(kwargs["seed"], int):
                kwargs["seed"] = int(kwargs["seed"][0])
            obs, info = self.env.reset(**kwargs)
            return np.array([obs]), np.array([info])

    def step(self, actions):
        if self.num_envs > 1:
            return self.env.step(actions)
        else:
            obs, reward, te, tr, info = self.env.step(actions[0])
            if te or tr:
                obs, info = self.env.reset()
            return (
                np.array([obs]),
                np.array([reward]),
                np.array([te]),
                np.array([tr]),
                np.array([info]),
            )

    @property
    def instance_id_list(self):
        return list(self.env.contexts.keys())

    @property
    def inst_ids(self):
        return self.env.context_selector.context_id

    @property
    def instances(self):
        return self.env.context

    @property
    def instance_set(self):
        return self.env.contexts

    def set_inst_id(self, inst_id):
        self.env.context_id = inst_id
        self.env.context = self.env.contexts[self.env.context_id]
        self.env._update_context()

    def set_instances(self, instance):
        self.env.context = instance
        self.env._update_context()

    def set_instance_set(self, instance_set):
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

    def set_inst_id(self, inst_ids):
        for i in range(self.num_envs):
            self.envs[i].set_inst_id(inst_ids[i % len(inst_ids)])

    def set_instance_set(self, instance_set):
        # Split instance set into parallel sets
        parallel_sets = [
            instance_set[i : i + self.num_envs]
            for i in range(0, len(instance_set), self.num_envs)
        ]
        addons = 0
        while len(parallel_sets[-1]) != len(parallel_sets[0]):
            parallel_sets[-1].append(parallel_sets[0][addons])
            addons += 1

            for i, ps in enumerate(parallel_sets):
                self.envs[i].set_instance_set(ps)

    def close(self):
        for env in self.envs:
            env.close()


class ProcgenVecEnv(gym.vector.SyncVectorEnv):
    def __init__(
        self,
        env,
        normalize_observations=False,
        normalize_reward=True,
        eps=1e-4,
        clip_obs=10,
        **kwargs,
    ) -> None:
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=np.ones((3, 64, 64)) * -np.inf,
            high=np.ones((3, 64, 64)) * np.inf,
            dtype=np.int32,
        )
        self.single_action_space = env.action_space
        self.single_observation_space = self.observation_space
        self.num_envs = env.num_envs
        self.envs = [env]
        self.obs_count = 1
        self.rew_count = 1
        self.normalize = normalize_observations
        self.norm_reward = normalize_reward

        self.eps = eps
        self.clip_obs = clip_obs

    def close(self):
        self.env.close()

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = obs["rgb"]

        if self.normalize:
            self.running_mean = np.zeros(obs.shape)
            self.running_std = np.zeros(obs.shape)
        if self.norm_reward:
            self.running_rew_mean = 0
            self.running_rew_std = 0

        if self.normalize:
            obs = self.normalize_obs(obs)
        return obs, {}

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        obs = obs["rgb"]
        if self.normalize:
            obs = self.normalize_obs(obs)
        if self.norm_reward:
            reward = self.normalize_reward(reward)
        return obs, reward, done, False, info

    def normalize_obs(self, obs):
        self.running_mean = (
            np.mean(obs) * len(obs)
        ) / self.obs_count + self.running_mean
        self.running_std = (
            self.obs_count * self.running_std
            + np.std(obs) * len(obs)
            + np.square(np.mean(obs) - self.running_mean)
            * self.obs_count
            * len(obs)
            / self.obs_count
        )
        obs = np.clip(
            (obs - self.running_mean) / np.sqrt(self.running_std + self.eps),
            -self.clip_obs,
            self.clip_obs,
        )
        self.obs_count += len(obs)
        return obs

    def normalize_reward(self, rews):
        self.running_rew_mean = (
            np.mean(rews) * len(rews)
        ) / self.rew_count + self.running_rew_mean
        self.running_rew_std = (
            self.rew_count * self.running_rew_std
            + np.std(rews) * len(rews)
            + np.square(np.mean(rews) - self.running_rew_mean)
            * self.rew_count
            * len(rews)
            / self.rew_count
        )
        rews = np.clip(
            (rews - self.running_mean) / np.sqrt(self.running_std + self.eps),
            -self.clip_obs,
            self.clip_obs,
        )

        self.rew_count += len(rews)
        return rews
