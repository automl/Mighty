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
