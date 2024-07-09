"""Mighty rollout buffer."""

from __future__ import annotations

from collections.abc import Iterable

import dill as pickle
import numpy as np
import torch
from mighty.mighty_replay.buffer import MightyBuffer

import torch


class RolloutBatch:
    def __init__(
        self,
        observations,
        actions,
        rewards,
        advantages,
        returns,
        episode_starts,
        log_probs,
        values,
    ):
        self.observations = torch.from_numpy(observations.astype(np.float32)).unsqueeze(0)
        self.actions = torch.from_numpy(actions.astype(np.float32)).unsqueeze(0)
        self.rewards = torch.from_numpy(rewards.astype(np.float32)).unsqueeze(0)
        self.advantages = torch.from_numpy(advantages.astype(np.float32)).unsqueeze(0)
        self.returns = torch.from_numpy(returns.astype(np.float32)).unsqueeze(0)
        self.episode_starts = torch.from_numpy(
            episode_starts.astype(np.float32)
        ).unsqueeze(0)
        self.log_probs = torch.from_numpy(log_probs.astype(np.float32)).unsqueeze(0)
        self.values = torch.from_numpy(values.astype(np.float32)).unsqueeze(0)

    @property
    def size(self):
        return len(self.observations)

    def __len__(self):
        return self.size

    def __iter__(self):
        yield from zip(
            self.observations,
            self.actions,
            self.rewards,
            self.advantages,
            self.returns,
            self.episode_starts,
            self.log_probs,
            self.values,
            strict=False,
        )


class MightyRolloutBuffer(MightyBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape,
        act_dim,
        device: str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.reset()

    def reset(self) -> None:

        self.observations = []
        self.actions = []
        self.rewards = []
        self.returns = []
        self.episode_starts = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: np.ndarray
    ) -> None:
        last_values = last_values.clone().cpu().squeeze(1)
        last_gae_lam = 0

        # import pdb; pdb.set_trace()

        for step in reversed(range(self.observations.shape[0])):
            if step == self.observations.shape[0] - 1:
                next_non_terminal = torch.FloatTensor(1.0 - dones.astype(np.float32))
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
        
            last_gae_lam = (delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam)
            
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self, rollout_batch: RolloutBatch, _):
        
        # import pdb; pdb.set_trace()
        
        if len(self.observations) == 0:
            self.observations = rollout_batch.observations
            self.actions = rollout_batch.actions
            self.rewards = rollout_batch.rewards
            self.advantages = rollout_batch.advantages
            self.returns = rollout_batch.returns
            self.episode_starts = rollout_batch.episode_starts
            self.log_probs = rollout_batch.log_probs
            self.values = rollout_batch.values
        else:
            self.observations = torch.cat(
                (self.observations, rollout_batch.observations)
            )
            self.actions = torch.cat((self.actions, rollout_batch.actions))
            self.rewards = torch.cat((self.rewards, rollout_batch.rewards))
            self.advantages = torch.cat((self.advantages, rollout_batch.advantages))
            self.returns = torch.cat((self.returns, rollout_batch.returns))
            self.episode_starts = torch.cat(
                (self.episode_starts, rollout_batch.episode_starts)
            )
            self.log_probs = torch.cat((self.log_probs, rollout_batch.log_probs))
            self.values = torch.cat((self.values, rollout_batch.values))
        # if len(self) > self.buffer_size:
            
        #     import pdb; pdb.set_trace()
            
            
        #     self.observations = self.observations[len(self) - self.buffer_size :]
        #     self.actions = self.actions[len(self) - self.buffer_size :]
        #     self.rewards = self.rewards[len(self) - self.buffer_size :]
        #     self.advantages = self.advantages[len(self) - self.buffer_size :]
        #     self.returns = self.returns[len(self) - self.buffer_size :]
        #     self.episode_starts = self.episode_starts[len(self) - self.buffer_size :]
        #     self.log_probs = self.log_probs[len(self) - self.buffer_size :]
        #     self.values = self.values[len(self) - self.buffer_size :]

    def sample(self, batch_size: int):
        indices = np.random.permutation(len(self.observations))
        start_idx = 0
        samples = []
        while start_idx < len(self.observations):
            batch_inds = indices[start_idx : start_idx + batch_size]
            samples.append(self._get_samples(batch_inds))
            start_idx += batch_size
        return samples

    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.observations[batch_inds].numpy(),
            self.actions[batch_inds].numpy(),
            self.rewards[batch_inds].numpy(),
            self.advantages[batch_inds].numpy(),
            self.returns[batch_inds].numpy(),
            self.episode_starts[batch_inds].numpy(),
            self.log_probs[batch_inds].numpy(),
            self.values[batch_inds].numpy(),
        )

        return RolloutBatch(*data)

    def __len__(self):
        return len(self.observations)*self.n_envs

    def __bool__(self):
        return bool(self.observations)

    def save(self, filename="buffer.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
