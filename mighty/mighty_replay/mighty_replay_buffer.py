"""Mighty replay buffer."""

from __future__ import annotations

from collections.abc import Iterable

import dill as pickle
import numpy as np
import torch

from mighty.mighty_replay.buffer import MightyBuffer


def flatten_infos(xs):
    """Transform info dict to flat list.

    :param xs: info dict
    :return: flattened infos
    """
    if isinstance(xs, dict):
        xs = list(xs.values())
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, str | bytes):
            yield from flatten_infos(x)
        else:
            yield x


class TransitionBatch:
    """Transition batch."""

    def __init__(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize TransitionBatch."""
        if isinstance(rewards, float | int):
            observations = np.array([observations], dtype=np.float32)
            actions = np.array([actions], dtype=np.float32)
            rewards = np.array([rewards], dtype=np.float32)
            next_observations = np.array([next_observations], dtype=np.float32)
            dones = np.array([dones], dtype=np.float32)
        if isinstance(rewards, np.ndarray):
            self.observations = torch.from_numpy(observations.astype(np.float32)).to(
                device
            )
            self.actions = torch.from_numpy(actions.astype(np.float32)).to(device)
            self.rewards = torch.from_numpy(rewards.astype(np.float32)).to(device)
            self.next_obs = torch.from_numpy(next_observations.astype(np.float32)).to(
                device
            )
            self.dones = torch.from_numpy(dones.astype(np.int64)).to(device)
        else:
            self.observations = observations.to(device)
            self.actions = actions.to(device)
            self.rewards = rewards.to(device)
            self.next_obs = next_observations.to(device)
            self.dones = dones.to(device)

    @property
    def size(self):
        """Current buffer size."""
        return len(self.observations)

    def __len__(self):
        return self.size

    def __iter__(self):
        yield from zip(
            self.observations,
            self.actions,
            self.rewards,
            self.next_obs,
            self.dones,
            strict=False,
        )


class MightyReplay(MightyBuffer):
    """Simple replay buffer."""

    def __init__(
        self,
        capacity,
        keep_infos=False,
        flatten_infos=False,
        device: torch.device | str = "cpu",
    ):
        """Initialize Buffer.

        :param capacity: Buffer size
        :param random_seed: Seed for sampling
        :param keep_infos: Keep the extra info dict. Required for some algorithms.
        :param flatten_infos: Make flat list from infos.
            Might be necessary, depending on info content.
        :return:
        """
        self.capacity = capacity
        self.keep_infos = keep_infos
        self.flatten_infos = flatten_infos
        self.device = torch.device(device)
        self.rng = np.random.default_rng()
        self.reset()

    @property
    def full(self):
        """Check if the buffer is full."""
        return self.index + 1 >= self.capacity

    def add(self, transition_batch, _):
        """Add transition(s).

        :param transition_batch: Transition(s) to add
        :param metrics: Current metrics dict
        :return:
        """
        if not self.keep_infos:
            transition_batch.extra_info = []
        elif self.flatten_infos:
            transition_batch.extra_info = [
                list(flatten_infos(transition_batch.extra_info))
            ]

        # Remove nan values from transitions (from autoreset envs) so they don't get learned from
        transition_batch.observations = transition_batch.observations[~torch.any(transition_batch.observations.isnan(), dim=1)]
        transition_batch.next_obs = transition_batch.next_obs[~torch.any(transition_batch.next_obs.isnan(), dim=1)]
        if transition_batch.actions.ndim > 1:
            transition_batch.actions = transition_batch.actions[~torch.any(transition_batch.actions.isnan(), dim=1)]
        else:
            transition_batch.actions = transition_batch.actions[~transition_batch.actions.isnan()]
        transition_batch.rewards = transition_batch.rewards[~transition_batch.rewards.isnan()]
        transition_batch.dones = transition_batch.dones[~transition_batch.dones.isnan()]

        self.index += transition_batch.observations.size(0)
        if len(self.obs) == 0:
            self.obs = transition_batch.observations
            self.next_obs = transition_batch.next_obs
            self.actions = transition_batch.actions
            self.rewards = transition_batch.rewards
            self.dones = transition_batch.dones
        else:
            self.obs = torch.cat((self.obs, transition_batch.observations))
            self.next_obs = torch.cat((self.next_obs, transition_batch.next_obs))
            self.actions = torch.cat((self.actions, transition_batch.actions))
            self.rewards = torch.cat((self.rewards, transition_batch.rewards))
            self.dones = torch.cat((self.dones, transition_batch.dones))
        if len(self) > self.capacity:
            self.obs = self.obs[len(self) - self.capacity :]
            self.next_obs = self.next_obs[len(self) - self.capacity :]
            self.actions = self.actions[len(self) - self.capacity :]
            self.rewards = self.rewards[len(self) - self.capacity :]
            self.dones = self.dones[len(self) - self.capacity :]
            self.index = self.capacity

    def sample(self, batch_size=32):
        """Sample transitions."""
        batch_indices = self.rng.choice(np.arange(len(self)), size=batch_size)
        return TransitionBatch(
            self.obs[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_obs[batch_indices],
            self.dones[batch_indices],
            device=self.device,
        )

    def reset(self):
        """Reset the buffer."""
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.index = 0

    def __len__(self):
        return len(self.actions)

    def __bool__(self):
        return bool(len(self))

    def save(self, filename="buffer.pkl"):
        """Save the buffer to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
