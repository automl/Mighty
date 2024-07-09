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
        self, observations, actions, rewards, next_observations, dones
    ) -> None:
        """Initialize TransitionBatch."""
        if isinstance(rewards, float | int):
            observations = np.array([observations], dtype=np.float32)
            actions = np.array([actions], dtype=np.float32)
            rewards = np.array([rewards], dtype=np.float32)
            next_observations = np.array([next_observations], dtype=np.float32)
            dones = np.array([dones], dtype=np.float32)
        if isinstance(rewards, np.ndarray):
            self.observations = torch.from_numpy(observations.astype(np.float32))
            self.actions = torch.from_numpy(actions.astype(np.float32))
            self.rewards = torch.from_numpy(rewards.astype(np.float32))
            self.next_obs = torch.from_numpy(next_observations.astype(np.float32))
            self.dones = torch.from_numpy(dones.astype(np.int64))
        else:
            self.observations = observations
            self.actions = actions
            self.rewards = rewards
            self.next_obs = next_observations
            self.dones = dones

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

    # TODO: add device
    def __init__(self, capacity, keep_infos=False, flatten_infos=False):
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

        self.index += transition_batch.size
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
        return len(self.obs)

    def __bool__(self):
        return bool(len(self))

    def save(self, filename="buffer.pkl"):
        """Save the buffer to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)


class PrioritizedReplay(MightyReplay):
    """Prioritized Replay Buffer."""

    def __init__(
        self,
        capacity,
        alpha=1.0,
        beta=1.0,
        epsilon=1e-4,
        keep_infos=False,
        flatten_infos=False,
    ):
        """Initialize Buffer.

        :param capacity: Buffer size
        :param alpha: Priorization exponent
        :param beta: Bias exponent
        :param epsilon: Step size
        :param random_seed: Seed for sampling
        :param keep_infos: Keep the extra info dict. Required for some algorithms.
        :param flatten_infos: Make flat list from infos.
            Might be necessary, depending on info content.
        :return:
        """
        super().__init__(capacity, keep_infos, flatten_infos)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def add(self, transition_batch, metrics):
        """Add transition(s).

        :param transition_batch: Transition(s) to add
        :param metrics: Current metrics dict
        :return:
        """
        super().add(transition_batch, metrics)
        advantage = metrics["td_error"]
        advantage = np.power(np.abs(advantage) + self.epsilon, self.alpha)
        if len(self.advantages) == 0:
            self.advantages = torch.from_numpy(advantage)
        else:
            self.advantages = torch.cat((self.advantages, torch.from_numpy(advantage)))
        while len(self.advantages) > self.capacity:
            self.advantages.pop(0)

    def reset(self):
        """Reset the buffer."""
        super().reset()
        self.advantages = []

    def sample(self, batch_size=32):
        """Sample transitions."""
        probabilities = np.array(self.advantages) / sum(self.advantages)
        sample_weights = np.power(probabilities * len(self), -self.beta)
        sample_weights /= sample_weights.max()
        normalizer = 1 / sum(sample_weights)
        sample_weights = np.array([x * normalizer for x in sample_weights])
        # Get rid of rounding errors
        sample_weights[-1] = max(0, 1 - np.sum(sample_weights[0:-1]))

        batch_indices = self.rng.choice(
            np.arange(len(self)), size=batch_size, p=sample_weights
        )
        return TransitionBatch(
            self.obs[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_obs[batch_indices],
            self.dones[batch_indices],
        )
