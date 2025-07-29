from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from mighty.mighty_replay.mighty_replay_buffer import (MightyReplay,
                                                       TransitionBatch)


class PrioritizedReplay(MightyReplay):
    """Much faster Prioritized Replay using a sum-tree + on-device storage."""

    def __init__(
        self,
        capacity,
        alpha=1.0,
        beta=1.0,
        epsilon=1e-6,
        device="cpu",
        keep_infos=False,
        flatten_infos=False,
        obs_shape: Tuple[int, ...] | list[int] = None,
        action_shape: Tuple[int, ...] | list[int] = None,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.device = torch.device(device)

        super().__init__(capacity, keep_infos, flatten_infos, device)

        # 1) Buffers for transitions, stored **on‐device**:
        self.obs_buffer = torch.zeros((capacity, *obs_shape), device=self.device)
        self.next_obs_buffer = torch.zeros((capacity, *obs_shape), device=self.device)
        self.action_buffer = torch.zeros((capacity, *action_shape), device=self.device)
        self.reward_buffer = torch.zeros(
            (capacity,), dtype=torch.float32, device=self.device
        )
        self.done_buffer = torch.zeros(
            (capacity,), dtype=torch.int32, device=self.device
        )

        # 2) Sum‐tree arrays (size 2*capacity), stored on CPU (float32).
        #    The leaves (indices [capacity:2*capacity]) store individual priorities.
        #    Internal nodes store sums over children.
        self.tree_size = 2 * capacity
        self.sum_tree = np.zeros((self.tree_size,), dtype=np.float32)
        self.data_idx = 0  # next index to overwrite (ring buffer)
        self.current_size = 0  # how many valid items are in the buffer

    def _propagate(self, idx, change):
        """Propagate a priority change up the tree."""
        parent = idx // 2
        while parent >= 1:
            self.sum_tree[parent] += change
            parent //= 2

    def _retrieve(self, idx, s):
        """Find the leaf index for sum s."""
        left = 2 * idx
        right = left + 1
        if left >= self.tree_size:
            return idx
        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])

    def _total_priority(self):
        """Total priority (root of sum‐tree)."""
        return self.sum_tree[1]

    def add(self, transition_batch: TransitionBatch, metrics: dict) -> None:
        """Add a single transition with the computed td_error."""
        # 1) store transition in buffers
        td_errors = metrics.get("td_error", None)
        assert td_errors is not None, "metrics must contain 'td_error'"

        # Convert td_errors to a numpy array (length B).
        td_errors = np.asarray(td_errors, dtype=np.float32)
        assert td_errors.ndim == 1, "metrics['td_error'] must be 1D numpy array"
        batch_size = td_errors.shape[0]

        # Extract everything from transition_batch:
        obs_batch = transition_batch.observations
        act_batch = transition_batch.actions
        rew_batch = transition_batch.rewards
        next_obs_batch = transition_batch.next_obs
        done_batch = transition_batch.dones

        # We’ll write each transition individually into the ring buffer.
        for i in range(batch_size):
            idx = self.data_idx

            # 1) Store transition i
            o = obs_batch[i]
            n_o = next_obs_batch[i]
            a = act_batch[i]
            r = rew_batch[i].item()  # scalar
            d = done_batch[i].item()  # 0/1

            # Copy into on‐device buffers
            self.obs_buffer[idx].copy_(o.to(self.device))
            self.next_obs_buffer[idx].copy_(n_o.to(self.device))
            self.action_buffer[idx].copy_(a.to(self.device))
            self.reward_buffer[idx] = float(r)
            self.done_buffer[idx] = int(d)

            # 2) Compute new priority
            td_err_val = float(td_errors[i])
            priority = (abs(td_err_val) + self.epsilon) ** self.alpha

            # 3) Write into sum‐tree leaf
            leaf_idx = idx + self.capacity
            change = priority - self.sum_tree[leaf_idx]
            self.sum_tree[leaf_idx] = priority
            self._propagate(leaf_idx, change)

            # 4) Advance ring‐buffer pointers
            self.data_idx = (self.data_idx + 1) % self.capacity
            if self.current_size < self.capacity:
                self.current_size += 1

    def sample(self, batch_size):
        """Sample a batch of transitions and return importance weights + indices."""
        batch_indices = np.empty((batch_size,), dtype=np.int32)
        priorities = np.empty((batch_size,), dtype=np.float32)
        total_prio = self._total_priority()
        segment = total_prio / float(batch_size)

        # 1) Stratified sampling
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            leaf = self._retrieve(1, s)  # leaf index in [capacity..2*capacity-1]
            data_idx = leaf - self.capacity  # ring-buffer index
            batch_indices[i] = data_idx
            priorities[i] = self.sum_tree[leaf]

        # 2) Compute importance-sampling weights
        N = float(self.current_size)
        probs = priorities / total_prio  # shape=(batch_size,)
        is_weights_np = (N * probs) ** (-self.beta)  # shape=(batch_size,)
        is_weights_np /= is_weights_np.max()  # normalize

        # 3) Gather transitions on-device
        idxs = batch_indices.tolist()
        obs_batch = self.obs_buffer[idxs]  # (B, *obs_shape)
        action_batch = self.action_buffer[idxs]  # (B, *action_shape)
        reward_batch = self.reward_buffer[idxs].unsqueeze(-1)  # (B, 1)
        next_obs_batch = self.next_obs_buffer[idxs]  # (B, *obs_shape)
        done_batch = self.done_buffer[idxs].unsqueeze(-1)  # (B, 1)

        # 4) Convert is_weights → torch.Tensor on self.device
        is_weights = (
            torch.from_numpy(is_weights_np).to(self.device).unsqueeze(-1)
        )  # (B, 1)

        return (
            obs_batch,
            action_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
            is_weights,
            batch_indices,  # to call update_priorities() later
        )

    def update_priorities(self, indices, new_td_errors):
        """After learning, update the priorities for the sampled indices."""
        for data_idx, td_err in zip(indices, new_td_errors):
            new_prio = (abs(float(td_err)) + self.epsilon) ** self.alpha
            leaf = data_idx + self.capacity
            change = new_prio - self.sum_tree[leaf]
            self.sum_tree[leaf] = new_prio
            self._propagate(leaf, change)
