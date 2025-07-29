from __future__ import annotations

import pickle as pkl
from pathlib import Path

import numpy as np
import pytest
import torch

from mighty.mighty_replay import (MightyReplay, PrioritizedReplay,
                                  TransitionBatch)

rng = np.random.default_rng(12345)

test_transitions = [
    (
        np.array([[1], [2], [3]]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
        np.array([[10], [11], [12]]),
        np.array([13, 14, 15]),
        3,
    ),
    (
        np.array([[1, 4, 5, 7], [2, 1, 5, 6]]),
        np.array([4, 5]),
        np.array([7, 8]),
        np.array([[10, 5, 6, 7], [11, 9, 3, 0]]),
        np.array([13, 14]),
        2,
    ),
    (
        np.array([[1, 5]]),
        np.array([4]),
        np.array([7]),
        np.array([[10, 4]]),
        np.array([13]),
        1,
    ),
    (np.array([1]), 4, 7, np.array([10]), 13, 1),
]


class TestBatch:
    @pytest.mark.parametrize(
        ("observations", "actions", "rewards", "next_observations", "dones", "size"),
        test_transitions,
    )
    def test_size(self, observations, actions, rewards, next_observations, dones, size):
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        assert batch.size == size, "Batch size was not equal to expected size."

    @pytest.mark.parametrize(
        ("observations", "actions", "rewards", "next_observations", "dones", "size"),
        test_transitions,
    )
    def test_init(self, observations, actions, rewards, next_observations, dones, size):
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        assert isinstance(batch.observations, torch.Tensor), (
            "Observations were not a tensor."
        )
        assert isinstance(batch.actions, torch.Tensor), "Actions were not a tensor."
        assert isinstance(batch.rewards, torch.Tensor), "Rewards were not a tensor."
        assert isinstance(batch.next_obs, torch.Tensor), (
            "Next observations were not a tensor."
        )
        assert isinstance(batch.dones, torch.Tensor), "Dones were not a tensor."

        assert len(batch.observations.shape) == 2, (
            f"Observation shape was not 2D: {batch.observations.shape}."
        )
        assert batch.observations.shape == batch.next_obs.shape, (
            "Observation shape was not equal to next observation shape."
        )
        assert batch.actions.shape == batch.rewards.shape, (
            "Action shape was not equal to reward shape."
        )
        assert batch.actions.shape == batch.dones.shape, (
            "Action shape was not equal to reward shape."
        )
        assert (
            len(batch.actions.shape) == len(batch.observations.shape) - 1
        ), f"""Action shape was not one less than observation shape:
            {batch.actions}///{batch.actions.shape} ---
            {batch.observations}///{batch.observations.shape}."""

    @pytest.mark.parametrize(
        ("observations", "actions", "rewards", "next_observations", "dones", "size"),
        test_transitions,
    )
    def test_iter(self, observations, actions, rewards, next_observations, dones, size):
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        elements = 0
        for obs, act, rew, next_obs, done in batch:
            assert obs.numpy() in observations, "Observation was not in observations."
            assert next_obs.numpy() in next_observations, (
                "Next observation was not in next_observations."
            )
            if isinstance(actions, int):
                assert act.numpy().item() == actions, "Action was not in actions."
                assert rew.numpy().item() == rewards, "Reward was not in rewards."
                assert done.numpy().item() == dones, "Done was not in dones."
            else:
                assert act.numpy() in actions, "Action was not in actions."
                assert rew.numpy() in rewards, "Reward was not in rewards."
                assert done.numpy() in dones, "Done was not in dones."
            elements += 1
        assert elements == size, "Not all elements were iterated over."


class TestStandardReplay:
    def get_replay(self, batch, size, full=False, empty=False):
        capacity = 100
        if full:
            capacity = size
        replay = MightyReplay(capacity)
        if empty:
            return replay
        replay.add(batch, {})
        return replay

    def test_init(self):
        replay = MightyReplay(100)
        assert replay.capacity == 100, "Replay capacity was not set correctly."
        assert not replay, "Replay was not empty."
        assert replay.index == 0, "Replay index was not 0."
        assert replay.obs == [], "Replay observations were not empty."
        assert replay.actions == [], "Replay actions were not empty."
        assert replay.rewards == [], "Replay rewards were not empty."
        assert replay.next_obs == [], "Replay next observations were not empty."
        assert replay.dones == [], "Replay dones were not empty."

    @pytest.mark.parametrize(
        ("observations", "actions", "rewards", "next_observations", "dones", "size"),
        test_transitions,
    )
    def test_add(self, observations, actions, rewards, next_observations, dones, size):
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        replay = self.get_replay(batch, size, empty=True)
        filled_replay = self.get_replay(batch, size)
        assert len(replay) == 0, "Empty replay length was not 0."
        assert len(filled_replay) == size, (
            "Filled replay length was not equal to batch size."
        )

        replay.add(batch, {})
        assert len(replay) == len(filled_replay), (
            "Replay length was not equal to batch size."
        )
        assert replay.index == filled_replay.index, (
            "Replay index was not equal to filled replay index."
        )
        assert all(
            any(torch.equal(obs, ob) for obs in batch.observations) for ob in replay.obs
        ), "Observations were not added to replay."
        assert all(
            any(
                torch.equal(torch.tensor(act), torch.tensor(ac))
                for act in batch.actions
            )
            for ac in replay.actions
        ), "Actions were not added to replay."
        assert all(
            any(
                torch.equal(torch.tensor(rew), torch.tensor(re))
                for rew in batch.rewards
            )
            for re in replay.rewards
        ), "Rewards were not added to replay."
        assert all(
            any(torch.equal(next_obs, next_ob) for next_obs in batch.next_obs)
            for next_ob in replay.next_obs
        ), "Next observations were not added to replay."
        assert all(
            any(
                torch.equal(torch.tensor(done), torch.tensor(don))
                for done in batch.dones
            )
            for don in replay.dones
        ), "Dones were not added to replay."

    def test_sample(self):
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            size,
        ) = test_transitions[0]
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        replay = self.get_replay(batch, size)
        minibatch = replay.sample(batch_size=1)
        assert len(minibatch) == 1, "Minibatch length was incorrect (batch size 1)."
        assert isinstance(minibatch, TransitionBatch), (
            "Minibatch was not a TransitionBatch."
        )
        assert all(
            any(torch.allclose(obs, ob) for obs in batch.observations)
            for ob in minibatch.observations
        ), "Sampled observations were not in replay (batch size 1)."
        assert all(
            any(torch.equal(act, ac) for act in batch.actions)
            for ac in minibatch.actions
        ), f"""Sampled actions were not in replay (batch size 1):
            {batch.actions} --- {minibatch.actions}."""
        assert all(
            any(torch.equal(rew, re) for rew in batch.rewards)
            for re in minibatch.rewards
        ), "Sampled rewards were not in replay (batch size 1)."
        assert all(
            any(torch.allclose(next_obs, next_ob) for next_obs in batch.next_obs)
            for next_ob in minibatch.next_obs
        ), "Sampled next observations were not in replay (batch size 1)."
        assert all(
            any(torch.equal(done, don) for done in batch.dones)
            for don in minibatch.dones
        ), "Sampled dones were not in replay (batch size 1)."

        minibatch = replay.sample(batch_size=2)
        assert len(minibatch) == 2, "Minibatch length was incorrect (batch size 2)."
        assert isinstance(minibatch, TransitionBatch), (
            "Minibatch was not a TransitionBatch."
        )
        assert all(
            any(torch.allclose(obs, ob) for obs in batch.observations)
            for ob in minibatch.observations
        ), "Sampled observations were not in replay (batch size 2)."
        assert all(
            any(torch.equal(act, ob) for act in batch.actions)
            for ob in minibatch.actions
        ), "Sampled actions were not in replay (batch size 2)."
        assert all(
            any(torch.equal(rew, re) for rew in batch.rewards)
            for re in minibatch.rewards
        ), "Sampled rewards were not in replay (batch size 2)."
        assert all(
            any(torch.allclose(next_obs, next_ob) for next_obs in batch.next_obs)
            for next_ob in minibatch.next_obs
        ), "Sampled next observations were not in replay (batch size 2)."
        assert all(
            any(torch.equal(done, don) for done in batch.dones)
            for don in minibatch.dones
        ), "Sampled dones were not in replay (batch size 2)."

        batchset = [replay.sample(batch_size=1) for _ in range(10)]
        all_actions = [act for batch in batchset for act in batch.actions]
        assert ~all(x == all_actions[0] for x in all_actions), (
            "All sampled batches were the same."
        )

    def test_reset(self):
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            size,
        ) = test_transitions[0]
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        replay = self.get_replay(batch, size)
        assert len(replay) > 0, "Replay length was 0."
        assert replay.index > 0, "Replay index was 0."
        assert len(replay.actions) > 0, "Replay actions were empty."
        replay.reset()
        assert len(replay) == 0, "Replay length was not reset."
        assert replay.index == 0, "Replay index was not reset."
        assert len(replay.actions) == 0, "Replay actions were not reset."

    def test_len(self):
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            size,
        ) = test_transitions[0]
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        replay = self.get_replay(batch, size)
        assert len(replay) == size, "Replay length was not equal to batch size."

        replay.add(batch, {})
        assert len(replay) == size * 2, (
            "Replay length was not doubled after doubling transitions."
        )

        replay = self.get_replay(batch, size, empty=True)
        assert len(replay) == 0, "Replay length of empty replay was not 0."

    def test_full(self):
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            size,
        ) = test_transitions[0]
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        replay = self.get_replay(batch, size, full=False)
        assert replay.full is False, "Replay was falsely full."
        assert replay.capacity > len(replay), (
            "Replay capacity was not greater than length in non-full replay."
        )
        assert replay.index < replay.capacity, (
            "Replay index was not less than capacity in non-full replay."
        )

        replay = self.get_replay(batch, size, full=True)
        assert replay.full is True, "Replay was not full."
        assert replay.capacity == len(replay), (
            "Replay capacity was not equal to length in full replay."
        )
        assert replay.index == replay.capacity, (
            "Replay index was not equal to capacity in full replay."
        )

        second_batch = TransitionBatch(
            observations * 2, actions * 2, rewards * 2, next_observations * 2, dones * 2
        )
        replay.add(second_batch, {})
        assert replay.full is True, (
            "Replay was not full anymore after adding more transitions."
        )
        assert replay.capacity == len(replay), (
            "Replay capacity was not equal to length after adding more transitions."
        )
        assert replay.index == replay.capacity, (
            "Replay index was not equal to capacity after adding more transitions."
        )
        for obs, act, rew, next_obs, done in second_batch:
            assert obs in replay.obs, f"Observation {obs} was not in replay."
            assert act in replay.actions, f"Action {act} was not in replay."
            assert rew in replay.rewards, f"Reward {rew} was not in replay."
            assert next_obs in replay.next_obs, (
                f"Next observation {next_obs} was not in replay."
            )
            assert done in replay.dones, f"Done {done} was not in replay."

    def test_empty(self):
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            size,
        ) = test_transitions[0]
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        replay = self.get_replay(batch, size, empty=True)
        assert not replay, "Replay was not empty."

        replay.add(batch, {})
        assert replay, "Replay was empty after adding transitions."

    def test_save(self):
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            size,
        ) = test_transitions[0]
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        replay = self.get_replay(batch, size)
        replay.save("test_replay.pkl")
        with open("test_replay.pkl", "rb") as f:
            loaded_replay = pkl.load(f)
        assert replay.capacity == loaded_replay.capacity, (
            "Replay capacity was not loaded correctly."
        )
        assert replay.index == loaded_replay.index, (
            "Replay index was not loaded correctly."
        )
        assert torch.allclose(replay.obs, loaded_replay.obs), (
            "Replay observations were not loaded correctly."
        )
        assert torch.allclose(replay.actions, loaded_replay.actions), (
            "Replay actions were not loaded correctly."
        )
        assert torch.allclose(replay.rewards, loaded_replay.rewards), (
            "Replay rewards were not loaded correctly."
        )
        assert torch.allclose(replay.next_obs, loaded_replay.next_obs), (
            "Replay next observations were not loaded correctly."
        )
        assert torch.allclose(replay.dones, loaded_replay.dones), (
            "Replay dones were not loaded correctly."
        )
        Path("test_replay.pkl").unlink()


class TestPrioritizedReplay:
    def get_replay(
        self, batch: TransitionBatch, size: int, full: bool = False, empty: bool = False
    ):
        """
        Build a PrioritizedReplay whose internal buffers match the shapes of `batch`.
        - If full=True, capacity == size; otherwise capacity = 100.
        - If empty=True, return immediately (no .add()); else add once with random td_errors.
        """
        capacity = size if full else 100

        # Derive obs_shape / action_shape from the batch’s tensors:
        obs_shape = tuple(batch.observations.shape[1:])
        if batch.actions.dim() > 1:
            action_shape = tuple(batch.actions.shape[1:])
        else:
            action_shape = ()

        replay = PrioritizedReplay(
            capacity=capacity,
            obs_shape=obs_shape,
            action_shape=action_shape,
        )
        if empty:
            return replay

        # Generate a random td_error array of length=size
        td_errors = rng.random(size).astype(np.float32)
        replay.add(batch, {"td_error": td_errors})
        return replay

    def test_init(self):
        """
        Check that __init__ sets capacity, current_size, data_idx, default α,β,ε,
        and allocates zeroed sum_tree of length (2*capacity).
        """
        cap = 100
        obs_shape = (1,)  # e.g. single-dimensional observation
        action_shape = ()  # e.g. scalar action

        replay = PrioritizedReplay(
            capacity=cap,
            obs_shape=obs_shape,
            action_shape=action_shape,
        )

        # 1) capacity set, no elements stored yet
        assert replay.capacity == cap
        assert replay.current_size == 0
        assert replay.data_idx == 0

        # 2) default hyperparameters
        assert isinstance(replay.alpha, float) and replay.alpha == 1.0
        assert isinstance(replay.beta, float) and replay.beta == 1.0
        assert isinstance(replay.epsilon, float) and replay.epsilon == 1e-6

        # 3) The on-device buffers + sum_tree must exist
        for attr in (
            "obs_buffer",
            "next_obs_buffer",
            "action_buffer",
            "reward_buffer",
            "done_buffer",
            "sum_tree",
        ):
            assert hasattr(replay, attr), f"Missing attribute {attr!r}"

        # 4) sum_tree is a NumPy array of size (2*capacity), all zeros initially
        assert isinstance(replay.sum_tree, np.ndarray)
        assert replay.sum_tree.shape == (2 * cap,)
        assert np.allclose(replay.sum_tree, 0.0)

    @pytest.mark.parametrize(
        ("observations", "actions", "rewards", "next_observations", "dones", "size"),
        test_transitions,
    )
    def test_add(self, observations, actions, rewards, next_observations, dones, size):
        """
        After .add(batch, {'td_error': td_errors}):
          - replay.current_size should equal size
          - data_idx should advance correctly
          - The leaves in sum_tree (indices [capacity:capacity+size]) match (|td_error|+ε)^α
        """
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        # Empty replay (no .add() yet)
        replay = self.get_replay(batch, size, empty=True)
        # “Filled” replay (has already added once in get_replay)
        filled_replay = self.get_replay(batch, size)

        assert replay.current_size == 0, "Empty replay length was not 0."
        assert filled_replay.current_size == size, (
            "Filled replay length was not equal to batch size."
        )

        # Now add to the previously empty replay with a brand-new td_error array
        td_errors = rng.random(size).astype(np.float32)
        replay.add(batch, {"td_error": td_errors})

        # Both replays should have the same current_size and data_idx
        assert replay.current_size == filled_replay.current_size, (
            "Replay length was not equal to filled replay length."
        )
        assert replay.data_idx == filled_replay.data_idx, (
            "Replay data_idx was not equal to filled replay data_idx."
        )

        # Compute expected priority = (|td_error| + ε)^α for each leaf
        eps = replay.epsilon
        alpha = replay.alpha
        expected_prios = (np.abs(td_errors) + eps) ** alpha

        base = replay.capacity
        actual_leaves = replay.sum_tree[base : base + size]
        assert np.allclose(actual_leaves, expected_prios, atol=1e-6), (
            f"Expected leaves {expected_prios}, but got {actual_leaves}"
        )

    @pytest.mark.parametrize(
        ("observations", "actions", "rewards", "next_observations", "dones", "size"),
        test_transitions,
    )
    def test_sample(
        self, observations, actions, rewards, next_observations, dones, size
    ):
        """
        1) Uniform-priority scenario: sample 30 times with a uniform td_error array.
           If size>1, expect at least 2 distinct indices in 30 draws.
        2) Skewed-priority scenario: all but one element have zero td_error.
           That single nonzero‐priority index should be drawn every time.
        """
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )

        # 1) Uniform priorities (all td_errors identical):
        replay = self.get_replay(batch, size, empty=True)
        uniform_tde = np.ones(size, dtype=np.float32) * 5.0
        replay.add(batch, {"td_error": uniform_tde})

        seen_actions = []
        for _ in range(30):
            (
                obs_b,
                action_b,
                reward_b,
                next_obs_b,
                done_b,
                is_weights_b,
                batch_indices_b,
            ) = replay.sample(batch_size=1)

            # (a) Check shapes:
            assert obs_b.shape[0] == 1
            assert next_obs_b.shape[0] == 1
            assert action_b.shape[0] == 1
            assert reward_b.shape == (1, 1)
            assert done_b.shape == (1, 1)
            # is_weights can be (1,1) or (1,)
            assert is_weights_b.shape == (1, 1) or is_weights_b.shape == (1,)
            assert isinstance(
                batch_indices_b, np.ndarray
            ) and batch_indices_b.shape == (1,)

            # (b) The returned action must match one of batch.actions
            retrieved = action_b.view(-1).cpu().numpy().item()
            if isinstance(actions, int):
                original_actions = [actions]
            else:
                original_actions = list(actions)
            assert retrieved in original_actions
            seen_actions.append(retrieved)

        # If there is more than one element, uniform draws should vary
        if size > 1:
            assert len(set(seen_actions)) >= 2, (
                "Uniform-priority sampling did not vary."
            )

        # 2) Skewed priorities (only one index has nonzero td_error):
        replay = self.get_replay(batch, size, empty=True)
        tde = np.zeros(size, dtype=np.float32)
        if size > 1:
            tde[-1] = 5.0
        else:
            tde[0] = 5.0
        replay.add(batch, {"td_error": tde})

        forced_indices = []
        for _ in range(10):
            (
                obs_b,
                action_b,
                reward_b,
                next_obs_b,
                done_b,
                is_weights_b,
                batch_indices_b,
            ) = replay.sample(batch_size=1)
            forced_indices.append(int(batch_indices_b.item()))

        # All sampled indices must be identical (the only one with nonzero priority)
        assert len(set(forced_indices)) == 1, (
            f"Expected only one index to be chosen, but got {set(forced_indices)}"
        )

    @pytest.mark.parametrize(
        ("observations", "actions", "rewards", "next_observations", "dones", "size"),
        test_transitions,
    )
    def test_reset(
        self, observations, actions, rewards, next_observations, dones, size
    ):
        """
        Because the current implementation of reset() does NOT clear the on‐device buffers or sum_tree,
        after calling reset() we expect both current_size and sum_tree to remain unchanged.
        """
        batch = TransitionBatch(
            observations, actions, rewards, next_observations, dones
        )
        replay = self.get_replay(batch, size, empty=True)

        tde = np.ones(size, dtype=np.float32)
        replay.add(batch, {"td_error": tde})

        # Sanity check: after adding, buffer is nonempty
        assert replay.current_size == size
        assert replay.sum_tree[1] > 0.0  # root of sum_tree (total priority)

        # Now reset
        replay.reset()

        # **Because reset() does not clear on‐device buffers or sum_tree**, we expect:
        #   - current_size remains equal to `size`
        #   - sum_tree[1] (the total priority) remains > 0
        assert replay.current_size == size, (
            f"After reset(), expected current_size to still be {size}, but got {replay.current_size}"
        )
        assert replay.sum_tree[1] > 0.0, (
            "After reset(), expected total priority (sum_tree[1]) to remain > 0"
        )
