from __future__ import annotations

import pickle as pkl
from pathlib import Path

import numpy as np
import pytest
import torch
from mighty.mighty_replay import MightyReplay, PrioritizedReplay, TransitionBatch

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
        assert isinstance(
            batch.observations, torch.Tensor
        ), "Observations were not a tensor."
        assert isinstance(batch.actions, torch.Tensor), "Actions were not a tensor."
        assert isinstance(batch.rewards, torch.Tensor), "Rewards were not a tensor."
        assert isinstance(
            batch.next_obs, torch.Tensor
        ), "Next observations were not a tensor."
        assert isinstance(batch.dones, torch.Tensor), "Dones were not a tensor."

        assert (
            len(batch.observations.shape) == 2
        ), f"Observation shape was not 2D: {batch.observations.shape}."
        assert (
            batch.observations.shape == batch.next_obs.shape
        ), "Observation shape was not equal to next observation shape."
        assert (
            batch.actions.shape == batch.rewards.shape
        ), "Action shape was not equal to reward shape."
        assert (
            batch.actions.shape == batch.dones.shape
        ), "Action shape was not equal to reward shape."
        assert len(batch.actions.shape) == len(batch.observations.shape) - 1, f"""Action shape was not one less than observation shape:
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
            assert (
                next_obs.numpy() in next_observations
            ), "Next observation was not in next_observations."
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
        assert (
            len(filled_replay) == size
        ), "Filled replay length was not equal to batch size."

        replay.add(batch, {})
        assert len(replay) == len(
            filled_replay
        ), "Replay length was not equal to batch size."
        assert (
            replay.index == filled_replay.index
        ), "Replay index was not equal to filled replay index."
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
        assert isinstance(
            minibatch, TransitionBatch
        ), "Minibatch was not a TransitionBatch."
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
        assert isinstance(
            minibatch, TransitionBatch
        ), "Minibatch was not a TransitionBatch."
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
        assert ~all(
            x == all_actions[0] for x in all_actions
        ), "All sampled batches were the same."

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
        assert (
            len(replay) == size * 2
        ), "Replay length was not doubled after doubling transitions."

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
        assert replay.capacity > len(
            replay
        ), "Replay capacity was not greater than length in non-full replay."
        assert (
            replay.index < replay.capacity
        ), "Replay index was not less than capacity in non-full replay."

        replay = self.get_replay(batch, size, full=True)
        assert replay.full is True, "Replay was not full."
        assert replay.capacity == len(
            replay
        ), "Replay capacity was not equal to length in full replay."
        assert (
            replay.index == replay.capacity
        ), "Replay index was not equal to capacity in full replay."

        second_batch = TransitionBatch(
            observations * 2, actions * 2, rewards * 2, next_observations * 2, dones * 2
        )
        replay.add(second_batch, {})
        assert (
            replay.full is True
        ), "Replay was not full anymore after adding more transitions."
        assert replay.capacity == len(
            replay
        ), "Replay capacity was not equal to length after adding more transitions."
        assert (
            replay.index == replay.capacity
        ), "Replay index was not equal to capacity after adding more transitions."
        for obs, act, rew, next_obs, done in second_batch:
            assert obs in replay.obs, f"Observation {obs} was not in replay."
            assert act in replay.actions, f"Action {act} was not in replay."
            assert rew in replay.rewards, f"Reward {rew} was not in replay."
            assert (
                next_obs in replay.next_obs
            ), f"Next observation {next_obs} was not in replay."
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
        assert (
            replay.capacity == loaded_replay.capacity
        ), "Replay capacity was not loaded correctly."
        assert (
            replay.index == loaded_replay.index
        ), "Replay index was not loaded correctly."
        assert torch.allclose(
            replay.obs, loaded_replay.obs
        ), "Replay observations were not loaded correctly."
        assert torch.allclose(
            replay.actions, loaded_replay.actions
        ), "Replay actions were not loaded correctly."
        assert torch.allclose(
            replay.rewards, loaded_replay.rewards
        ), "Replay rewards were not loaded correctly."
        assert torch.allclose(
            replay.next_obs, loaded_replay.next_obs
        ), "Replay next observations were not loaded correctly."
        assert torch.allclose(
            replay.dones, loaded_replay.dones
        ), "Replay dones were not loaded correctly."
        Path("test_replay.pkl").unlink()


class TestPrioritizedReplay:
    def get_replay(self, batch, size, full=False, empty=False):
        capacity = 100
        if full:
            capacity = size
        replay = PrioritizedReplay(capacity)
        if empty:
            return replay
        replay.add(batch, {"td_error": rng.random(size)})
        return replay

    def test_init(self):
        replay = PrioritizedReplay(100)
        assert replay.capacity == 100, "Replay capacity was not set correctly."
        assert not replay, "Replay was not empty."
        assert replay.index == 0, "Replay index was not 0."
        assert replay.epsilon, "Replay epsilon was not set correctly."
        assert replay.alpha, "Replay alpha was not set correctly."
        assert replay.beta, "Replay beta was not set correctly."
        assert replay.obs == [], "Replay observations were not empty."
        assert replay.actions == [], "Replay actions were not empty."
        assert replay.rewards == [], "Replay rewards were not empty."
        assert replay.next_obs == [], "Replay next observations were not empty."
        assert replay.dones == [], "Replay dones were not empty."
        assert replay.advantages == [], "Replay advantages were not empty."

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
        assert (
            len(filled_replay) == size
        ), "Filled replay length was not equal to batch size."

        td_errors = rng.random(size)
        replay.add(batch, {"td_error": td_errors})
        assert len(replay) == len(
            filled_replay
        ), "Replay length was not equal to batch size."
        assert (
            replay.index == filled_replay.index
        ), "Replay index was not equal to filled replay index."
        assert all(
            any(np.isclose(adv, td_errors, atol=1e-4)) for adv in replay.advantages
        ), f"Advantages were not added to replay: {replay.advantages}///{td_errors}."

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
        replay = self.get_replay(batch, size, empty=True)
        replay.add(batch, {"td_error": [0, 0, 1]})
        batchset = [replay.sample(batch_size=1) for _ in range(50)]
        all_actions = [act for batch in batchset for act in batch.actions]
        assert ~all(x == all_actions[0] for x in all_actions), """All sampled batches were different
            (even though probabilities should prevent this)."""

        replay = self.get_replay(batch, size, empty=True)
        replay.add(batch, {"td_error": [4, 4, 4]})
        batchset = [replay.sample(batch_size=1) for _ in range(10)]
        all_actions = [act for batch in batchset for act in batch.actions]
        assert ~all(x == all_actions[0] for x in all_actions), """All sampled batches were the same
            (although probabilities should prevent this)."""

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
        assert len(replay.advantages) > 0, "Replay advantages were empty."
        replay.reset()
        assert len(replay.advantages) == 0, "Replay advantages were not reset."
