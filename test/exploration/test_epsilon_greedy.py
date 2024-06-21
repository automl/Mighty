from __future__ import annotations

import numpy as np
import pytest
import torch
from mighty.mighty_exploration import EpsilonGreedy
from utils import DummyModel


class TestEpsilonGreedy:
    def get_policy(self, epsilon=0.1):
        return EpsilonGreedy(algo="q", model=DummyModel(), epsilon=epsilon)

    @pytest.mark.parametrize(
        "state",
        [
            torch.tensor([[0, 1], [0, 1]]),
            torch.tensor([[0, 235, 67], [0, 1, 2]]),
            torch.tensor(
                [[0, 235, 67], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
            ),
        ],
    )
    def test_exploration_func(self, state):
        policy = self.get_policy(epsilon=0.0)
        actions, qvals = policy.explore_func(state)
        greedy_actions, greedy_qvals = policy.sample_action(state)
        assert len(actions) == len(state), "Action should be predicted per state."
        assert all(
            a == g for g in greedy_actions for a in actions
        ), f"Actions should match greedy: {actions}///{greedy_actions}"
        assert torch.equal(
            qvals, greedy_qvals
        ), f"Q-values should match greedy: {qvals}///{greedy_qvals}"

        policy = self.get_policy(epsilon=0.5)
        actions = np.array(
            [policy.explore_func(state)[0] for _ in range(100)]
        ).flatten()
        assert (
            sum([a == 1 for a in actions]) / (100 * len(state)) > 0.5
        ), "Actions should match greedy at least in half of cases."
        assert (
            sum([a == 1 for a in actions]) / (100 * len(state)) < 0.8
        ), "Actions should match greedy in less than 4/5 of cases."

        policy = self.get_policy(epsilon=np.linspace(0, 1, len(state)))
        actions = np.array([policy.explore_func(state)[0] for _ in range(100)])
        assert all(actions[:, 0] == 1), "Low index actions should match greedy."
        assert (
            sum(actions[:, -1] == 1) / 100 < 0.33
        ), "High index actions should not match greedy more than 1/3 of the time."

    @pytest.mark.parametrize(
        "state",
        [
            torch.tensor([[0, 1], [0, 1]]),
            torch.tensor([[0, 235, 67], [0, 1, 2]]),
            torch.tensor(
                [[0, 235, 67], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
            ),
        ],
    )
    def test_multiple_epsilons(self, state):
        """Test multiple epsilon values."""
        policy = self.get_policy(epsilon=[0.1, 0.5])
        assert np.all(policy.epsilon == [0.1, 0.5]), "Epsilon should be [0.1, 0.5]."
        action, _ = policy.explore_func(state)
        assert len(action) == len(state.numpy()), f"Action should be predicted per state: len({action}) != len({state.numpy()})."
