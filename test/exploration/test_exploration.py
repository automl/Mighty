from __future__ import annotations

import pytest
import torch
from mighty.mighty_exploration import MightyExplorationPolicy
from utils import DummyModel


class TestPolicy:
    def get_policy(self, action=1):
        return MightyExplorationPolicy(algo="q", model=DummyModel(action=action))

    def test_exploration_func(self):
        with pytest.raises(NotImplementedError):
            self.get_policy().explore_func([0])

    @pytest.mark.parametrize(
        "state",
        [
            torch.tensor([0]),
            torch.tensor([0, 1]),
            torch.tensor([[0, 235, 67], [0, 1, 2]]),
        ],
    )
    def test_call(self, state):
        policy = self.get_policy()
        with pytest.raises(NotImplementedError):
            policy(state)

        greedy_actions, qvals = policy(state, evaluate=True, return_logp=True)
        assert all(
            greedy_actions == 1
        ), f"Greedy actions should be 1: {greedy_actions}///{qvals}"
        assert qvals.shape[-1] == 5, "Q-value shape should not be changed."
        assert len(qvals) == len(state), "Q-value length should not be changed."

        policy = self.get_policy(action=3)
        greedy_actions, qvals = policy(state, evaluate=True, return_logp=True)
        assert all(
            greedy_actions == 3
        ), f"Greedy actions should be 3: {greedy_actions}///{qvals}"
        assert qvals.shape[-1] == 5, "Q-value shape should not be changed."
        assert len(qvals) == len(state), "Q-value length should not be changed."
