from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mighty.mighty_replay import TransitionBatch
from mighty.mighty_update import (
    ClippedDoubleQLearning,
    DoubleQLearning,
    QLearning,
    SPRQLearning,
)

RANDOM = 0.5
rng = np.random.default_rng(12345)

batch = {
    "observations": rng.random(size=(32, 2)),
    "actions": rng.integers(0, high=5, size=32),
    "rewards": rng.random(32),
    "next_observations": rng.random(size=(32, 2)),
    "dones": rng.random(32) < RANDOM,
}
batch = TransitionBatch(**batch)


class DummyModel(torch.nn.Module):
    """Dummy model for testing Q-learning."""

    def __init__(self, initial_weights=0, initial_biases=0):
        """Initialize the model."""
        super().__init__()
        self.layer = torch.nn.Linear(2, 5)
        self.layer.weight.data.fill_(initial_weights)
        self.layer.bias.data.fill_(initial_biases)

    def __call__(self, s):
        """Forward pass."""
        return self.layer(s)


class TestQLearning:
    """Test Q-learning update."""

    def get_update(self, initial_weights=0, initial_biases=0):
        """Return an instance of QLearning and a dummy model."""
        return QLearning(
            model=DummyModel(initial_weights, initial_biases), gamma=0.99
        ), DummyModel(initial_weights, initial_biases)

    @pytest.mark.parametrize(
        ("weight", "bias"), [(0, 0), (3, 0), (20, 1), (200, 100), (4, 4)]
    )
    def test_update(self, weight, bias):
        """Test Q-learning update."""
        update, model = self.get_update(initial_weights=weight, initial_biases=bias)
        checked_model = deepcopy(model)
        assert torch.allclose(
            model.layer.weight, checked_model.layer.weight
        ), "Wrong initial weights."
        assert torch.allclose(
            model.layer.bias, checked_model.layer.bias
        ), "Wrong initial biases."

        preds, targets = update.get_targets(batch, model)
        loss_stats = update.apply_update(preds, targets)

        optimizer = torch.optim.Adam(params=checked_model.parameters())
        optimizer.zero_grad()
        preds, targets = update.get_targets(batch, checked_model)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()

        assert np.isclose(
            loss_stats["Q-Update/loss"], loss.detach().numpy().item(), atol=1e-6
        ), "Wrong loss after update."
        assert torch.allclose(
            model.layer.weight, checked_model.layer.weight, atol=1e-3
        ), "Wrong weights after update"
        assert torch.allclose(
            model.layer.bias, checked_model.layer.bias, atol=1e-3
        ), "Wrong biases after update."

    def test_get_targets(self):
        """Test get_targets method."""
        update, model = self.get_update()
        preds, targets = update.get_targets(batch, model)
        assert preds.shape == (32, 1), "Wrong shape for predictions."
        assert targets.shape == (32, 1), "Wrong shape for targets."
        assert sum(preds) == 0, "Wrong initial predictions (weight 0)."
        correct_targets = (batch.rewards + (~batch.dones) * 0.99 * 0).unsqueeze(-1)
        assert torch.equal(
            targets.detach(), torch.as_tensor(correct_targets, dtype=torch.float32)
        ), "Wrong targets (weight 0)."

        update, model = self.get_update(initial_weights=3)
        preds, targets = update.get_targets(batch, model)
        assert torch.allclose(
            preds.type(torch.float32).detach(),
            torch.mul(batch.observations, 3).sum(axis=1).unsqueeze(-1),
        ), "Wrong initial predictions (weight 3)."
        correct_targets = (
            batch.rewards
            + (~batch.dones)
            * 0.99
            * model(torch.as_tensor(batch.next_obs, dtype=torch.float32)).max(1)[0]
        ).unsqueeze(-1)
        assert torch.allclose(
            targets.detach(), torch.as_tensor(correct_targets, dtype=torch.float32)
        ), "Wrong targets (weight 3)."

    def test_td_error(self):
        """Test TD error computation."""
        update, model = self.get_update()
        preds, targets = update.get_targets(batch, model)
        td_error = update.td_error(batch, model)
        assert td_error.shape == (32,), "Wrong shape for TD error."
        assert torch.allclose(
            td_error,
            torch.as_tensor((preds - targets).pow(2).detach().mean(axis=1)),
            atol=1e-6,
        ), "Wrong TD error."


class TestDoubleQLearning:
    """Test double Q-learning update."""

    def get_update(self, initial_weights=0, initial_biases=0):
        """Return an instance of DoubleQLearning and two dummy models."""
        return (
            DoubleQLearning(
                model=DummyModel(initial_weights, initial_biases), gamma=0.99
            ),
            DummyModel(initial_weights, initial_biases),
            DummyModel(initial_weights + 1, initial_biases),
        )

    def test_get_targets(self):
        """Test get_targets method."""
        update, model, target = self.get_update()
        preds, targets = update.get_targets(batch, model, target)
        assert preds.shape == (32, 1), f"Wrong shape for predictions: {preds.shape}"
        assert targets.shape == (32, 1), f"Wrong shape for targets: {targets.shape}"
        assert sum(preds) == 0, "Wrong initial predictions (weight 0)."
        correct_targets = batch.rewards.unsqueeze(-1) + (
            ~batch.dones.unsqueeze(-1)
        ) * 0.99 * batch.next_obs.sum(axis=1).unsqueeze(-1)
        assert torch.allclose(
            targets.detach(), correct_targets.type(torch.float32)
        ), "Wrong targets (weight 0)."

        update, model, target = self.get_update(initial_weights=3)
        preds, targets = update.get_targets(batch, model, target)
        assert torch.allclose(
            preds.type(torch.float32).detach(),
            torch.mul(batch.observations, 3).sum(axis=1).unsqueeze(-1),
        ), "Wrong initial predictions (weight 3)."
        correct_targets = (
            batch.rewards
            + (~batch.dones) * 0.99 * torch.mul(batch.next_obs, 4).sum(axis=1)
        ).unsqueeze(-1)
        assert torch.allclose(
            targets.detach(), torch.as_tensor(correct_targets, dtype=torch.float32)
        ), "Wrong targets (weight 3)."


class TestClippedDoubleQLearning:
    """Test clipped double Q-learning update."""

    def get_update(self, initial_weights=0, initial_biases=0):
        """Return an instance of ClippedDoubleQLearning and two dummy models."""
        return (
            ClippedDoubleQLearning(
                model=DummyModel(initial_weights, initial_biases), gamma=0.99
            ),
            DummyModel(initial_weights, initial_biases),
            DummyModel(initial_weights + 1, initial_biases),
        )

    def test_get_targets(self):
        """Test get_targets method."""
        update, model, target = self.get_update()
        preds, targets = update.get_targets(batch, model, target)
        assert preds.shape == (32, 1), f"Wrong shape for predictions: {preds.shape}"
        assert targets.shape == (32, 1), f"Wrong shape for targets: {targets.shape}"
        assert sum(preds) == 0, "Wrong initial predictions (weight 0)."
        correct_targets = batch.rewards.unsqueeze(-1) + (
            ~batch.dones.unsqueeze(-1)
        ) * 0.99 * torch.minimum(
            batch.next_obs.sum(axis=1), torch.zeros(batch.next_obs.shape).sum(axis=1)
        ).unsqueeze(-1)
        assert torch.allclose(
            targets.detach(), correct_targets.type(torch.float32)
        ), "Wrong targets (weight 0)."

        update, model, target = self.get_update(initial_weights=3)
        preds, targets = update.get_targets(batch, model, target)
        assert torch.allclose(
            preds.type(torch.float32).detach(),
            torch.mul(batch.observations, 3).sum(axis=1).unsqueeze(-1),
        ), "Wrong initial predictions (weight 3)."
        correct_targets = (
            batch.rewards
            + (~batch.dones)
            * 0.99
            * torch.minimum(
                torch.mul(batch.next_obs, 4).sum(axis=1),
                torch.mul(batch.next_obs, 3).sum(axis=1),
            )
        ).unsqueeze(-1)
        assert torch.allclose(
            targets.detach(), torch.as_tensor(correct_targets, dtype=torch.float32)
        ), "Wrong targets (weight 3)."


class TestSPRQLearning:
    """Test SPR Q-learning update."""

    def get_update(self):
        """Return an instance of SPR Q-learning."""
        return SPRQLearning(model=DummyModel(), gamma=0.99)

    def test_update(self):
        """Test SPR Q-learning update."""
