from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mighty.mighty_update.sac_update import SACUpdate


class DummySACModel(nn.Module):
    """Dummy SAC model for testing."""

    def __init__(self, obs_dim=4, action_dim=2, initial_weights=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_size = action_dim  # For compatibility with your SACUpdate

        # Policy network (outputs mean and log_std)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2),  # mean + log_std
        )

        # Q-networks (take state-action pairs)
        self.q_net1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.q_net2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # Target Q-networks (copies of Q-networks)
        self.target_q_net1 = deepcopy(self.q_net1)
        self.target_q_net2 = deepcopy(self.q_net2)

        # Initialize weights
        self._init_weights(initial_weights)

    def _init_weights(self, weight_val):
        """Initialize all weights to a specific value."""
        for param in self.parameters():
            param.data.fill_(weight_val)

    def forward(self, obs):
        """
        Forward pass through policy network.
        Returns: (action, z_raw, mean, log_std)
        """
        policy_output = self.policy_net(obs)
        mean = policy_output[..., : self.action_dim]
        log_std = policy_output[..., self.action_dim :]

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        # Sample raw action (pre-tanh)
        z = torch.randn_like(mean) * std + mean

        # Apply tanh to get bounded action
        action = torch.tanh(z)

        return action, z, mean, log_std

    def policy_log_prob(self, z, mean, log_std):
        """
        Calculate log probability of action given raw pre-tanh action z.
        """
        # Log probability from normal distribution
        log_prob = -0.5 * (
            ((z - mean) / log_std.exp()).pow(2) + 2 * log_std + np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Correction for tanh squashing
        log_prob -= torch.log(1 - torch.tanh(z).pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return log_prob


class DummyTransitionBatch:
    """Dummy transition batch for SAC testing."""

    def __init__(self, batch_size=32, obs_dim=4, action_dim=2):
        self.batch_size = batch_size
        self.observations = torch.randn(batch_size, obs_dim)
        self.actions = torch.randn(batch_size, action_dim)  # Continuous actions
        self.rewards = torch.randn(batch_size)
        self.next_obs = torch.randn(batch_size, obs_dim)
        self.dones = torch.randint(0, 2, (batch_size,)).float()


class TestSACUpdate:
    """Test SAC update mechanism."""

    def get_update_and_model(self, initial_weights=0.0, **sac_kwargs):
        """Create SAC update instance and model for testing."""
        model = DummySACModel(initial_weights=initial_weights)

        # Default SAC parameters for testing
        default_kwargs = {
            "policy_lr": 0.001,
            "q_lr": 0.001,
            "tau": 0.005,
            "alpha": 0.2,
            "gamma": 0.99,
            "auto_alpha": True,
            "alpha_lr": 3e-4,
        }
        default_kwargs.update(sac_kwargs)

        update = SACUpdate(model, **default_kwargs)
        return update, model

    def test_initialization(self):
        """Test SAC update initialization."""
        update, model = self.get_update_and_model()

        # Check optimizers
        assert hasattr(update, "policy_optimizer")
        assert hasattr(update, "q_optimizer1")
        assert hasattr(update, "q_optimizer2")

        # Check auto-alpha setup
        assert hasattr(update, "log_alpha")
        assert hasattr(update, "alpha_optimizer")
        assert hasattr(update, "target_entropy")
        assert update.target_entropy == -float(model.action_dim)

        # Check hyperparameters
        assert update.gamma == 0.99
        assert update.tau == 0.005
        assert update.auto_alpha

    def test_initialization_without_auto_alpha(self):
        """Test SAC initialization with fixed alpha."""
        update, model = self.get_update_and_model(auto_alpha=False, alpha=0.1)

        assert not update.auto_alpha
        # NOTE: Current implementation hardcodes alpha=0.2 when auto_alpha=False
        assert update.alpha == 0.1
        assert not hasattr(update, "log_alpha")
        assert not hasattr(update, "alpha_optimizer")

    def test_basic_update(self):
        """Test basic SAC update functionality."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch()

        # Store initial parameters
        initial_policy_params = [p.clone() for p in model.policy_net.parameters()]
        initial_q1_params = [p.clone() for p in model.q_net1.parameters()]
        initial_q2_params = [p.clone() for p in model.q_net2.parameters()]
        initial_alpha = update.log_alpha.clone()

        # Run update
        metrics = update.update(batch)

        # Check that parameters changed
        policy_changed = any(
            not torch.allclose(p1, p2, atol=1e-6)
            for p1, p2 in zip(initial_policy_params, model.policy_net.parameters())
        )
        q1_changed = any(
            not torch.allclose(p1, p2, atol=1e-6)
            for p1, p2 in zip(initial_q1_params, model.q_net1.parameters())
        )
        q2_changed = any(
            not torch.allclose(p1, p2, atol=1e-6)
            for p1, p2 in zip(initial_q2_params, model.q_net2.parameters())
        )
        alpha_changed = not torch.allclose(initial_alpha, update.log_alpha, atol=1e-6)

        assert policy_changed, "Policy parameters should change after update"
        assert q1_changed, "Q1 parameters should change after update"
        assert q2_changed, "Q2 parameters should change after update"
        assert alpha_changed, "Alpha should change when auto_alpha=True"

        # Check metrics
        required_metrics = [
            "q_loss1",
            "q_loss2",
            "policy_loss",
            "td_error1",
            "td_error2",
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), (
                f"Metric {metric} should be numeric"
            )
            assert np.isfinite(metrics[metric]), f"Metric {metric} should be finite"

    def test_target_network_updates(self):
        """Test that target networks are updated via polyak averaging."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch()

        # Store initial target parameters
        initial_target_q1 = [p.clone() for p in model.target_q_net1.parameters()]
        initial_target_q2 = [p.clone() for p in model.target_q_net2.parameters()]

        # Run update
        update.update(batch)

        # Check that target networks changed
        target_q1_changed = any(
            not torch.allclose(p1, p2, atol=1e-6)
            for p1, p2 in zip(initial_target_q1, model.target_q_net1.parameters())
        )
        target_q2_changed = any(
            not torch.allclose(p1, p2, atol=1e-6)
            for p1, p2 in zip(initial_target_q2, model.target_q_net2.parameters())
        )

        assert target_q1_changed, "Target Q1 should change via polyak update"
        assert target_q2_changed, "Target Q2 should change via polyak update"

        # Check that target update is indeed a weighted average
        # target = (1-tau) * target + tau * main
        tau = update.tau
        for p_target, p_main, p_old_target in zip(
            model.target_q_net1.parameters(),
            model.q_net1.parameters(),
            initial_target_q1,
        ):
            expected = (1 - tau) * p_old_target + tau * p_main
            assert torch.allclose(p_target, expected, atol=1e-5), (
                "Target update should follow polyak averaging"
            )

    def test_td_error_calculation(self):
        """Test TD error calculation."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch()

        td_error1, td_error2 = update.calculate_td_error(batch)

        # Check shapes
        assert td_error1.shape == (batch.batch_size, 1), (
            f"TD error1 shape: {td_error1.shape}"
        )
        assert td_error2.shape == (batch.batch_size, 1), (
            f"TD error2 shape: {td_error2.shape}"
        )

        # Check that values are finite
        assert torch.all(torch.isfinite(td_error1)), "TD error1 should be finite"
        assert torch.all(torch.isfinite(td_error2)), "TD error2 should be finite"

    def test_fixed_alpha_mode(self):
        """Test SAC with fixed alpha (no automatic tuning)."""
        fixed_alpha = 0.15
        update, model = self.get_update_and_model(auto_alpha=False, alpha=fixed_alpha)
        batch = DummyTransitionBatch()

        # Store initial alpha
        initial_alpha = update.alpha

        # Run update
        metrics = update.update(batch)

        # Alpha should remain unchanged
        assert update.alpha == initial_alpha == fixed_alpha

        # Should not have alpha-related attributes
        assert not hasattr(update, "log_alpha")
        assert not hasattr(update, "alpha_optimizer")

        # Metrics should still be valid
        required_metrics = [
            "q_loss1",
            "q_loss2",
            "policy_loss",
            "td_error1",
            "td_error2",
        ]
        for metric in required_metrics:
            assert metric in metrics

    def test_custom_target_entropy(self):
        """Test SAC with custom target entropy."""
        custom_entropy = -1.5
        update, model = self.get_update_and_model(target_entropy=custom_entropy)

        assert update.target_entropy == custom_entropy

        # Run update to ensure it works with custom entropy
        batch = DummyTransitionBatch()
        metrics = update.update(batch)

        assert "policy_loss" in metrics

    def test_different_learning_rates(self):
        """Test SAC with different learning rates for policy and Q-networks."""
        update, model = self.get_update_and_model(
            policy_lr=0.0005, q_lr=0.002, alpha_lr=0.001
        )

        # Check that optimizers have correct learning rates
        assert update.policy_optimizer.param_groups[0]["lr"] == 0.0005
        assert update.q_optimizer1.param_groups[0]["lr"] == 0.002
        assert update.q_optimizer2.param_groups[0]["lr"] == 0.002
        assert update.alpha_optimizer.param_groups[0]["lr"] == 0.001

    def test_different_tau_values(self):
        """Test SAC with different tau values for target updates."""
        # Test with very small tau (slow updates)
        update_small, model_small = self.get_update_and_model(tau=0.001)
        batch = DummyTransitionBatch()

        # Store initial target parameters
        initial_target = [p.clone() for p in model_small.target_q_net1.parameters()]

        # Run update
        update_small.update(batch)

        # With small tau, target should change very little
        total_change = sum(
            torch.norm(p1 - p2).item()
            for p1, p2 in zip(initial_target, model_small.target_q_net1.parameters())
        )

        # Test with larger tau (faster updates)
        update_large, model_large = self.get_update_and_model(tau=0.1)

        # Copy same initial weights for fair comparison
        for p1, p2 in zip(model_small.parameters(), model_large.parameters()):
            p2.data.copy_(p1.data)

        initial_target_large = [
            p.clone() for p in model_large.target_q_net1.parameters()
        ]
        update_large.update(batch)

        total_change_large = sum(
            torch.norm(p1 - p2).item()
            for p1, p2 in zip(
                initial_target_large, model_large.target_q_net1.parameters()
            )
        )

        assert total_change_large > total_change, (
            "Larger tau should cause bigger target network changes"
        )

    def test_zero_rewards_batch(self):
        """Test SAC with zero rewards."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch()
        batch.rewards.fill_(0.0)

        # Should handle zero rewards gracefully
        metrics = update.update(batch)

        for metric_name, metric_value in metrics.items():
            assert np.isfinite(metric_value), (
                f"Metric {metric_name} should be finite with zero rewards"
            )

    def test_all_done_batch(self):
        """Test SAC with all episodes terminated."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch()
        batch.dones.fill_(1.0)  # All episodes done

        # Should handle all-done batch gracefully
        metrics = update.update(batch)

        for metric_name, metric_value in metrics.items():
            assert np.isfinite(metric_value), (
                f"Metric {metric_name} should be finite with all done"
            )

    def test_metric_ranges(self):
        """Test that metrics are in reasonable ranges."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch()

        metrics = update.update(batch)

        # Q losses should be non-negative (MSE loss)
        assert metrics["q_loss1"] >= 0, "Q loss 1 should be non-negative"
        assert metrics["q_loss2"] >= 0, "Q loss 2 should be non-negative"

        # Policy loss can be negative (we want to maximize Q - alpha*entropy)
        assert np.isfinite(metrics["policy_loss"]), "Policy loss should be finite"

        # TD errors can be positive or negative
        assert np.isfinite(metrics["td_error1"]), "TD error 1 should be finite"
        assert np.isfinite(metrics["td_error2"]), "TD error 2 should be finite"

    @pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
    def test_different_batch_sizes(self, batch_size):
        """Test SAC with different batch sizes."""
        update, model = self.get_update_and_model()
        batch = DummyTransitionBatch(batch_size=batch_size)

        # Should work with any reasonable batch size
        metrics = update.update(batch)

        required_metrics = [
            "q_loss1",
            "q_loss2",
            "policy_loss",
            "td_error1",
            "td_error2",
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert np.isfinite(metrics[metric])

    def test_gradient_flow(self):
        """Test that gradients flow properly through the networks."""
        # Use non-zero initialization to ensure meaningful gradients
        update, model = self.get_update_and_model(initial_weights=0.1)
        batch = DummyTransitionBatch()

        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone()

        # Run update
        metrics = update.update(batch)

        # Check that parameters changed (indicating gradient flow)
        changed_params = 0
        total_params = 0
        param_changes = {}

        for name, param in model.named_parameters():
            if "target" not in name:  # Skip target networks (they update via polyak)
                total_params += 1
                change = torch.norm(param - initial_params[name]).item()
                param_changes[name] = change

                # Use more lenient tolerance since SAC updates can be small
                if change > 1e-8:  # Much more lenient than 1e-6
                    changed_params += 1

        # Debug information
        if changed_params == 0:
            print("Parameter changes:")
            for name, change in param_changes.items():
                print(f"  {name}: {change:.2e}")

        # At least some parameters should change
        change_ratio = changed_params / total_params
        assert change_ratio > 0.1, (
            f"Only {change_ratio:.2%} of parameters changed, gradient flow might be broken. Changes: {param_changes}"
        )

        # Additional check: ensure losses are reasonable
        assert np.isfinite(metrics["q_loss1"]) and metrics["q_loss1"] >= 0
        assert np.isfinite(metrics["q_loss2"]) and metrics["q_loss2"] >= 0
        assert np.isfinite(metrics["policy_loss"])
