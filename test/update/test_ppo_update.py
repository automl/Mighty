
import numpy as np
import pytest
import torch
import torch.nn as nn

from mighty.mighty_update import PPOUpdate


class DummyPPOModel(nn.Module):
    """Dummy PPO model for testing."""

    def __init__(
        self, obs_dim=4, action_dim=2, continuous_action=False, initial_weights=0.0
    ):
        super().__init__()
        self.continuous_action = continuous_action
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Linear(obs_dim, 64)

        # Policy head
        if continuous_action:
            self.policy_head = nn.Linear(64, action_dim)  # mean
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.policy_head = nn.Linear(64, action_dim)  # logits

        # Value head
        self.value_head = nn.Linear(64, 1)

        # Initialize weights
        self._init_weights(initial_weights)

    def _init_weights(self, weight_val):
        """Initialize all weights to a specific value."""
        for param in self.parameters():
            param.data.fill_(weight_val)

    def forward(self, obs):
        """Forward pass - returns different outputs based on action type."""
        features = torch.relu(self.shared(obs))

        if self.continuous_action:
            mean = self.policy_head(features)
            log_std = self.log_std.expand_as(mean)
            return None, None, mean, log_std  # Match your model's return format
        else:
            logits = self.policy_head(features)
            return logits

    def forward_value(self, obs):
        """Forward pass for value estimation."""
        features = torch.relu(self.shared(obs))
        return self.value_head(features)


class DummyMiniBatch:
    """Dummy minibatch for testing."""

    def __init__(self, batch_size=32, obs_dim=4, action_dim=2, continuous_action=False):
        self.batch_size = batch_size
        self.observations = torch.randn(batch_size, obs_dim)

        if continuous_action:
            self.actions = torch.randn(batch_size, action_dim)
            self.latents = torch.randn(batch_size, action_dim)  # Pre-tanh actions
        else:
            self.actions = torch.randint(0, action_dim, (batch_size,))
            self.latents = None

        self.log_probs = torch.randn(batch_size)
        self.returns = torch.randn(batch_size)
        self.advantages = torch.randn(batch_size)


class DummyMaxiBatch:
    """Dummy maxibatch containing multiple minibatches."""

    def __init__(
        self,
        n_minibatches=4,
        batch_size=32,
        obs_dim=4,
        action_dim=2,
        continuous_action=False,
    ):
        self.minibatches = [
            DummyMiniBatch(batch_size, obs_dim, action_dim, continuous_action)
            for _ in range(n_minibatches)
        ]

        # Flatten advantages for normalization
        all_advantages = torch.cat([mb.advantages for mb in self.minibatches])
        self.advantages = all_advantages


class TestPPOUpdate:
    """Test PPO update mechanism."""

    def get_update_and_model(
        self, continuous_action=False, initial_weights=0.0, **ppo_kwargs
    ):
        """Create PPO update instance and model for testing."""
        model = DummyPPOModel(
            continuous_action=continuous_action, initial_weights=initial_weights
        )

        # Default PPO parameters for testing
        default_kwargs = {
            "policy_lr": 3e-4,
            "value_lr": 3e-4,
            "epsilon": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_epochs": 1,
            "minibatch_size": 32,
            "kl_target": 0.01,
            "adaptive_lr": False,  # Disable for predictable testing
        }
        default_kwargs.update(ppo_kwargs)

        update = PPOUpdate(model, **default_kwargs)
        return update, model

    def test_initialization(self):
        """Test PPO update initialization."""
        update, model = self.get_update_and_model()

        # Check that optimizers are created
        assert hasattr(update, "optimizer")
        assert hasattr(update, "scheduler")

        # Check parameter groups
        assert len(update.optimizer.param_groups) >= 2  # Policy and value

        # Check learning rates
        policy_lr = update.optimizer.param_groups[0]["lr"]
        value_lr = update.optimizer.param_groups[1]["lr"]
        assert policy_lr == 3e-4
        assert value_lr == 3e-4

    def test_discrete_action_update(self):
        """Test PPO update with discrete actions."""
        update, model = self.get_update_and_model(continuous_action=False)
        batch = DummyMaxiBatch(continuous_action=False)

        # Store initial parameters
        initial_policy_params = [p.clone() for p in model.policy_head.parameters()]
        initial_value_params = [p.clone() for p in model.value_head.parameters()]

        # Run update
        metrics = update.update(batch)

        # Check that parameters changed
        policy_changed = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(initial_policy_params, model.policy_head.parameters())
        )
        value_changed = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(initial_value_params, model.value_head.parameters())
        )

        assert policy_changed, "Policy parameters should change after update"
        assert value_changed, "Value parameters should change after update"

        # Check metrics
        required_metrics = [
            "Update/policy_loss",
            "Update/value_loss",
            "Update/entropy",
            "Update/approx_kl",
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), (
                f"Metric {metric} should be numeric"
            )

    def test_continuous_action_update(self):
        """Test PPO update with continuous actions."""
        update, model = self.get_update_and_model(continuous_action=True)
        batch = DummyMaxiBatch(continuous_action=True)

        # Store initial parameters
        initial_policy_params = [p.clone() for p in model.policy_head.parameters()]
        initial_log_std = model.log_std.clone()
        initial_value_params = [p.clone() for p in model.value_head.parameters()]

        # Run update
        metrics = update.update(batch)

        # Check that parameters changed
        policy_changed = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(initial_policy_params, model.policy_head.parameters())
        )
        log_std_changed = not torch.equal(initial_log_std, model.log_std)
        value_changed = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(initial_value_params, model.value_head.parameters())
        )

        assert policy_changed, "Policy parameters should change after update"
        assert log_std_changed, "Log std parameters should change after update"
        assert value_changed, "Value parameters should change after update"

        # Check metrics
        required_metrics = [
            "Update/policy_loss",
            "Update/value_loss",
            "Update/entropy",
            "Update/approx_kl",
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), (
                f"Metric {metric} should be numeric"
            )

    def test_value_clipping(self):
        """Test value clipping mechanism."""
        # Test with value clipping enabled
        update_clip, model_clip = self.get_update_and_model(
            use_value_clip=True, value_clip_eps=0.2
        )
        batch = DummyMaxiBatch()

        metrics_clip = update_clip.update(batch)

        # Test with value clipping disabled
        update_no_clip, model_no_clip = self.get_update_and_model(use_value_clip=False)

        # Use same initial weights for fair comparison
        for p1, p2 in zip(model_clip.parameters(), model_no_clip.parameters()):
            p2.data.copy_(p1.data)

        metrics_no_clip = update_no_clip.update(batch)

        # Both should produce valid metrics (values may differ due to clipping)
        assert "Update/value_loss" in metrics_clip
        assert "Update/value_loss" in metrics_no_clip
        assert metrics_clip["Update/value_loss"] >= 0
        assert metrics_no_clip["Update/value_loss"] >= 0

    def test_multiple_epochs(self):
        """Test PPO update with multiple epochs."""
        update, model = self.get_update_and_model(n_epochs=3)
        batch = DummyMaxiBatch()

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Run update
        metrics = update.update(batch)

        # Check that parameters changed more significantly with multiple epochs
        total_change = sum(
            torch.norm(p1 - p2).item()
            for p1, p2 in zip(initial_params, model.parameters())
        )

        assert total_change > 0, "Parameters should change with multiple epochs"
        assert "Update/approx_kl" in metrics, "Should track KL divergence"

    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate adjustment."""
        update, _ = self.get_update_and_model(adaptive_lr=True, kl_target=0.01)

        # Create batch that might trigger LR adaptation
        batch = DummyMaxiBatch()

        # Run update
        _ = update.update(batch)

        # Learning rates might have changed (depending on KL divergence)
        final_policy_lr = update.optimizer.param_groups[0]["lr"]
        final_value_lr = update.optimizer.param_groups[1]["lr"]

        # LRs should be positive and within reasonable bounds
        assert final_policy_lr > 0
        assert final_value_lr > 0
        assert final_policy_lr >= update.min_lr
        assert final_value_lr >= update.min_lr

    def test_gradient_clipping(self):
        """Test gradient clipping mechanism."""
        update, model = self.get_update_and_model(max_grad_norm=0.5)
        batch = DummyMaxiBatch()

        # Run update (gradient clipping happens internally)
        metrics = update.update(batch)

        # Should complete without errors and produce valid metrics
        assert "Update/policy_loss" in metrics
        assert "Update/value_loss" in metrics
        assert not torch.isnan(torch.tensor(metrics["Update/policy_loss"]))
        assert not torch.isnan(torch.tensor(metrics["Update/value_loss"]))

    def test_zero_advantages(self):
        """Test behavior with zero advantages."""
        update, model = self.get_update_and_model()
        batch = DummyMaxiBatch()

        # Set all advantages to zero
        for mb in batch.minibatches:
            mb.advantages.fill_(0.0)
        batch.advantages.fill_(0.0)

        # Should handle zero advantages gracefully
        metrics = update.update(batch)

        assert "Update/policy_loss" in metrics
        assert not torch.isnan(torch.tensor(metrics["Update/policy_loss"]))

    @pytest.mark.parametrize("continuous_action", [True, False])
    def test_metric_shapes_and_types(self, continuous_action):
        """Test that all metrics have correct shapes and types."""
        update, model = self.get_update_and_model(continuous_action=continuous_action)
        batch = DummyMaxiBatch(continuous_action=continuous_action)

        metrics = update.update(batch)

        expected_metrics = [
            "Update/policy_loss",
            "Update/value_loss",
            "Update/entropy",
            "Update/approx_kl",
        ]

        for metric_name in expected_metrics:
            assert metric_name in metrics, f"Missing metric: {metric_name}"
            metric_value = metrics[metric_name]
            assert isinstance(metric_value, (int, float)), (
                f"Metric {metric_name} should be scalar"
            )
            assert not np.isnan(metric_value), f"Metric {metric_name} should not be NaN"
            assert np.isfinite(metric_value), f"Metric {metric_name} should be finite"
