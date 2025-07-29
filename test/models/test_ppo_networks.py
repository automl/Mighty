from __future__ import annotations


import torch
import torch.nn as nn

from mighty.mighty_models.networks import MLP
from mighty.mighty_models.ppo import PPOModel


class TestPPOModel:
    def test_init_discrete(self):
        """Test initialization for discrete action spaces."""
        ppo = PPOModel(obs_shape=4, action_size=2, continuous_action=False)

        assert ppo.obs_size == 4, "Obs size should be 4"
        assert ppo.action_size == 2, "Action size should be 2"
        assert ppo.continuous_action is False, "Should be discrete action"

        # Check network structure
        assert isinstance(ppo.feature_extractor_policy, MLP), (
            "Policy feature extractor should be MLP"
        )
        assert isinstance(ppo.feature_extractor_policy.layers[1], nn.Tanh), (
            "Feature extractor should have tanh activation"
        )
        assert ppo.feature_extractor_policy.layers[0].in_features == 4, (
            "First layer should take obs_size as input"
        )
        assert ppo.feature_extractor_policy.layers[-2].out_features == 64, (
            "Last layer should output 64 features"
        )
        assert isinstance(ppo.feature_extractor_value, MLP), (
            "Value feature extractor should be MLP"
        )
        assert isinstance(ppo.policy_head, nn.Sequential), (
            "Policy head should be Sequential"
        )
        assert len(ppo.policy_head) == 4, (
            "Policy head should have 2 layers, 1 activation layer and 1 normalization layer"
        )
        assert isinstance(ppo.policy_head[1], nn.LayerNorm), (
            "Policy head should have LayerNorm"
        )
        assert isinstance(ppo.policy_head[2], nn.Tanh), (
            "Policy head should have tanh activation"
        )
        assert isinstance(ppo.value_head, nn.Sequential), (
            "Value head should be Sequential"
        )

        # Test forward pass shapes
        dummy_input = torch.rand((10, 4))
        logits = ppo(dummy_input)
        values = ppo.forward_value(dummy_input)

        assert logits.shape == (10, 2), "Logits should have shape (10, 2)"
        assert values.shape == (10, 1), "Values should have shape (10, 1)"

    def test_init_continuous(self):
        """Test initialization for continuous action spaces."""
        ppo = PPOModel(
            obs_shape=8,
            action_size=3,
            continuous_action=True,
            feature_extractor_kwargs={
                "hidden_sizes": [32, 32],
                "n_layers": 2,
                "activation": "relu",
            },
        )

        assert ppo.obs_size == 8, "Obs size should be 8"
        assert ppo.action_size == 3, "Action size should be 3"
        assert ppo.continuous_action is True, "Should be continuous action"
        assert ppo.log_std_min == -20.0, "Default log_std_min should be -20.0"
        assert ppo.log_std_max == 2.0, "Default log_std_max should be 2.0"

        assert ppo.feature_extractor_policy.layers[-2].out_features == 32, (
            "Last layer should output 32 features"
        )
        assert isinstance(ppo.feature_extractor_policy.layers[1], nn.ReLU), (
            "Feature extractor should have ReLU activation"
        )

        # Test forward pass shapes for continuous actions
        dummy_input = torch.rand((5, 8))
        action, z, mean, log_std = ppo(dummy_input)
        values = ppo.forward_value(dummy_input)

        assert action.shape == (5, 3), "Action should have shape (5, 3)"
        assert z.shape == (5, 3), "Raw action (z) should have shape (5, 3)"
        assert mean.shape == (5, 3), "Mean should have shape (5, 3)"
        assert log_std.shape == (5, 3), "Log_std should have shape (5, 3)"
        assert values.shape == (5, 1), "Values should have shape (5, 1)"

        # Check that actions are in [-1, 1] range due to tanh
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0), (
            "Actions should be in [-1, 1] range"
        )

        # Check log_std clamping
        assert torch.all(log_std >= ppo.log_std_min), "Log_std should be >= log_std_min"
        assert torch.all(log_std <= ppo.log_std_max), "Log_std should be <= log_std_max"

    def test_forward_discrete(self):
        """Test forward pass for discrete actions."""
        ppo = PPOModel(obs_shape=6, action_size=4, continuous_action=False)
        dummy_input = torch.rand((20, 6))

        logits = ppo(dummy_input)
        assert logits.shape == (20, 4), "Logits should have shape (20, 4)"
        assert isinstance(logits, torch.Tensor), "Output should be tensor"

        # Test that logits are reasonable (not NaN or inf)
        assert torch.all(torch.isfinite(logits)), "Logits should be finite"

    def test_forward_continuous(self):
        """Test forward pass for continuous actions."""
        ppo = PPOModel(obs_shape=3, action_size=2, continuous_action=True)
        dummy_input = torch.rand((15, 3))

        action, z, mean, log_std = ppo(dummy_input)

        # Check shapes
        assert action.shape == (15, 2), "Action should have shape (15, 2)"
        assert z.shape == (15, 2), "Raw action should have shape (15, 2)"
        assert mean.shape == (15, 2), "Mean should have shape (15, 2)"
        assert log_std.shape == (15, 2), "Log_std should have shape (15, 2)"

        # Check that all outputs are finite
        assert torch.all(torch.isfinite(action)), "Actions should be finite"
        assert torch.all(torch.isfinite(z)), "Raw actions should be finite"
        assert torch.all(torch.isfinite(mean)), "Means should be finite"
        assert torch.all(torch.isfinite(log_std)), "Log_stds should be finite"

        # Check tanh constraint on actions
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0), (
            "Actions should be in [-1, 1]"
        )

        # Check relationship: action = tanh(z) where z = mean + std * eps
        expected_action = torch.tanh(z)
        assert torch.allclose(action, expected_action, atol=1e-6), (
            "Action should equal tanh(z)"
        )

    def test_forward_value(self):
        """Test value network forward pass."""
        ppo = PPOModel(obs_shape=5, action_size=3, continuous_action=False)
        dummy_input = torch.rand((12, 5))

        values = ppo.forward_value(dummy_input)
        assert values.shape == (12, 1), "Values should have shape (12, 1)"
        assert torch.all(torch.isfinite(values)), "Values should be finite"

    def test_custom_log_std_bounds(self):
        """Test custom log_std bounds for continuous actions."""
        log_std_min = -10.0
        log_std_max = 1.0
        ppo = PPOModel(
            obs_shape=4,
            action_size=2,
            continuous_action=True,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

        dummy_input = torch.rand((10, 4))
        _, _, _, log_std = ppo(dummy_input)

        assert torch.all(log_std >= log_std_min), (
            "Log_std should be >= custom log_std_min"
        )
        assert torch.all(log_std <= log_std_max), (
            "Log_std should be <= custom log_std_max"
        )

    def test_deterministic_with_same_input(self):
        """Test that same input produces different outputs due to sampling."""
        ppo = PPOModel(obs_shape=4, action_size=2, continuous_action=True)
        dummy_input = torch.rand((5, 4))

        # Get two forward passes with same input
        action1, z1, mean1, log_std1 = ppo(dummy_input)
        action2, z2, mean2, log_std2 = ppo(dummy_input)

        # Mean and log_std should be the same (deterministic)
        assert torch.allclose(mean1, mean2), "Means should be identical"
        assert torch.allclose(log_std1, log_std2), "Log_stds should be identical"

        # Actions and z should be different due to random sampling
        # (Note: there's a tiny chance they could be the same, but extremely unlikely)
        assert not torch.allclose(action1, action2), (
            "Actions should be different due to sampling"
        )
        assert not torch.allclose(z1, z2), (
            "Raw actions should be different due to sampling"
        )

    def test_orthogonal_initialization(self):
        """Test that weights are initialized with orthogonal initialization."""
        ppo = PPOModel(obs_shape=4, action_size=2, continuous_action=False)

        # Check that linear layers have been initialized
        for module in ppo.modules():
            if isinstance(module, nn.Linear):
                # Check that weights are not all zeros (indicating initialization occurred)
                assert not torch.allclose(
                    module.weight, torch.zeros_like(module.weight)
                ), "Weights should not be all zeros"
                # Check that biases are initialized to zero
                assert torch.allclose(module.bias, torch.zeros_like(module.bias)), (
                    "Biases should be initialized to zero"
                )

    def test_separate_feature_extractors(self):
        """Test that policy and value networks have separate feature extractors."""
        ppo = PPOModel(obs_shape=6, action_size=3, continuous_action=False)
        dummy_input = torch.rand((8, 6))

        # Extract features from both networks
        policy_features = ppo.feature_extractor_policy(dummy_input)
        value_features = ppo.feature_extractor_value(dummy_input)

        # They should have the same shape but potentially different values
        assert policy_features.shape == value_features.shape, (
            "Feature extractors should output same shape"
        )

        # Verify they are separate networks by checking if parameters are different
        policy_params = list(ppo.feature_extractor_policy.parameters())
        value_params = list(ppo.feature_extractor_value.parameters())

        # Check that at least some parameters are different (they're separate networks)
        params_different = False
        for p1, p2 in zip(policy_params, value_params, strict=False):
            if not torch.allclose(p1, p2):
                params_different = True
                break

        assert params_different, "Feature extractors should have different parameters"

        # Note: Due to random initialization, they should be different
        # But we'll just check that they exist as separate modules
        assert ppo.feature_extractor_policy is not ppo.feature_extractor_value, (
            "Policy and value feature extractors should be separate objects"
        )

    def test_state_dict_completeness(self):
        """Test that state dict contains all expected parameters."""
        ppo = PPOModel(obs_shape=4, action_size=2, continuous_action=True)
        state_dict = ppo.state_dict()

        assert isinstance(state_dict, dict), "State dict should be a dictionary"

        # Check for expected keys
        expected_prefixes = [
            "feature_extractor_policy",
            "feature_extractor_value",
            "policy_head",
            "value_head",
        ]

        for prefix in expected_prefixes:
            found_key = any(key.startswith(prefix) for key in state_dict.keys())
            assert found_key, f"Should find keys starting with {prefix}"

    def test_load_state_dict(self):
        """Test loading state dict preserves model behavior."""
        ppo1 = PPOModel(obs_shape=4, action_size=2, continuous_action=False)
        ppo2 = PPOModel(obs_shape=4, action_size=2, continuous_action=False)

        dummy_input = torch.rand((5, 4))

        # Get predictions from first model
        with torch.no_grad():
            logits1 = ppo1(dummy_input)
            values1 = ppo1.forward_value(dummy_input)

        # Initially, models should produce different outputs
        with torch.no_grad():
            logits2_before = ppo2(dummy_input)
            values2_before = ppo2.forward_value(dummy_input)

        assert not torch.allclose(logits1, logits2_before), (
            "Models should produce different outputs initially"
        )
        assert not torch.allclose(values1, values2_before), (
            "Value networks should produce different outputs initially"
        )

        # Load state dict
        ppo2.load_state_dict(ppo1.state_dict())

        # Now they should produce the same outputs
        with torch.no_grad():
            logits2_after = ppo2(dummy_input)
            values2_after = ppo2.forward_value(dummy_input)

        assert torch.allclose(logits1, logits2_after), (
            "Models should produce same outputs after loading state dict"
        )
        assert torch.allclose(values1, values2_after), (
            "Value networks should produce same outputs after loading state dict"
        )

    def test_different_architectures(self):
        """Test with different hidden layer configurations."""
        # Single layer
        ppo_single = PPOModel(
            obs_shape=3, action_size=2, hidden_sizes=[32], continuous_action=False
        )

        # Multiple layers
        ppo_multi = PPOModel(
            obs_shape=3,
            action_size=2,
            hidden_sizes=[64, 32, 16],
            continuous_action=False,
        )

        dummy_input = torch.rand((4, 3))

        # Both should work
        logits_single = ppo_single(dummy_input)
        logits_multi = ppo_multi(dummy_input)

        assert logits_single.shape == (4, 2), "Single layer model should work"
        assert logits_multi.shape == (4, 2), "Multi layer model should work"

        values_single = ppo_single.forward_value(dummy_input)
        values_multi = ppo_multi.forward_value(dummy_input)

        assert values_single.shape == (4, 1), "Single layer value should work"
        assert values_multi.shape == (4, 1), "Multi layer value should work"
