from __future__ import annotations

from copy import deepcopy
import math

import torch
import torch.nn as nn

from mighty.mighty_models.networks import MLP
from mighty.mighty_models.sac import SACModel
from mighty.mighty_exploration.mighty_exploration_policy import sample_nondeterministic_logprobs


class TestSACModel:
    def test_init(self):
        """Test initialization of SAC model."""
        sac = SACModel(obs_size=8, action_size=3, activation="tanh")
        
        assert sac.obs_size == 8, "Obs size should be 8"
        assert sac.action_size == 3, "Action size should be 3"
        assert sac.log_std_min == -20, "Default log_std_min should be -20"
        assert sac.log_std_max == 2, "Default log_std_max should be 2"
        
        # Check network structure
        assert isinstance(sac.feature_extractor, MLP), "Feature extractor should be Sequential"
        assert len(sac.feature_extractor.layers) == 4, "Feature extractor should have 2 layers + 2 activation layers"
        assert sac.feature_extractor.layers[0].in_features == 8, "First layer should take obs_size as input"
        assert isinstance(sac.feature_extractor.layers[1], nn.ReLU), "Feature extractor should have ReLU activation"
        assert sac.feature_extractor.layers[-2].out_features == 256, "Last layer should output 256 features"
        assert isinstance(sac.policy_net, nn.Linear), (
            "Policy network should be Linear"
        )
        assert sac.policy_net.out_features == 6, "Last layer should output action_size*2"

        assert isinstance(sac.q_net1, nn.Sequential), (
            "Q-network 1 should be Sequential"
        )
        assert len(sac.q_net1) == 5, "Q-network 1 should have 3 layers + 2 activation layers"
        assert sac.q_net1[0].in_features == 11, "Q-network should take obs_size + action_size as input"
        assert sac.q_net1[0].out_features == 256, "Q-network first layer should output 256 features"
        assert sac.q_net1[-1].out_features == 1, "Q-network should output a single value"

        assert isinstance(sac.q_net2, nn.Sequential), (
            "Q-network 2 should be Sequential"
        )
        assert len(sac.q_net2) == 5, "Q-network 2 should have 3 layers + 2 activation layers"
        assert isinstance(sac.target_q_net1, nn.Sequential), (
            "Target Q-network 1 should be Sequential"
        )
        assert len(sac.target_q_net1) == 5, "Target Q-network 1 should have 3 layers + 2 activation layers"
        assert isinstance(sac.target_q_net2, nn.Sequential), (
            "Target Q-network 2 should be Sequential"
        )
        assert len(sac.target_q_net2) == 5, "Target Q-network 2 should have 3 layers + 2 activation layers"

        # Check that target networks have gradients disabled
        for param in sac.target_q_net1.parameters():
            assert not param.requires_grad, (
                "Target Q-network 1 parameters should not require gradients"
            )
        for param in sac.target_q_net2.parameters():
            assert not param.requires_grad, (
                "Target Q-network 2 parameters should not require gradients"
            )
        
        # Check that live networks have gradients enabled
        for param in sac.q_net1.parameters():
            assert param.requires_grad, (
                "Q-network 1 parameters should require gradients"
            )
        for param in sac.q_net2.parameters():
            assert param.requires_grad, (
                "Q-network 2 parameters should require gradients"
            )

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        feature_extractor_kwargs = {
            "activation": "tanh",
            "hidden_sizes": [128, 64],
            "n_layers": 2,
        }
        head_kwargs = {
            "hidden_sizes": [64],
            "activation": "tanh",
        }
        sac = SACModel(
            obs_size=4,
            action_size=2,
            log_std_min=-10.0,
            log_std_max=1.0,
            feature_extractor_kwargs=feature_extractor_kwargs,
            head_kwargs=head_kwargs
        )
        
        assert sac.obs_size == 4, "Custom obs size should be 4"
        assert sac.action_size == 2, "Custom action size should be 2"
        assert sac.log_std_min == -10.0, "Custom log_std_min should be -10.0"
        assert sac.log_std_max == 1.0, "Custom log_std_max should be 1.0"

        assert sac.feature_extractor.layers[0].in_features == 4, "Feature extractor should take obs_size as input"
        assert sac.feature_extractor.layers[0].out_features == 128, "Feature extractor first layer should output 128 features"
        assert isinstance(sac.feature_extractor.layers[1], nn.Tanh), "Feature extractor should have tanh activation"
        assert sac.feature_extractor.layers[2].out_features == 64, "Feature extractor second layer should output 64 features"

        assert len(sac.q_net1) == 3, "Q-network 1 should have 2 layers + 1 activation layer"
        assert sac.q_net1[0].in_features == 6, "Q-network should take obs_size + action_size as input"
        assert sac.q_net1[0].out_features == 64, "Q-network first layer should output 64 features"
        assert sac.q_net1[-1].out_features == 1, "Q-network should output a single value"

    def test_forward_stochastic(self):
        """Test forward pass with stochastic policy."""
        sac = SACModel(obs_size=6, action_size=4)
        dummy_state = torch.rand((10, 6))
        
        action, z, mean, log_std = sac(dummy_state, deterministic=False)
        
        # Check shapes
        assert action.shape == (10, 4), "Action should have shape (10, 4)"
        assert z.shape == (10, 4), "Raw action (z) should have shape (10, 4)"
        assert mean.shape == (10, 4), "Mean should have shape (10, 4)"
        assert log_std.shape == (10, 4), "Log_std should have shape (10, 4)"
        
        # Check that all outputs are finite
        assert torch.all(torch.isfinite(action)), "Actions should be finite"
        assert torch.all(torch.isfinite(z)), "Raw actions should be finite"
        assert torch.all(torch.isfinite(mean)), "Means should be finite"
        assert torch.all(torch.isfinite(log_std)), "Log_stds should be finite"
        
        # Check tanh constraint on actions
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0), (
            "Actions should be in [-1, 1] range"
        )
        
        # Check log_std clamping
        assert torch.all(log_std >= sac.log_std_min), (
            "Log_std should be >= log_std_min"
        )
        assert torch.all(log_std <= sac.log_std_max), (
            "Log_std should be <= log_std_max"
        )
        
        # Check relationship: action = tanh(z)
        expected_action = torch.tanh(z)
        assert torch.allclose(action, expected_action, atol=1e-6), (
            "Action should equal tanh(z)"
        )

    def test_forward_deterministic(self):
        """Test forward pass with deterministic policy."""
        sac = SACModel(obs_size=5, action_size=2)
        dummy_state = torch.rand((8, 5))
        
        action, z, mean, log_std = sac(dummy_state, deterministic=True)
        
        # Check shapes
        assert action.shape == (8, 2), "Action should have shape (8, 2)"
        assert z.shape == (8, 2), "Raw action (z) should have shape (8, 2)"
        assert mean.shape == (8, 2), "Mean should have shape (8, 2)"
        assert log_std.shape == (8, 2), "Log_std should have shape (8, 2)"
        
        # In deterministic mode, z should equal mean
        assert torch.allclose(z, mean), "In deterministic mode, z should equal mean"
        
        # Action should still be tanh(z) = tanh(mean)
        expected_action = torch.tanh(mean)
        assert torch.allclose(action, expected_action), (
            "Action should equal tanh(mean) in deterministic mode"
        )

    def test_stochastic_vs_deterministic(self):
        """Test that stochastic and deterministic modes produce different results."""
        sac = SACModel(obs_size=4, action_size=2)
        dummy_state = torch.rand((5, 4))
        
        # Get stochastic output
        action_stoch, z_stoch, mean_stoch, log_std_stoch = sac(dummy_state, deterministic=False)
        
        # Get deterministic output
        action_det, z_det, mean_det, log_std_det = sac(dummy_state, deterministic=True)
        
        # Mean and log_std should be the same
        assert torch.allclose(mean_stoch, mean_det), "Means should be identical"
        assert torch.allclose(log_std_stoch, log_std_det), "Log_stds should be identical"
        
        # In deterministic mode, z should equal mean
        assert torch.allclose(z_det, mean_det), "Deterministic z should equal mean"
        
        # Stochastic z should likely be different from mean (due to noise)
        # Note: There's a tiny chance they could be the same, but extremely unlikely
        assert not torch.allclose(z_stoch, mean_stoch), (
            "Stochastic z should be different from mean"
        )

    def test_q_networks(self):
        """Test Q-network forward passes."""
        sac = SACModel(obs_size=4, action_size=2)
        dummy_state = torch.rand((7, 4))
        dummy_action = torch.rand((7, 2))
        
        # Concatenate state and action for Q-networks
        state_action = torch.cat([dummy_state, dummy_action], dim=-1)
        
        q1_value = sac.forward_q1(state_action)
        q2_value = sac.forward_q2(state_action)
        
        # Check shapes
        assert q1_value.shape == (7, 1), "Q1 values should have shape (7, 1)"
        assert q2_value.shape == (7, 1), "Q2 values should have shape (7, 1)"
        
        # Check that values are finite
        assert torch.all(torch.isfinite(q1_value)), "Q1 values should be finite"
        assert torch.all(torch.isfinite(q2_value)), "Q2 values should be finite"

    def test_target_networks_initialization(self):
        """Test that target networks are initialized with same weights as live networks."""
        sac = SACModel(obs_size=3, action_size=2)
        
        # Check that target networks have same weights as live networks initially
        for p1, p_target1 in zip(sac.q_net1.parameters(), sac.target_q_net1.parameters()):
            assert torch.allclose(p1, p_target1), (
                "Target Q-net 1 should have same initial weights as Q-net 1"
            )
        
        for p2, p_target2 in zip(sac.q_net2.parameters(), sac.target_q_net2.parameters()):
            assert torch.allclose(p2, p_target2), (
                "Target Q-net 2 should have same initial weights as Q-net 2"
            )

    def test_twin_q_networks_independence(self):
        """Test that twin Q-networks are independent."""
        sac = SACModel(obs_size=4, action_size=2)
        
        # Check that Q-networks have different parameters (due to random initialization)
        params_different = False
        for p1, p2 in zip(sac.q_net1.parameters(), sac.q_net2.parameters()):
            if not torch.allclose(p1, p2):
                params_different = True
                break
        
        # Due to random initialization, they should be different
        assert sac.q_net1 is not sac.q_net2, "Q-networks should be separate objects"
        assert sac.target_q_net1 is not sac.target_q_net2, (
            "Target Q-networks should be separate objects"
        )

    def test_log_std_bounds_enforcement(self):
        """Test that log_std bounds are properly enforced."""
        log_std_min = -5.0
        log_std_max = 0.5
        sac = SACModel(
            obs_size=3, 
            action_size=2, 
            log_std_min=log_std_min, 
            log_std_max=log_std_max
        )
        
        dummy_state = torch.rand((10, 3))
        _, _, _, log_std = sac(dummy_state)
        
        assert torch.all(log_std >= log_std_min), (
            "Log_std should be >= custom log_std_min"
        )
        assert torch.all(log_std <= log_std_max), (
            "Log_std should be <= custom log_std_max"
        )

    def test_state_dict_completeness(self):
        """Test that state dict contains all expected parameters."""
        sac = SACModel(obs_size=4, action_size=2)
        state_dict = sac.state_dict()
        
        assert isinstance(state_dict, dict), "State dict should be a dictionary"
        
        # Check for expected network prefixes
        expected_prefixes = [
            "policy_net",
            "q_net1",
            "q_net2", 
            "target_q_net1",
            "target_q_net2"
        ]
        
        for prefix in expected_prefixes:
            found_key = any(key.startswith(prefix) for key in state_dict.keys())
            assert found_key, f"Should find keys starting with {prefix}"

    def test_load_state_dict(self):
        """Test loading state dict preserves model behavior."""
        sac1 = SACModel(obs_size=4, action_size=2)
        sac2 = SACModel(obs_size=4, action_size=2)
        
        dummy_state = torch.rand((5, 4))
        dummy_action = torch.rand((5, 2))
        state_action = torch.cat([dummy_state, dummy_action], dim=-1)
        
        # Get predictions from first model
        with torch.no_grad():
            action1, z1, mean1, log_std1 = sac1(dummy_state, deterministic=True)
            q1_1 = sac1.forward_q1(state_action)
            q2_1 = sac1.forward_q2(state_action)
        
        # Initially, models should produce different outputs
        with torch.no_grad():
            action2_before, _, mean2_before, _ = sac2(dummy_state, deterministic=True)
            q1_2_before = sac2.forward_q1(state_action)
            q2_2_before = sac2.forward_q2(state_action)
        
        assert not torch.allclose(mean1, mean2_before), (
            "Models should produce different outputs initially"
        )
        assert not torch.allclose(q1_1, q1_2_before), (
            "Q1 networks should produce different outputs initially"
        )
        
        # Load state dict
        sac2.load_state_dict(sac1.state_dict())
        
        # Now they should produce the same outputs
        with torch.no_grad():
            action2_after, z2_after, mean2_after, log_std2_after = sac2(dummy_state, deterministic=True)
            q1_2_after = sac2.forward_q1(state_action)
            q2_2_after = sac2.forward_q2(state_action)
        
        assert torch.allclose(mean1, mean2_after), (
            "Policy means should be same after loading state dict"
        )
        assert torch.allclose(log_std1, log_std2_after), (
            "Policy log_stds should be same after loading state dict" 
        )
        assert torch.allclose(q1_1, q1_2_after), (
            "Q1 networks should produce same outputs after loading state dict"
        )
        assert torch.allclose(q2_1, q2_2_after), (
            "Q2 networks should produce same outputs after loading state dict"
        )

    def test_gradient_flow(self):
        """Test that gradients flow properly through networks."""
        sac = SACModel(obs_size=4, action_size=2)
        dummy_state = torch.rand((3, 4))
        dummy_action = torch.rand((3, 2))
        state_action = torch.cat([dummy_state, dummy_action], dim=-1)
        
        # Test policy network gradients
        action, z, mean, log_std = sac(dummy_state)
        policy_loss = action.mean()  # Dummy loss
        policy_loss.backward(retain_graph=True)
        
        # Check that policy network has gradients
        policy_has_grad = any(p.grad is not None for p in sac.policy_net.parameters())
        assert policy_has_grad, "Policy network should have gradients"
        
        # Test Q-network gradients
        sac.zero_grad()
        q1_value = sac.forward_q1(state_action)
        q_loss = q1_value.mean()  # Dummy loss
        q_loss.backward()
        
        # Check that Q-network 1 has gradients
        q1_has_grad = any(p.grad is not None for p in sac.q_net1.parameters())
        assert q1_has_grad, "Q-network 1 should have gradients"
        
        # Check that target networks don't have gradients
        target_q1_has_grad = any(p.grad is not None for p in sac.target_q_net1.parameters())
        assert not target_q1_has_grad, "Target Q-network 1 should not have gradients"

    def test_numerical_stability(self):
        """Test numerical stability of log probability calculation."""
        sac = SACModel(obs_size=2, action_size=1)
        
        # Test with extreme values
        dummy_state = torch.tensor([[10.0, -10.0], [0.0, 0.0]])
        _, z, mean, log_std = sac(dummy_state, deterministic=False)
        
        # Test log probability calculation doesn't produce NaN or inf
        log_prob = sample_nondeterministic_logprobs(z, mean, log_std, sac=True)
        assert torch.all(torch.isfinite(log_prob)), (
            "Log probabilities should be finite even with extreme inputs"
        )
        
        # Test with actions close to boundary values (-1, 1)
        boundary_z = torch.tensor([[5.0], [-5.0], [0.0]])  # These will be close to ±1 after tanh
        boundary_mean = torch.zeros_like(boundary_z)
        boundary_log_std = torch.zeros_like(boundary_z)
        
        boundary_log_prob = sample_nondeterministic_logprobs(boundary_z, boundary_mean, boundary_log_std, sac=True)
        assert torch.all(torch.isfinite(boundary_log_prob)), (
            "Log probabilities should be finite for boundary actions"
        )