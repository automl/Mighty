import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Callable

# Import the classes to test
from mighty.mighty_agents.ppo import MightyPPOAgent
from mighty.mighty_models.ppo import PPOModel
from mighty.mighty_update.ppo_update import PPOUpdate

import gymnasium as gym
from copy import deepcopy

# Import shared test helpers
from mighty.mighty_utils.test_helpers import DummyEnv, DummyContinuousEnv, clean
from mighty.mighty_replay.mighty_rollout_buffer import MaxiBatch, RolloutBatch


class TestPPOAgent:
    def test_init_continuous(self):
        """Test PPO agent initialization with continuous actions."""
        env = gym.vector.SyncVectorEnv([DummyContinuousEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent_continuous")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        agent = MightyPPOAgent(
            output_dir=output_dir,
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            ppo_clip=0.2,
            n_epochs=4,
            rollout_buffer_kwargs ={
                "buffer_size": 256,
            }
        )
        
        # Test initialization parameters
        assert agent.gamma == 0.99, "Gamma should be 0.99"
        assert agent.ppo_clip == 0.2, "PPO clip should be 0.2"
        assert agent.n_epochs == 4, "N epochs should be 4"
        assert agent.learning_rate == 3e-4, "Learning rate should be 3e-4"
        
        # Test that model is initialized
        assert agent.model is not None, "Model should be initialized"
        assert agent.model.continuous_action is True, "Should be continuous action"
        assert agent.update_fn is not None, "Update function should be initialized"
        
        # Test basic step functionality
        test_obs, _ = env.reset()
        metrics = {
            "env": agent.env,
            "step": 0,
            "hp/lr": agent.learning_rate,
            "hp/pi_epsilon": agent._epsilon,
            "hp/batch_size": agent._batch_size,
            "hp/learning_starts": agent._learning_starts,
        }
        
        
        
        
        prediction = agent.step(test_obs, metrics)[0]
        assert len(prediction) == 1, "Prediction should have shape (1, 2)"
        assert prediction.shape[1] == 2, "Action dimension should be 2"
    
        clean(output_dir)
    
    def test_init_discrete(self):
        """Test PPO agent initialization with discrete actions."""
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent_discrete")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        agent = MightyPPOAgent(
            output_dir=output_dir,
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            ppo_clip=0.2,
            n_epochs=4,
            rollout_buffer_kwargs ={
                "buffer_size": 256,
            }
        )
        
        # Test initialization parameters
        assert agent.model.continuous_action is False, "Should be discrete action"
        assert agent.discrete_action is True, "Discrete action flag should be True"
        
        # Test basic step functionality
        test_obs, _ = env.reset()
        metrics = {
            "env": agent.env, 
            "step": 0, 
            "hp/lr": agent.learning_rate,
            "hp/pi_epsilon": agent._epsilon,
            "hp/batch_size": agent._batch_size,
            "hp/learning_starts": agent._learning_starts,
        }
        
        prediction = agent.step(test_obs, metrics)[0]
        assert len(prediction) == 1, "Prediction should have shape (1,)"
        assert 0 <= prediction[0] < 4, "Action should be in valid range [0, 4)"  # Updated for 4 actions
        
        clean(output_dir)
    
    def test_update(self):
        """Test PPO agent update functionality with manual data collection."""
        torch.manual_seed(0)
        env = gym.vector.SyncVectorEnv([DummyContinuousEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent_update_simple")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        agent = MightyPPOAgent(
            output_dir=output_dir,
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            ppo_clip=0.2,
            n_epochs=2,
            batch_size=64,
            rollout_buffer_kwargs={
                "buffer_size": 128,
            }
        )
        
        metrics = {
            "env": agent.env, 
            "step": 0, 
            "hp/lr": agent.learning_rate,
            "hp/pi_epsilon": agent._epsilon,
            "hp/batch_size": agent._batch_size,
            "hp/learning_starts": agent._learning_starts,
        }
       
        
        # Store original parameters
        original_params = deepcopy(list(agent.model.policy_head.parameters()))
        original_value_params = deepcopy(list(agent.model.value_head.parameters()))
        
        # Manually collect data to fill the buffer
        curr_s, _ = env.reset(seed=42)
        
        
        # Collect enough transitions to fill the buffer
        for step in range(128):  # Fill buffer with 128 transitions
            # Get action from agent
            action, log_prob = agent.step(curr_s, metrics)
            
            # Take environment step
            next_s, reward, terminated, truncated, _ = env.step(action)
            dones = np.logical_or(terminated, truncated)
            
            # Process the transition (this adds to buffer)
            transition_metrics = agent.process_transition(
                curr_s,
                action,
                reward,
                next_s,
                dones,
                log_prob.detach().cpu().numpy(),
                {"step": step}
            )
            
            # Update current state
            curr_s = next_s
            
            # Reset environment if done
            if np.any(dones):
                curr_s, _ = env.reset()
        
        print(f"Buffer size after manual collection: {len(agent.buffer)}")
        
        # Ensure we have enough data in buffer
        assert len(agent.buffer) >= agent._batch_size, f"Buffer size {len(agent.buffer)} should be >= batch size {agent._batch_size}"
        
        # Perform update
        
        
        update_kwargs = {
            "next_s": curr_s,
            "dones": np.array([False])
        }
        
        # Call the update method
        result_metrics = agent.update(metrics, update_kwargs)
        
        print(f"Update completed. Returned metrics keys: {list(result_metrics.keys())}")
        
        # Check that parameters have changed
        new_params = list(agent.model.policy_head.parameters())
        new_value_params = list(agent.model.value_head.parameters())
        
        # Check if policy parameters changed
        params_changed = False
        for old, new in zip(original_params, new_params):
            if not torch.allclose(old, new, atol=1e-6):
                params_changed = True
                print(f"Policy parameter changed: max diff = {torch.max(torch.abs(old - new)).item()}")
                break
        
        # Check if value parameters changed
        value_params_changed = False
        for old, new in zip(original_value_params, new_value_params):
            if not torch.allclose(old, new, atol=1e-6):
                value_params_changed = True
                print(f"Value parameter changed: max diff = {torch.max(torch.abs(old - new)).item()}")
                break
        
        # Assertions
        assert params_changed, "Policy parameters should change after update"
        assert value_params_changed, "Value parameters should change after update"
        assert isinstance(result_metrics, dict), "Update should return metrics dict"
        
        # Check for expected metrics in the result
        expected_metrics = ["Update/policy_loss", "Update/value_loss", "Update/entropy", "Update/approx_kl"]
        for metric in expected_metrics:
            assert metric in result_metrics, f"Should have {metric} metric"
            print(f"{metric}: {result_metrics[metric]}")
        
        print("All parameter update checks passed!")
        
        clean(output_dir) 
    
    def test_properties(self):
        """Test agent properties."""
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        output_dir = Path("test_ppo_agent_properties")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ppo = MightyPPOAgent(
            output_dir=output_dir,
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            ppo_clip=0.2,
            n_epochs=4,
            rollout_buffer_kwargs ={
                "buffer_size": 256,
            }
        )
        
        # Test parameters property
        params = ppo.parameters
        assert isinstance(params, list), "Parameters should be a list"
        assert len(params) > 0, "Should have parameters"
        assert all(isinstance(p, torch.nn.Parameter) for p in params), "All should be Parameters"
        
        # Test value_function property
        value_fn = ppo.value_function
        assert isinstance(value_fn, Callable), "Value function should be callable"
        
        clean(output_dir)