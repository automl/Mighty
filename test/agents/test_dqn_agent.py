from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from utils import DummyEnv, clean
from mighty.mighty_agents.dqn import MightyDQNAgent
from mighty.mighty_exploration.epsilon_greedy import EpsilonGreedy
from mighty.mighty_models.dqn import DQN
from mighty.mighty_replay import PrioritizedReplay, TransitionBatch
from mighty.mighty_update.q_learning import DoubleQLearning, QLearning
from mighty.utils.logger import Logger


class TestDQNAgent:
    def test_init(self):
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_dqn_agent", "test_dqn_agent")
        dqn = MightyDQNAgent(env, logger, use_target=False)
        assert isinstance(dqn.q, DQN), "Model should be an instance of DQN"
        assert isinstance(dqn.value_function, DQN), "Vf should be an instance of DQN"
        assert isinstance(
            dqn.policy, EpsilonGreedy
        ), "Policy should be an instance of EpsilonGreedy"
        assert isinstance(
            dqn.qlearning, QLearning
        ), "Update should be an instance of QLearning"
        assert dqn.q_target is None, "Q_target should be None"

        test_obs, _ = env.reset()
        metrics = {
            "env": dqn.env,
            "vf": dqn.value_function,
            "policy": dqn.policy,
            "step": 0,
            "hp/lr": dqn.learning_rate,
            "hp/pi_epsilon": dqn._epsilon,
            "hp/batch_size": dqn._batch_size,
            "hp/learning_starts": dqn._learning_starts,
        }
        prediction = dqn.step(test_obs, metrics)
        assert len(prediction) == 1, "Prediction should have shape (1, 1)"
        assert max(prediction) < 4, "Prediction should be less than 4"

        dqn = MightyDQNAgent(
            env,
            logger,
            batch_size=32,
            learning_rate=0.01,
            epsilon=0.1,
            replay_buffer_class=PrioritizedReplay,
            td_update_class="mighty.mighty_update.DoubleQLearning",
            q_class=DQN,
        )
        assert isinstance(dqn.q, DQN), "Model should be an instance of IQN"
        assert isinstance(dqn.q_target, DQN), "Model should be an instance of IQN"
        test_obs = torch.tensor(test_obs)
        q_pred = dqn.q(test_obs)
        target_pred = dqn.q_target(test_obs)
        assert torch.allclose(q_pred, target_pred), "Q and Q_target should be equal"
        assert isinstance(
            dqn.replay_buffer, PrioritizedReplay
        ), "Replay buffer should be an instance of PrioritizedReplay"
        assert isinstance(
            dqn.qlearning, DoubleQLearning
        ), "Update should be an instance of DoubleQLearning"
        assert dqn._batch_size == 32, "Batch size should be 32"
        assert dqn.learning_rate == 0.01, "Learning rate should be 0.01"
        assert dqn._epsilon == 0.1, "Epsilon should be 0.1"
        clean(logger)

    def test_update(self):
        torch.manual_seed(0)
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_dqn_agent", "test_dqn_agent")
        dqn = MightyDQNAgent(env, logger, batch_size=2)
        dqn.run(10, 1)
        original_optimizer = torch.optim.Adam(dqn.q.parameters(), lr=dqn.learning_rate)
        original_params = deepcopy(list(dqn.q.parameters()))
        original_target_params = deepcopy(list(dqn.q_target.parameters()))
        original_feature_params = deepcopy(list(dqn.q.feature_extractor.parameters()))
        metrics = dqn.update_agent()
        new_params = deepcopy(list(dqn.q.parameters()))
        new_target_params = deepcopy(list(dqn.q_target.parameters()))
        for old, new in zip(original_params[:10], new_params[:10], strict=False):
            assert not torch.allclose(old, new), "Model parameters should change"
        for old, new in zip(
            original_target_params[:10], new_target_params[:10], strict=False
        ):
            assert not torch.allclose(
                old, new
            ), "Target model parameters should be changed"
        for old, new in zip(
            original_feature_params, dqn.q.feature_extractor.parameters(), strict=False
        ):
            assert not torch.allclose(old, new), "Feature parameters should be changed"
        for old, new, new_target in zip(
            original_params[:10], new_params[:10], new_target_params[:10], strict=False
        ):
            assert not torch.allclose(
                old * (1 - 0.01) + new * 0.01, new_target
            ), "Target model parameters should be scaled correctly"

        batch = dqn.replay_buffer.sample(2)
        preds, targets = dqn.qlearning.get_targets(batch, dqn.q, dqn.q_target)
        assert (
            np.mean(targets.detach().numpy() - metrics["Q-Update/td_targets"]) < 0.1
        ), "TD_targets should be equal"
        assert (
            np.mean((targets - preds).detach().numpy() - metrics["Q-Update/td_errors"])
            < 0.1
        ), "TD errors should be equal"

        original_optimizer.zero_grad()
        loss = F.mse_loss(preds, targets)
        assert (
            np.mean(loss.detach().numpy() - metrics["Q-Update/loss"]) < 0.05
        ), "Loss should be equal"
        loss.backward()
        original_optimizer.step()
        manual_params = deepcopy(list(dqn.q.parameters()))
        for manual, agent in zip(manual_params[:10], new_params[:10], strict=False):
            assert torch.allclose(
                manual, agent, atol=1e-2
            ), "Model parameters should be equal to manual update"

        clean(logger)

    def test_adapt_hps(self):
        metrics = {
            "hp/pi_epsilon": 0.5,
            "hp/lr": 0.1,
            "hp/soft_update_weight": 0.5,
            "hp/batch_size": 32,
            "hp/learning_starts": 15,
        }
        torch.manual_seed(0)
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_dqn_agent", "test_dqn_agent")
        dqn = MightyDQNAgent(env, logger, batch_size=2)
        assert dqn._epsilon == 0.1, "Epsilon should be 0.1"
        assert dqn.learning_rate == 0.01, "Learning rate should be 0.01"
        assert dqn.soft_update_weight == 0.01, "Soft update weight should be 0.01"
        assert dqn._batch_size == 2, "Batch size should be 2"
        assert dqn._learning_starts == 1, "Learning starts should be 1"
        dqn.adapt_hps(metrics)
        assert dqn._epsilon == 0.5, "Epsilon should be 0.5"
        assert dqn.learning_rate == 0.1, "Learning rate should be 0.1"
        assert dqn.soft_update_weight == 0.5, "Soft update weight should be 0.5"
        assert dqn._batch_size == 32, "Batch size should be 32"
        assert dqn._learning_starts == 15, "Learning starts should be 15"

    def test_save(self):
        pass

    def test_load(self):
        pass

    def test_get_transition_metrics(self):
        torch.manual_seed(0)
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_dqn_agent", "test_dqn_agent")
        dqn = MightyDQNAgent(env, logger, batch_size=2)
        state, _ = env.reset()
        action = dqn.policy([state])
        next_state, reward, tr, te, _ = env.step(action)
        next_action = dqn.policy([next_state])
        ff_state, ff_reward, ff_tr, ff_te, _ = env.step(next_action)
        transition = TransitionBatch(
            np.array([state, next_state]),
            np.array([action[0], next_action[0]]),
            np.array([reward, ff_reward]),
            np.array([next_state, ff_state]),
            np.array([te or tr, ff_te or ff_tr]),
        )
        metrics = dqn.get_transition_metrics(transition, {})
        assert (
            len(metrics["rollout_values"]) == 2
        ), f"One value prediction per state, got: {metrics['rollout_values']}"
        assert (
            len(metrics["td_error"]) == 2
        ), f"TD error should be computed per transition, got: {metrics['td_errors']}"

        state, _ = env.reset()
        action = dqn.policy(state)
        next_state, reward, tr, te, _ = env.step(action)
        transition = TransitionBatch(state, action, reward, next_state, te or tr)
        metrics = dqn.get_transition_metrics(transition, metrics)
        assert (
            len(metrics["rollout_values"]) == 3
        ), "New value prediction should be added"
        assert len(metrics["td_error"]) == 1, "TD error is overwritten"
        clean(logger)

    def test_act(self):
        pass
