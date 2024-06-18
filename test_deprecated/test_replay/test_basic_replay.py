import unittest
from unittest.mock import MagicMock

from mighty.agent.dqn_deprecated import MightyDQNAgent
from test.mock_environment import MockEnvDiscreteActions
from coax.reward_tracing import TransitionBatch
import numpy as np


class TestBaseAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MockEnvDiscreteActions()
        self.dqn = self.make_dqn(False, False)

    def make_dqn(self, keep_info, flatten_info):
        return MightyDQNAgent(
            env=self.env,
            eval_env=self.env,
            epsilon=0.1,
            batch_size=4,
            logger=MagicMock(),
            log_tensorboard=False,
            replay_buffer_kwargs={
                "capacity": 1000000,
                "random_seed": 0,
                "keep_infos": keep_info,
                "flatten_infos": flatten_info,
            },
        )

    def test_add(self) -> None:
        self.dqn = self.make_dqn(False, False)
        state, _ = self.env.reset()
        action = self.dqn.policy(state)
        _, reward, term, trunc, _ = self.env.step(action)
        transition = TransitionBatch.from_single(
            state, action, -0.2, reward, (term or trunc), 0.9
        )
        transition.extra_info = {"test": 1}
        self.dqn.replay_buffer.add(transition, {})
        assert self.dqn.replay_buffer._storage[0].A == action
        assert np.array_equal(self.dqn.replay_buffer._storage[0].S, [state])
        assert self.dqn.replay_buffer._storage[0].extra_info == []

    def test_flatten(self) -> None:
        self.dqn = self.make_dqn(True, True)
        state, _ = self.env.reset()
        action = self.dqn.policy(state)
        _, reward, term, trunc, _ = self.env.step(action)
        transition = TransitionBatch.from_single(
            state, action, -0.2, reward, (term or trunc), 0.9
        )
        transition.extra_info = {"test": 1}
        self.dqn.replay_buffer.add(transition, {})
        assert self.dqn.replay_buffer._storage[0].A == action
        assert np.array_equal(self.dqn.replay_buffer._storage[0].S, [state])
        assert self.dqn.replay_buffer._storage[0].extra_info == [[1]]

    def test_infos(self) -> None:
        self.dqn = self.make_dqn(True, False)
        state, _ = self.env.reset()
        action = self.dqn.policy(state)
        _, reward, term, trunc, _ = self.env.step(action)
        transition = TransitionBatch.from_single(
            state, action, -0.2, reward, (term or trunc), 0.9
        )
        transition.extra_info = {"test": 1}
        self.dqn.replay_buffer.add(transition, {})
        assert self.dqn.replay_buffer._storage[0].A == action
        assert np.array_equal(self.dqn.replay_buffer._storage[0].S, [state])
        assert self.dqn.replay_buffer._storage[0].extra_info == {"test": 1}
