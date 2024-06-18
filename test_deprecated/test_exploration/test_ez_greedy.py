import unittest
from unittest.mock import MagicMock

from mighty.agent.dqn_deprecated import MightyDQNAgent
from mighty.mighty_exploration import EZGreedy
from test.mock_environment import MockEnvDiscreteActions
import jax.numpy as jnp


class TestBaseAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MockEnvDiscreteActions()
        self.dqn = MightyDQNAgent(
            env=self.env,
            eval_env=self.env,
            epsilon=0.1,
            batch_size=4,
            logger=MagicMock(),
            log_tensorboard=False,
            policy_class=EZGreedy,
        )
        self.assertFalse(self.dqn.q is None)
        self.assertFalse(self.dqn.q_target is None)
        self.assertFalse(self.dqn.policy is None)
        self.assertFalse(self.dqn.replay_buffer is None)
        self.assertFalse(self.dqn.tracer is None)
        self.assertFalse(self.dqn.qlearning is None)

    def test_skip(self) -> None:
        state, _ = self.env.reset()

        def get_zero(arg, arg2, another_arg, more_arg, last_arg):
            return ({"logits": jnp.array([[0.2, 0.2, 0.2, 0.2, 0.2]])}), None

        self.dqn.policy._function = get_zero
        self.dqn.policy(state)
        skip_value = self.dqn.policy.skip
        actions = [self.dqn.policy(state) for _ in range(skip_value)]
        assert all(x == actions[0] for x in actions)

        matches = True
        skip_match = True
        next_action = self.dqn.policy(state)
        matches = matches and next_action == actions[-1]
        skip_match = skip_match and skip_value == self.dqn.policy.skip
        for _ in range(10):
            skip_value = self.dqn.policy.skip
            actions = [self.dqn.policy(state) for _ in range(skip_value)]
            next_action = self.dqn.policy(state)
            print(actions)
            matches = matches and next_action == actions[-1]
            skip_match = skip_match and skip_value == self.dqn.policy.skip
        assert not matches
        assert not skip_match
