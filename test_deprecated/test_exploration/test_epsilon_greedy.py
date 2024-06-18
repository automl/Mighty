import unittest
from unittest.mock import MagicMock

from mighty.agent.dqn_deprecated import MightyDQNAgent
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
        )
        self.assertFalse(self.dqn.q is None)
        self.assertFalse(self.dqn.q_target is None)
        self.assertFalse(self.dqn.policy is None)
        self.assertFalse(self.dqn.replay_buffer is None)
        self.assertFalse(self.dqn.tracer is None)
        self.assertFalse(self.dqn.qlearning is None)

    def test_eval(self) -> None:
        """Test that epsilon is 0 in eval mode."""
        state, _ = self.env.reset()

        def get_zero(arg, arg2, another_arg, more_arg, last_arg):
            return ({"logits": jnp.array([[0.6, 0.1, 0.1, 0.1, 0.1]])}), None

        self.dqn.policy._function = get_zero
        actions = [self.dqn.policy(state, eval=True) for _ in range(20)]
        assert all(x == actions[0] for x in actions)

    def test_sampling(self) -> None:
        """Make sure that actions are changed when using positive epsilon."""
        state, _ = self.env.reset()

        def get_zero(arg, arg2, another_arg, more_arg, last_arg):
            return ({"logits": jnp.array([[0.6, 0.1, 0.1, 0.1, 0.1]])}), None

        self.dqn.policy._function = get_zero
        actions = [self.dqn.policy(state) for _ in range(20)]
        assert not all(x == actions[0] for x in actions)
