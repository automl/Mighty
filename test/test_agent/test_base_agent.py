import unittest
from unittest.mock import MagicMock

from mighty.agent.base_agent import MightyAgent
from mighty.agent.dqn import MightyDQNAgent
from .mock_environment import MockEnvDiscreteActions


class TestBaseAgent(unittest.TestCase):
    def setUp(self) -> None:
        env = MockEnvDiscreteActions()
        with self.assertRaises(NotImplementedError):
            self.agent = MightyAgent(
                env=env,
                eval_env=env,
                epsilon=0.1,
                batch_size=4,
                logger=MagicMock(),
                log_tensorboard=False,
            )

        # Use DQN to check basic functionality
        self.agent = MightyDQNAgent(
            env=env,
            eval_env=env,
            epsilon=0.1,
            batch_size=4,
            logger=MagicMock(),
            log_tensorboard=False,
        )

    def testEval(self):
        self.agent.eval(MockEnvDiscreteActions(), 10)

    def testTrain(self):
        self.agent.train(n_steps=2, n_episodes_eval=10)
        self.assertTrue(len(self.agent.replay_buffer) > 0)


if __name__ == "__main__":
    unittest.main()
