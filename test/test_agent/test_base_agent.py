import unittest
from unittest.mock import MagicMock

from mighty.agent.base_agent import MightyAgent
from .mock_environment import MockEnvDiscreteActions

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        env = MockEnvDiscreteActions()
        self.agent = MightyAgent(env=env, eval_env=env, epsilon=.1,
                            batch_size=4, logger=MagicMock(), log_tensorboard=False)

    def testEval(self):
        self.agent.eval(MockEnvDiscreteActions(), 10)

    def testTrain(self):
        self.agent.train(n_steps=2, n_episodes_eval=10)
        self.assertTrue(len(self.agent.replay_buffer) > 0)
        with self.assertRaises(NotImplementedError):
            self.agent.train(n_steps=10, n_episodes_eval=10)


if __name__ == '__main__':
    unittest.main()
