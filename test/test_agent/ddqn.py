import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from mighty.agent.ddqn import DDQNAgent
from mock_environment import MockEnv


class MyTestCase(unittest.TestCase):

    @patch('mighty.agent.ddqn.optim.Adam')
    def setUp(self, mocked_optimizer) -> None:
        env = MockEnv
        self.ddqn = DDQNAgent(env=MagicMock(), env_eval=MagicMock(),
                              gamma=.99, epsilon=.1, batch_size=64, logger=MagicMock())
        mocked_optimizer.assert_called_once()

if __name__ == '__main__':
    unittest.main()
