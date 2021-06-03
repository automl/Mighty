import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from mighty.agent.ddqn import DDQNAgent


class MyTestCase(unittest.TestCase):

    # TODO properly mock environment
    @patch('mighty.agent.ddqn.optim.Adam')
    @patch('mighty.agent.ddqn.FullyConnectedQ')
    def setUp(self, mocked_fully_connected_q, mocked_optimizer) -> None:
        self.ddqn = DDQNAgent(env=MagicMock(), env_eval=MagicMock(),
                              gamma=.99, epsilon=.1, batch_size=64, logger=MagicMock())
        mocked_optimizer.assert_called_once()



if __name__ == '__main__':
    unittest.main()
