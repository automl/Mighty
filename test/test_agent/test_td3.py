import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import call

import numpy as np
from torch.autograd import Variable
from torch import from_numpy as tensor_from_numpy

from mighty.agent.td3 import TD3Agent
from .mock_environment import MockEnvContinuousActions


class MyTestCase(unittest.TestCase):

    @patch('mighty.agent.td3.optim.Adam')
    def setUp(self, mocked_optimizer) -> None:
        env = MockEnvContinuousActions()
        np.random.seed(12345)
        self.td3 = TD3Agent(env=env, env_eval=env,
                              gamma=.99, epsilon=.1, batch_size=64, logger=MagicMock(), log_tensorboard=False)
        self.env = env
        mocked_optimizer.assert_called_once()

    @patch('mighty.agent.td3.Critic.forward',
           return_value=Variable(tensor_from_numpy(np.array([0, 1, 100, 3, 4])).float(), requires_grad=False),
           autospec=True)

    @patch('mighty.agent.td3.Actor.forward',
           return_value=Variable(tensor_from_numpy(np.array([0.5])).float(), requires_grad=False),
           autospec=True)

    def testStartEpisode(self):
        self.td3.start_episode(engine=None)
        # Test if the last state is now set to the initial reset state of the mock environment
        self.assertTrue(np.all(self.td3.last_state == np.array([0, 0])))

    def testStep(self):
        self.td3.start_episode(engine=None)
        s = self.td3.step()
        self.assertTrue(self.td3._replay_buffer._size == 1)
        # Assert that we record the origin state in the replay buffer and the correct next state
        self.assertTrue(np.all(self.td3._replay_buffer._data.states[0] == np.array([0, 0])))
        self.assertTrue(np.all(self.td3._replay_buffer._data.next_states[0] == np.array([1, 0])))
        self.assertEqual(self.td3._replay_buffer._data.rewards[0], 1)
        self.assertFalse(self.td3._replay_buffer._data.terminal_flags[0])
        self.assertTrue(np.all(self.td3.last_state == np.array([1, 0])))
        self.assertTrue(np.all(s == np.array([1, 0])))

        self.td3.step()
        self.td3.step()
        s = self.td3.step()
        self.assertTrue(self.td3._replay_buffer._size == 4)
        # Assert that we record the origin state in the replay buffer and the correct next state
        self.assertTrue(np.all(self.td3._replay_buffer._data.states[-1] == np.array([3, 0])))
        self.assertTrue(np.all(self.td3._replay_buffer._data.next_states[-1] == np.array([4, 0])))
        self.assertEqual(self.td3._replay_buffer._data.rewards[-1], 1)
        self.assertFalse(self.td3._replay_buffer._data.terminal_flags[-1])
        self.assertTrue(np.all(self.td3.last_state == np.array([4, 0])))
        self.assertTrue(np.all(s == np.array([4, 0])))

        for i in range(6):  # Run until episode end and see if it correctly is marked as done
            self.assertFalse(self.td3._replay_buffer._data.terminal_flags[-1])
            self.td3.step()
        self.assertTrue(self.td3._replay_buffer._data.terminal_flags[-1])



if __name__ == '__main__':
    unittest.main()
