import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import call

import numpy as np
from torch.autograd import Variable
from torch import from_numpy as tensor_from_numpy

from mighty.agent.ddqn import DDQNAgent
from .mock_environment import MockEnvDiscreteActions


class MyTestCase(unittest.TestCase):

    @patch('mighty.agent.ddqn.optim.Adam')
    def setUp(self, mocked_optimizer) -> None:
        env = MockEnvDiscreteActions()
        np.random.seed(12345)
        self.ddqn = DDQNAgent(env=env, env_eval=env,
                              gamma=.99, epsilon=.1, batch_size=64, logger=MagicMock(),
                              eval_logger=MagicMock(), log_tensorboard=False)
        self.env = env
        mocked_optimizer.assert_called_once()

    @patch('mighty.agent.ddqn.FullyConnectedQ.forward',
           return_value=Variable(tensor_from_numpy(np.array([0, 1, 100, 3, 4])).float(), requires_grad=False),
           autospec=True)
    def testEpsilonGreedy(self, mocked_forward):
        # check if epsilon 0 we return the action with the highest Q value
        a = self.ddqn.get_action(np.array([0, 0]), 0)
        self.assertEqual(a, 2)
        self.assertTrue(mocked_forward.called)

        # See if we uniformly sample action given epsilon = 1
        acts = []
        for _ in range(10000):
            acts.append(self.ddqn.get_action(np.array([0, 0]), 1))
        vals, counts = np.unique(acts, return_counts=True)
        print(vals, counts)
        for a in vals:
            self.assertTrue(0 <= a < self.env.action_space.n)

    def testEvalOnce(self):
        steps, rewards, decisions, policies = self.ddqn.eval(self.env, episodes=1)
        # make sure we have the correct number of episodes
        self.assertTrue(len(steps) == 1)
        # make sure we have the correct number of steps per episode
        self.assertTrue(steps[0] == rewards[0] == 10)
        self.assertTrue(decisions[0] == steps[0])  # DDQN always makes a decision at every step

    def testEvalMulti(self):
        steps, rewards, decisions, policies = map(np.array, self.ddqn.eval(self.env, episodes=10))
        # make sure we have the correct number of episodes
        self.assertTrue(len(steps) == 10)
        # make sure we have the correct number of steps per episode
        self.assertTrue(np.all(steps == 10))
        self.assertTrue(np.all(decisions == steps))  # DDQN always makes a decision at every step

    def testStartEpisode(self):
        self.ddqn.start_episode(engine=None)
        # Test if the last state is now set to the initial reset state of the mock environment
        self.assertTrue(np.all(self.ddqn.last_state == np.array([0, 0])))

    def testStep(self):
        self.ddqn.start_episode(engine=None)
        s = self.ddqn.step()
        self.assertTrue(self.ddqn._replay_buffer._size == 1)
        # Assert that we record the origin state in the replay buffer and the correct next state
        self.assertTrue(np.all(self.ddqn._replay_buffer._data.states[0] == np.array([0, 0])))
        self.assertTrue(np.all(self.ddqn._replay_buffer._data.next_states[0] == np.array([1, 0])))
        self.assertEqual(self.ddqn._replay_buffer._data.rewards[0], 1)
        self.assertFalse(self.ddqn._replay_buffer._data.terminal_flags[0])
        self.assertTrue(np.all(self.ddqn.last_state == np.array([1, 0])))
        self.assertTrue(np.all(s == np.array([1, 0])))

        self.ddqn.step()
        self.ddqn.step()
        s = self.ddqn.step()
        self.assertTrue(self.ddqn._replay_buffer._size == 4)
        # Assert that we record the origin state in the replay buffer and the correct next state
        self.assertTrue(np.all(self.ddqn._replay_buffer._data.states[-1] == np.array([3, 0])))
        self.assertTrue(np.all(self.ddqn._replay_buffer._data.next_states[-1] == np.array([4, 0])))
        self.assertEqual(self.ddqn._replay_buffer._data.rewards[-1], 1)
        self.assertFalse(self.ddqn._replay_buffer._data.terminal_flags[-1])
        self.assertTrue(np.all(self.ddqn.last_state == np.array([4, 0])))
        self.assertTrue(np.all(s == np.array([4, 0])))

        for i in range(6):  # Run until episode end and see if it correctly is marked as done
            self.assertFalse(self.ddqn._replay_buffer._data.terminal_flags[-1])
            self.ddqn.step()
        self.assertTrue(self.ddqn._replay_buffer._data.terminal_flags[-1])



if __name__ == '__main__':
    unittest.main()
