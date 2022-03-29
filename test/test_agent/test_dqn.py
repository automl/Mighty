import unittest
from unittest.mock import MagicMock

from mighty.agent.dqn import DQNAgent
from .mock_environment import MockEnvDiscreteActions


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        env = MockEnvDiscreteActions()
        self.dqn = DQNAgent(env=env, eval_env=env, epsilon=.1,
                            batch_size=4, logger=MagicMock(), log_tensorboard=False)
        self.assertFalse(self.dqn.q is None)
        self.assertFalse(self.dqn.q_target is None)
        self.assertFalse(self.dqn.policy is None)
        self.assertFalse(self.dqn.replay_buffer is None)
        self.assertFalse(self.dqn.tracer is None)
        self.assertFalse(self.dqn.qlearning is None)

    def testGetState(self):
        q_params, q_state, target_params, target_state = self.dqn.get_state()
        self.assertFalse(q_params is None)
        self.assertFalse(target_params is None)
        self.assertFalse(q_state is None)
        self.assertFalse(target_state is None)

    def testSetState(self):
        self.dqn.set_state((1, 2, 3, 4))
        self.assertTrue(self.dqn.q.params == 1)
        self.assertTrue(self.dqn.q.function_state == 2)
        self.assertTrue(self.dqn.q_target.params == 3)
        self.assertTrue(self.dqn.q_target.function_state == 4)

    def testUpdate(self):
        self.dqn.tracer.add(0, 0, 5, False)
        self.dqn.tracer.add(0, 1, -1, False)
        self.dqn.tracer.add(0, 0, 5, False)
        self.dqn.tracer.add(0, 0, 5, False)
        self.dqn.tracer.add(0, 1, -1, False)
        self.dqn.tracer.add(0, 0, 5, False)
        self.dqn.tracer.add(0, 1, -1, False)
        self.dqn.tracer.add(0, 1, -1, True)
        while self.dqn.tracer:
            self.dqn.replay_buffer.add(self.dqn.tracer.pop())

        q_previous = self.dqn.q.copy(True)
        target_previous = self.dqn.q_target.copy(True)
        self.dqn.update_agent(1)
        self.assertFalse(self.dqn.q.params == q_previous.params)
        self.assertTrue(self.dqn.q_target.params == target_previous.params)

        self.dqn.update_agent(10)
        self.assertFalse(self.dqn.q_target.params == target_previous.params)

if __name__ == '__main__':
    unittest.main()
