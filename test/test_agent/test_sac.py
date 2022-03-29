import unittest
from unittest.mock import MagicMock

import numpy as np
from mighty.agent.sac import SACAgent
from .mock_environment import MockEnvDiscreteActions


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        env = MockEnvDiscreteActions()
        self.sac = SACAgent(env=env, eval_env=env, epsilon=.1,
                            batch_size=4, logger=MagicMock(), log_tensorboard=False)
        self.assertFalse(self.sac.q1 is None)
        self.assertFalse(self.sac.q1_target is None)
        self.assertFalse(self.sac.q2 is None)
        self.assertFalse(self.sac.q2_target is None)
        self.assertFalse(self.sac.policy is None)
        self.assertFalse(self.sac.replay_buffer is None)
        self.assertFalse(self.sac.tracer is None)
        self.assertFalse(self.sac.qlearning1 is None)
        self.assertFalse(self.sac.qlearning2 is None)
        self.assertFalse(self.sac.soft_pg is None)
        self.assertFalse(self.sac.policy_regularizer is None)

    def testGetState(self):
        policy_dist, policy_state, q1_params, q1_state, q2_params, q2_state, target1_params, target1_state, target2_params, target2_state = self.sac.get_state()
        self.assertFalse(policy_dist is None)
        self.assertFalse(policy_state is None)
        self.assertFalse(q1_params is None)
        self.assertFalse(target1_params is None)
        self.assertFalse(q1_state is None)
        self.assertFalse(target1_state is None)
        self.assertFalse(q2_params is None)
        self.assertFalse(target2_params is None)
        self.assertFalse(q2_state is None)
        self.assertFalse(target2_state is None)

    def testSetState(self):
        self.sac.set_state((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        self.assertTrue(self.sac.policy.proba_dist == 1)
        self.assertTrue(self.sac.policy.function_state == 2)
        self.assertTrue(self.sac.q1.params == 3)
        self.assertTrue(self.sac.q1.function_state == 4)
        self.assertTrue(self.sac.q2.params == 5)
        self.assertTrue(self.sac.q2.function_state == 6)
        self.assertTrue(self.sac.q1_target.params == 7)
        self.assertTrue(self.sac.q1_target.function_state == 8)
        self.assertTrue(self.sac.q1_target.params == 9)
        self.assertTrue(self.sac.q1_target.function_state == 10)

    def testUpdate(self):
        self.sac.tracer.add(0, 0, 5, False)
        self.sac.tracer.add(0, 1, -1, False)
        self.sac.tracer.add(0, 0, 5, False)
        self.sac.tracer.add(0, 0, 5, False)
        self.sac.tracer.add(0, 1, -1, False)
        self.sac.tracer.add(0, 0, 5, False)
        self.sac.tracer.add(0, 1, -1, False)
        self.sac.tracer.add(0, 1, -1, True)
        while self.sac.tracer:
            self.sac.replay_buffer.add(self.sac.tracer.pop())

        q1_previous = self.sac.q1.copy(True)
        target1_previous = self.sac.q1_target.copy(True)
        q2_previous = self.sac.q2.copy(True)
        target2_previous = self.sac.q2_target.copy(True)
        policy_previous = self.sac.policy.copy(True)

        self.sac.update_agent(1)

        # Check that policy and v have been updated
        self.assertFalse(self.sac.q1.params == q1_previous.params and self.sac.q2.params == q2_previous.params)
        self.assertFalse(self.sac.q1.params != q1_previous.params and self.sac.q2.params != q2_previous.params)
        self.assertFalse(self.sac.policy.params == policy_previous.params)

        # Check that target updates are small enough
        self.assertTrue(np.abs(target1_previous.params * self.sac.soft_update_weight) >= np.abs(
            self.sac.q1_target.params - target1_previous.params))
        self.assertTrue(np.abs(target2_previous.params * self.sac.soft_update_weight) >= np.abs(
            self.sac.q2_target.params - target2_previous.params))

if __name__ == '__main__':
    unittest.main()
