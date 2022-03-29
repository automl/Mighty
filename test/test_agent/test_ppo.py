import unittest
from unittest.mock import MagicMock

import numpy as np
from mighty.agent.ppo import PPOAgent
from .mock_environment import MockEnvContinuousActions


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        env = MockEnvContinuousActions()
        self.ppo = PPOAgent(env=env, eval_env=env, epsilon=.1,
                            batch_size=4, logger=MagicMock(), log_tensorboard=False)
        self.assertFalse(self.ppo.v is None)
        self.assertFalse(self.ppo.v_targ is None)
        self.assertFalse(self.ppo.policy is None)
        self.assertFalse(self.ppo.pi_old is None)
        self.assertFalse(self.ppo.ppo_clip is None)
        self.assertFalse(self.ppo.td_update is None)

    def testGetState(self):
        v_params, v_state, target_params, target_state, policy_params, policy_state, pold_params, pold_state = self.ppo.get_state()
        self.assertFalse(v_params is None)
        self.assertFalse(target_params is None)
        self.assertFalse(policy_params is None)
        self.assertFalse(pold_params is None)
        self.assertFalse(v_state is None)
        self.assertFalse(target_state is None)
        self.assertFalse(policy_state is None)
        self.assertFalse(pold_state is None)

    def testSetState(self):
        self.ppo.set_state((1, 2, 3, 4, 5, 6, 7, 8))
        self.assertTrue(self.ppo.v.params == 1)
        self.assertTrue(self.ppo.v.function_state == 2)
        self.assertTrue(self.ppo.v_targ.params == 3)
        self.assertTrue(self.ppo.v_targ.function_state == 4)
        self.assertTrue(self.ppo.policy.params == 5)
        self.assertTrue(self.ppo.policy.function_state == 6)
        self.assertTrue(self.ppo.pi_old.params == 7)
        self.assertTrue(self.ppo.pi_old.function_state == 8)

    def testUpdate(self):
        self.ppo.tracer.add(0, 0, 5, False)
        self.ppo.tracer.add(0, 1, -1, False)
        self.ppo.tracer.add(0, 0, 5, False)
        self.ppo.tracer.add(0, 0, 5, False)
        self.ppo.tracer.add(0, 1, -1, False)
        self.ppo.tracer.add(0, 0, 5, False)
        self.ppo.tracer.add(0, 1, -1, False)
        self.ppo.tracer.add(0, 1, -1, True)
        while self.ppo.tracer:
            self.ppo.replay_buffer.add(self.ppo.tracer.pop())

        q_previous = self.ppo.v.copy(True)
        target_previous = self.ppo.v_targ.copy(True)
        policy_previous = self.ppo.policy.copy(True)
        policy_target_previous = self.ppo.pi_old.copy(True)

        self.ppo.update_agent(1)

        # Check that policy and v have been updated
        self.assertFalse(self.ppo.q.params == q_previous.params)
        self.assertFalse(self.ppo.policy.params == policy_previous.params)

        # Check that target updates are small enough
        self.assertTrue(np.abs(target_previous.params * self.ppo.soft_update_weight) >= np.abs(self.ppo.v_targ.params - target_previous.params))
        self.assertTrue(np.abs(policy_target_previous.params * self.ppo.soft_update_weight) >= np.abs(
            self.ppo.pi_old.params - policy_target_previous.params))


if __name__ == '__main__':
    unittest.main()
