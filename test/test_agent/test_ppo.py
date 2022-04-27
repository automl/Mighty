import unittest
from unittest.mock import MagicMock

import numpy as np
from mighty.agent.ppo import PPOAgent
from copy import deepcopy
import gym


class TestPPO(unittest.TestCase):
    def setUp(self) -> None:
        env = gym.make("Pendulum-v1")
        self.ppo = PPOAgent(
            env=env,
            eval_env=env,
            epsilon=0.1,
            batch_size=4,
            logger=MagicMock(),
            log_tensorboard=False,
        )
        self.assertFalse(self.ppo.v is None)
        self.assertFalse(self.ppo.v_targ is None)
        self.assertFalse(self.ppo.policy is None)
        self.assertFalse(self.ppo.pi_old is None)
        self.assertFalse(self.ppo.ppo_clip is None)
        self.assertFalse(self.ppo.td_update is None)

    def testGetState(self):
        (
            v_params,
            v_state,
            target_params,
            target_state,
            policy_params,
            policy_state,
            pold_params,
            pold_state,
        ) = self.ppo.get_state()
        self.assertFalse(v_params is None)
        self.assertFalse(target_params is None)
        self.assertFalse(policy_params is None)
        self.assertFalse(pold_params is None)
        self.assertFalse(v_state is None)
        self.assertFalse(target_state is None)
        self.assertFalse(policy_state is None)
        self.assertFalse(pold_state is None)

    def testSetState(self):
        altered_v_params = deepcopy(self.ppo.v.params)
        altered_v_params["linear"]["b"] += 1
        altered_v_target_params = deepcopy(self.ppo.v_targ.params)
        altered_v_target_params["linear"]["b"] += 2
        altered_policy_params = deepcopy(self.ppo.policy.params)
        altered_policy_params["linear"]["b"] += 3
        altered_pi_old_params = deepcopy(self.ppo.pi_old.params)
        altered_pi_old_params["linear"]["b"] += 4

        self.ppo.set_state(
            (
                altered_v_params,
                self.ppo.v.function_state,
                altered_v_target_params,
                self.ppo.v_targ.function_state,
                altered_policy_params,
                self.ppo.policy.function_state,
                altered_pi_old_params,
                self.ppo.pi_old.function_state,
            )
        )
        self.assertTrue(self.ppo.v.params["linear"]["b"][0] == 1)
        self.assertTrue(self.ppo.v_targ.params["linear"]["b"][0] == 2)
        self.assertTrue(self.ppo.policy.params["linear"]["b"][0] == 3)
        self.assertTrue(self.ppo.pi_old.params["linear"]["b"][0] == 4)

    def testUpdate(self):
        q_previous = deepcopy(self.ppo.v.params)
        target_previous = deepcopy(self.ppo.v_targ.params)
        policy_previous = deepcopy(self.ppo.policy.params)
        policy_target_previous = deepcopy(self.ppo.pi_old.params)

        # self.ppo.update_agent(1)
        self.ppo.train(10, 1)

        # Check that policy and v have been updated
        self.assertFalse(
            np.all(self.ppo.v.params["linear_3"]["w"] == q_previous["linear_3"]["w"])
        )
        self.assertFalse(
            np.all(
                self.ppo.policy.params["linear_3"]["w"]
                == policy_previous["linear_3"]["w"]
            )
        )

        # Check that target updates are small enough
        self.assertTrue(
            np.all(
                (
                    np.abs(target_previous["linear_3"]["w"] + 1)
                    * self.ppo.soft_update_weight
                    * 15
                )
                >= np.abs(
                    self.ppo.v_targ.params["linear_3"]["w"]
                    - target_previous["linear_3"]["w"]
                    + 1
                )
            )
        )
        self.assertTrue(
            np.all(
                np.abs(
                    (policy_target_previous["linear_3"]["w"] + 1)
                    * self.ppo.soft_update_weight
                    * 15
                )
                >= np.abs(
                    self.ppo.pi_old.params["linear_3"]["w"]
                    - policy_target_previous["linear_3"]["w"]
                    + 1
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
