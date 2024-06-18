import unittest
from unittest.mock import MagicMock

import numpy as np
from mighty.agent.sac_deprecated import MightySACAgent

from copy import deepcopy
import gymnasium as gym


class TestSAC(unittest.TestCase):
    def setUp(self) -> None:
        env = gym.make("Pendulum-v1")
        buffer_kwargs = {"capacity": 1000000, "random_seed": 0, "keep_infos": True}
        self.sac = MightySACAgent(
            env=env,
            eval_env=env,
            epsilon=0.1,
            batch_size=4,
            logger=MagicMock(),
            log_tensorboard=False,
            replay_buffer_kwargs=buffer_kwargs,
        )
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
        (
            policy_dist,
            policy_state,
            q1_params,
            q1_state,
            q2_params,
            q2_state,
            target1_params,
            target1_state,
            target2_params,
            target2_state,
        ) = self.sac.get_state()
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
        altered_q1_params = deepcopy(self.sac.q1.params)
        altered_q1_params["linear"]["b"] += 1
        altered_q1_target_params = deepcopy(self.sac.q1_target.params)
        altered_q1_target_params["linear"]["b"] += 3
        altered_q2_params = deepcopy(self.sac.q2.params)
        altered_q2_params["linear"]["b"] += 2
        altered_q2_target_params = deepcopy(self.sac.q2_target.params)
        altered_q2_target_params["linear"]["b"] += 4

        self.sac.set_state(
            (
                self.sac.policy.proba_dist,
                self.sac.policy.function_state,
                altered_q1_params,
                self.sac.q1.function_state,
                altered_q2_params,
                self.sac.q2.function_state,
                altered_q1_target_params,
                self.sac.q1_target.function_state,
                altered_q2_target_params,
                self.sac.q1_target.function_state,
            )
        )
        self.assertTrue(self.sac.q1.params["linear"]["b"][0] == 1)
        self.assertTrue(self.sac.q2.params["linear"]["b"][0] == 2)
        self.assertTrue(self.sac.q1_target.params["linear"]["b"][0] == 3)
        self.assertTrue(self.sac.q2_target.params["linear"]["b"][0] == 4)

    def testUpdate(self):
        q1_previous = deepcopy(self.sac.q1.params)
        target1_previous = deepcopy(self.sac.q1_target.params)
        q2_previous = deepcopy(self.sac.q2.params)
        target2_previous = deepcopy(self.sac.q2_target.params)
        policy_previous = deepcopy(self.sac.policy.params)

        self.sac.train(100, 1)

        # Check that policy and v have been updated
        self.assertFalse(
            np.all(self.sac.q1.params["linear_3"]["w"] == q1_previous["linear_3"]["w"])
            and np.all(
                self.sac.q2.params["linear_3"]["w"] == q2_previous["linear_3"]["w"]
            )
        )
        self.assertFalse(
            np.all(
                self.sac.policy.params["linear_3"]["w"]
                == policy_previous["linear_3"]["w"]
            )
        )

        # Check that target updates are small enough
        self.assertTrue(
            np.all(
                np.abs(
                    (target1_previous["linear_3"]["w"] + 1)
                    * self.sac.soft_update_weight
                    * 15
                )
                >= np.abs(
                    self.sac.q1_target.params["linear_3"]["w"]
                    - target1_previous["linear_3"]["w"]
                    + 1
                )
            )
        )
        self.assertTrue(
            np.all(
                np.abs(
                    (target2_previous["linear_3"]["w"] + 1)
                    * self.sac.soft_update_weight
                    * 15
                )
                >= np.abs(
                    self.sac.q2_target.params["linear_3"]["w"]
                    - target2_previous["linear_3"]["w"]
                    + 1
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
