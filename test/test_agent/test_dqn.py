import unittest
from unittest.mock import MagicMock

from mighty.agent.dqn import MightyDQNAgent
from .mock_environment import MockEnvDiscreteActions

from copy import deepcopy
import numpy as np


class TestDQN(unittest.TestCase):
    def setUp(self) -> None:
        env = MockEnvDiscreteActions()
        self.dqn = MightyDQNAgent(
            env=env,
            eval_env=env,
            epsilon=0.1,
            batch_size=4,
            logger=MagicMock(),
            log_tensorboard=False,
        )
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
        altered_q_params = deepcopy(self.dqn.q.params)
        altered_q_params["linear"]["b"] += 1
        altered_q_target_params = deepcopy(self.dqn.q_target.params)
        altered_q_target_params["linear"]["b"] += 3

        self.dqn.set_state(
            (
                altered_q_params,
                self.dqn.q.function_state,
                altered_q_target_params,
                self.dqn.q_target.function_state,
            )
        )
        self.assertTrue(self.dqn.q.params["linear"]["b"][0] == 1)
        self.assertTrue(self.dqn.q_target.params["linear"]["b"][0] == 3)

    def testUpdate(self):
        self.dqn.tracer.add([0, 0], 1, 5, False)
        self.dqn.tracer.add([0, 1], 1, -5, False)
        self.dqn.tracer.add([0, 1], 1, -5, False)
        self.dqn.tracer.add([0, 1], 1, -5, False)
        self.dqn.tracer.add([0, 0], 1, 5, False)
        self.dqn.tracer.add([0, 0], 1, 5, False)
        self.dqn.tracer.add([0, 0], 1, 5, False)
        self.dqn.tracer.add([0, 0], 0, 5, True)
        while self.dqn.tracer:
            self.dqn.replay_buffer.add(self.dqn.tracer.pop())

        q_previous = deepcopy(self.dqn.q.params)
        target_previous = deepcopy(self.dqn.q_target.params)
        self.dqn.update_agent(8)
        self.assertFalse(
            np.all(self.dqn.q.params["linear_3"]["w"] == q_previous["linear_3"]["w"])
        )
        self.assertTrue(
            np.all(
                self.dqn.q_target.params["linear_3"]["w"]
                == target_previous["linear_3"]["w"]
            )
        )

        self.dqn.update_agent(10)
        self.assertFalse(
            np.all(
                self.dqn.q_target.params["linear_3"]["w"]
                == target_previous["linear_3"]["w"]
            )
        )


if __name__ == "__main__":
    unittest.main()
