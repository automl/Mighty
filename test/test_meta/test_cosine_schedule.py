import unittest
from unittest.mock import MagicMock

from mighty.agent.dqn import MightyDQNAgent
from test.mock_environment import MockEnvDiscreteActions


class TestBaseAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MockEnvDiscreteActions()
        self.dqn = MightyDQNAgent(
            env=self.env,
            eval_env=self.env,
            epsilon=0.1,
            batch_size=4,
            logger=MagicMock(),
            log_tensorboard=False,
            meta_methods=["mighty.mighty_meta.CosineLRSchedule"],
            meta_kwargs=[
                {
                    "restart_every": 5,
                    "initial_lr": 1.5,
                    "num_decay_steps": 1000,
                    "restart_multiplier": 2,
                }
            ],
        )

    def test_decay(self) -> None:
        lr = 1.5
        for _ in range(4):
            self.dqn.train(1, 0)
            assert self.dqn.learning_rate < lr
            lr = self.dqn.learning_rate

    def test_restart(self) -> None:
        lr = 1.5
        for _ in range(5):
            self.dqn.train(1, 0)
        assert self.dqn.learning_rate == lr * 2
