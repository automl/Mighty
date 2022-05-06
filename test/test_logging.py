import unittest

from mighty.agent.dqn import DQNAgent
from mighty.utils.logger import Logger
from .test_agent.mock_environment import MockEnvDiscreteActions


class TestBaseAgent(unittest.TestCase):
    def setUp(self) -> None:
        env = MockEnvDiscreteActions()
        logger = Logger(
            experiment_name=f"test",
            step_write_frequency=10,
            episode_write_frequency=None,
            log_to_wandb=False,
            log_to_tensorboad=False,
            hydra_config=None
        )

        self.assertTrue(logger.instance is None)
        self.assertFalse(logger.log_file is None)
        self.assertFalse(logger.eval)
        self.assertTrue(logger.step == 0)
        self.assertTrue(logger.episode == 0)
        self.assertTrue(len(logger.buffer) == 0)
        self.assertTrue(len(logger.current_step.keys()) == 0)

        # Use DQN to check basic functionality
        self.agent = DQNAgent(
            env=env,
            eval_env=env,
            epsilon=0.1,
            batch_size=4,
            logger=logger,
            log_tensorboard=False,
        )

        self.assertFalse(self.agent.logger is None)

    def testEvalSwitch(self):
        self.assertFalse(self.agent.logger.eval)
        self.assertTrue(self.agent.logger.reward_log_file == self.agent.logger.log_file)
        self.agent.logger.set_eval(True)
        self.assertTrue(self.agent.logger.eval)
        self.assertTrue(self.agent.logger.eval_log_file == self.agent.logger.log_file)
        self.agent.logger.set_eval(False)
        self.assertFalse(self.agent.logger.eval)
        self.assertTrue(self.agent.logger.reward_log_file == self.agent.logger.log_file)


if __name__ == "__main__":
    unittest.main()