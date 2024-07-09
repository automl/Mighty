from __future__ import annotations
import pytest
import shutil
from omegaconf import OmegaConf
from mighty.mighty_runners import MightyRunner, MightyOnlineRunner
from mighty.mighty_agents import MightyAgent
from mighty.utils.logger import Logger
from mighty.utils.wrappers import PufferlibToGymAdapter


class TestMightyRunner:
    runner_config = OmegaConf.create(
        {
            "runner": "standard",
            "debug": False,
            "seed": 0,
            "output_dir": "test_runner",
            "wandb_project": None,
            "tensorboard_file": None,
            "experiment_name": "mighty_experiment",
            "eval_every_n_steps": 1e4,
            "n_episodes_eval": 10,
            "checkpoint": None,
            "save_model_every_n_steps": 5e5,
            "num_steps": 100,
            "env": "pufferlib.environments.ocean.bandit",
            "env_kwargs": {},
            "env_wrappers": [],
            "num_envs": 1,
            "algorithm": "DQN",
            "algorithm_kwargs": {
                "n_units": 8,
                "epsilon": 0.2,
                "replay_buffer_class": "mighty.mighty_replay.PrioritizedReplay",
                "replay_buffer_kwargs": {"capacity": 1000000, "alpha": 0.6},
                "learning_rate": 0.001,
                "batch_size": 64,
                "gamma": 0.9,
                "soft_update_weight": 1.0,
                "td_update_class": "mighty.mighty_update.QLearning",
                "q_kwargs": {
                    "dueling": False,
                    "feature_extractor_kwargs": {
                        "architecture": "mlp",
                        "n_layers": 1,
                        "hidden_sizes": [32],
                    },
                    "head_kwargs": {"hidden_sizes": [32]},
                },
            },
        }
    )

    def test_init(self):
        runner = MightyOnlineRunner(self.runner_config)
        assert isinstance(
            runner, MightyRunner
        ), "MightyOnlineRunner should be an instance of MightyRunner"
        assert isinstance(
            runner.agent, MightyAgent
        ), "MightyOnlineRunner should have a MightyAgent"
        assert isinstance(
            runner.logger, Logger
        ), "MightyOnlineRunner should have a Logger"
        assert isinstance(
            runner.agent.eval_env, PufferlibToGymAdapter
        ), "Eval env should be a PufferlibToGymAdapter"
        assert runner.agent.env is not None, "Env should not be None"
        assert (
            runner.eval_every_n_steps == self.runner_config.eval_every_n_steps
        ), "Eval every n steps should be set"
        assert (
            runner.num_steps == self.runner_config.num_steps
        ), "Num steps should be set"

    def test_train(self):
        runner = MightyOnlineRunner(self.runner_config)
        results = runner.train(100)
        assert isinstance(results, dict), "Results should be a dictionary"
        alternate_env = True
        with pytest.raises(AttributeError):
            runner.train(100, alternate_env)

    def test_evaluate(self):
        runner = MightyOnlineRunner(self.runner_config)
        results = runner.evaluate()
        assert isinstance(results, dict), "Results should be a dictionary"
        assert "mean_eval_reward" in results, "Results should have mean_eval_reward"
        alternate_env = True
        with pytest.raises(AttributeError):
            runner.evaluate(alternate_env)

    def test_close(self):
        runner = MightyOnlineRunner(self.runner_config)
        assert (
            not runner.logger.reward_log_file.closed
        ), "Reward log file should be open"
        assert not runner.logger.eval_log_file.closed, "Eval log file should be open"
        runner.close()
        assert runner.logger.reward_log_file.closed, "Reward log file should be closed"
        assert runner.logger.eval_log_file.closed, "Eval log file should be closed"

    def test_run(self):
        runner = MightyOnlineRunner(self.runner_config)
        train_results, eval_results = runner.run()
        assert isinstance(train_results, dict), "Train results should be a dictionary"
        assert isinstance(eval_results, dict), "Eval results should be a dictionary"
        assert (
            "mean_eval_reward" in eval_results
        ), "Eval results should have mean_eval_reward"
        shutil.rmtree("test_runner")
