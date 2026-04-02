from __future__ import annotations

import shutil

import gymnasium as gym
import pytest
from omegaconf import OmegaConf

from mighty.mighty_agents import MightyAgent
from mighty.mighty_runners import MightyOnlineRunner, MightyRunner
from mighty.mighty_utils.test_helpers import DummyEnv
from mighty.mighty_utils.wrappers import PufferWrapperEnv


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
            "env": "pufferlib.ocean.puffer_squared",
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
        assert isinstance(runner, MightyRunner), (
            "MightyOnlineRunner should be an instance of MightyRunner"
        )
        assert isinstance(runner.agent, MightyAgent), (
            "MightyOnlineRunner should have a MightyAgent"
        )
        assert isinstance(runner.agent.eval_env, PufferWrapperEnv), (
            "Eval env should be a PufferWrapperEnv"
        )
        assert runner.agent.env is not None, "Env should not be None"
        assert runner.eval_every_n_steps == self.runner_config.eval_every_n_steps, (
            "Eval every n steps should be set"
        )
        assert runner.num_steps == self.runner_config.num_steps, (
            "Num steps should be set"
        )

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

    def test_run(self):
        runner = MightyOnlineRunner(self.runner_config)
        train_results, eval_results = runner.run()
        assert isinstance(train_results, dict), "Train results should be a dictionary"
        assert isinstance(eval_results, dict), "Eval results should be a dictionary"
        assert "mean_eval_reward" in eval_results, (
            "Eval results should have mean_eval_reward"
        )
        shutil.rmtree("test_runner")

    def test_run_with_alternate_env(self):
        dummy_env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(3)])
        dummy_eval_func = lambda: gym.vector.SyncVectorEnv(  # noqa: E731
            [DummyEnv for _ in range(10)]
        )
        eval_default = 10
        runner = MightyOnlineRunner(
            self.runner_config,
            env=dummy_env,
            base_eval_env=dummy_eval_func,
            eval_default=eval_default,
        )
        assert isinstance(runner.agent.env.envs[0], DummyEnv), (
            "Runner env should be set to dummy_env"
        )
        assert isinstance(runner.agent.eval_env.envs[0], DummyEnv), (
            "Runner base_eval_env should be set to dummy_eval_func"
        )

        runner = MightyOnlineRunner(
            self.runner_config,
            env=None,
            base_eval_env=dummy_eval_func,
            eval_default=eval_default,
        )
        assert not isinstance(runner.agent.env.envs[0], DummyEnv), (
            "If env is None, runner env should set from config"
        )
        assert runner.agent.env is not None, "Env should not be None"

        runner = MightyOnlineRunner(
            self.runner_config,
            env=dummy_env,
            base_eval_env=None,
            eval_default=eval_default,
        )
        assert not isinstance(runner.agent.env.envs[0], DummyEnv), (
            "If base_eval_env is None, runner env should set from config"
        )
        assert runner.agent.env is not None, "Env should not be None"

        runner = MightyOnlineRunner(
            self.runner_config,
            env=dummy_env,
            base_eval_env=dummy_eval_func,
            eval_default=None,
        )
        assert not isinstance(runner.agent.env.envs[0], DummyEnv), (
            "If eval_default is None, runner env should set from config"
        )
        assert runner.agent.env is not None, "Env should not be None"
