from __future__ import annotations
import shutil
from omegaconf import OmegaConf
from copy import deepcopy
from mighty.mighty_runners import MightyRunner, MightyNESRunner
from mighty.mighty_agents import MightyAgent
from mighty.utils.logger import Logger
from mighty.utils.wrappers import PufferlibToGymAdapter

class TestMightyNESRunner:
    runner_config = OmegaConf.create(
        {
            "runner": "nes",
            "es": "xnes",
            "iterations": 2,
            "popsize": 3,
            "debug": False,
            "seed": 0,
            "output_dir": "test_nes_runner",
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
        runner = MightyNESRunner(self.runner_config)
        assert isinstance(
            runner, MightyRunner
        ), "MightyNESRunner should be an instance of MightyRunner"
        assert isinstance(runner.agent, MightyAgent), "MightyNESRunner should have a MightyAgent"
        assert isinstance(runner.logger, Logger), "MightyNESRunner should have a Logger"
        assert isinstance(runner.agent.eval_env, PufferlibToGymAdapter), "Eval env should be a PufferlibToGymAdapter"
        assert runner.agent.env is not None, "Env should be set"
        assert runner.iterations is not None, "Iterations should be set"
        assert runner.es is not None, "ES should be set"
        assert runner.fit_shaper is not None, "Fit shaper should be set"
        assert runner.rng is not None, "RNG should be set"

    def test_run(self):
        runner = MightyNESRunner(self.runner_config)
        old_params = deepcopy(runner.agent.parameters)
        train_results, eval_results = runner.run()
        new_params = runner.agent.parameters
        assert isinstance(train_results, dict), "Train results should be a dictionary"
        assert isinstance(eval_results, dict), "Eval results should be a dictionary"
        assert "mean_eval_reward" in eval_results, "Mean eval reward should be in eval results"
        param_equals = [o==p for o,p in zip(old_params,new_params)]
        for params in param_equals:
            assert not all(params.flatten()), "Parameters should have changed in training"
        shutil.rmtree("test_nes_runner")