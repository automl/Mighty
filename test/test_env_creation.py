from __future__ import annotations

import importlib
import carl
import gymnasium as gym
from dacbench import benchmarks
from omegaconf import OmegaConf
from mighty.mighty_utils.envs import (
    make_mighty_env,
    make_gym_env,
    make_dacbench_env,
    make_carl_env,
    make_procgen_env,
    make_pufferlib_env,
)
from mighty.mighty_utils.wrappers import (
    PufferlibToGymAdapter,
    CARLVectorEnvSimulator,
    ProcgenVecEnv,
)

try:
    import envpool

    ENVPOOL = True
except ImportError:
    ENVPOOL = False

try:
    import procgen  # noqa: F401

    PROCGEN = True
except ImportError:
    PROCGEN = False


class TestEnvCreation:
    gym_config = OmegaConf.create(
        {
            "env": "CartPole-v1",
            "env_kwargs": {},
            "env_wrappers": [],
            "num_envs": 1,
            "n_episodes_eval": 5,
        }
    )
    dacbench_config_benchmark = OmegaConf.create(
        {
            "env": "FunctionApproximationBenchmark",
            "env_kwargs": {
                "benchmark": True,
                "dimension": 1,
                "config_space": {"x": {"type": "float", "lower": -10, "upper": 10}},
            },
            "env_wrappers": ["mighty.utils.wrappers.MultiDiscreteActionWrapper"],
            "num_envs": 16,
            "n_episodes_eval": 2,
        }
    )
    dacbench_config = OmegaConf.create(
        {
            "env": "FunctionApproximationBenchmark",
            "env_kwargs": {
                "instance_set_path": "sigmoid_2D3M_train.csv",
                "test_set_path": "sigmoid_2D3M_train.csv",
            },
            "env_wrappers": ["mighty.utils.wrappers.MultiDiscreteActionWrapper"],
            "num_envs": 1,
            "n_episodes_eval": 1,
        }
    )
    carl_config = OmegaConf.create(
        {
            "env": "CARLBraxAnt",
            "env_kwargs": {},
            "env_wrappers": [],
            "num_envs": 10,
            "n_episodes_eval": 5,
        }
    )
    carl_config_context = OmegaConf.create(
        {
            "env": "CARLBraxAnt",
            "env_kwargs": {
                "num_contexts": 100,
                "context_feature_args": {
                    "target_distance": ["normal", 0.1, 20, 9.8, 1],
                    "target_direction": ["categorical", [1, 2, 3, 4]],
                    "friction": ["uniform-float", 0, 10],
                    "gravity": ["uniform-int", -5, 5],
                },
                "batch_size": 256,
            },
            "env_wrappers": [],
            "num_envs": 1,
            "n_episodes_eval": 1,
        }
    )
    procgen_config = OmegaConf.create(
        {
            "env": "procgen:bigfish",
            "env_kwargs": {},
            "env_wrappers": [],
            "num_envs": 256,
            "n_episodes_eval": 50,
        }
    )
    pufferlib_config = OmegaConf.create(
        {
            "env": "pufferlib.environments.ocean.memory",
            "env_kwargs": {},
            "env_wrappers": [],
            "num_envs": 10,
            "n_episodes_eval": 10,
        }
    )

    def check_vector_env(self, env):
        """Check if environment is a vector environment."""
        assert hasattr(
            env, "num_envs"
        ), f"Vector environment should have num_envs attribute: {env}"
        assert hasattr(
            env, "reset"
        ), f"Vector environment should have reset method: {env}."
        assert hasattr(
            env, "step"
        ), f"Vector environment should have step method: {env}."
        assert hasattr(
            env, "close"
        ), f"Vector environment should have close method: {env}."
        assert hasattr(
            env, "single_action_space"
        ), f"Vector environment should have single action space view: {env}."
        assert hasattr(
            env, "single_observation_space"
        ), f"Vector environment should have single observation space view: {env}."
        assert hasattr(
            env, "envs"
        ), f"Environments should be kept in envs attribute: {env}."

    def test_make_gym_env(self):
        """Test env creation with make_gym_env."""
        env, eval_env, eval_default = make_gym_env(self.gym_config)
        self.check_vector_env(env)
        self.check_vector_env(eval_env())
        assert (
            eval_default == self.gym_config.n_episodes_eval
        ), "Default number of eval episodes should match config"
        assert (
            len(env.envs) == self.gym_config.num_envs
        ), "Number of environments should match config."
        assert (
            len(eval_env().envs) == self.gym_config.n_episodes_eval
        ), "Number of environments should match config."

        assert (
            self.gym_config.env == env.envs[0].spec.id
        ), "Environment should be created with the correct id."
        assert (
            self.gym_config.env == eval_env().envs[0].spec.id
        ), "Eval environment should be created with the correct id."

        assert isinstance(
            env, gym.vector.SyncVectorEnv
        ), "Gym environment should be a SyncVectorEnv."
        assert isinstance(
            eval_env(), gym.vector.SyncVectorEnv
        ), "Eval environment should be a SyncVectorEnv."

    def test_make_dacbench_env(self):
        """Test env creation with make_dacbench_env."""
        env, eval_env, eval_default = make_dacbench_env(self.dacbench_config)
        self.check_vector_env(env)
        self.check_vector_env(eval_env())
        assert (
            eval_default
            == len(env.envs[0].instance_set.keys())
            * self.dacbench_config.n_episodes_eval
        ), "Default number of eval episodes should instance set size times evaluation episodes."
        assert (
            len(env.envs) == self.dacbench_config.num_envs
        ), "Number of environments should match config."
        assert (
            len(eval_env().envs)
            == len(env.envs[0].instance_set.keys())
            * self.dacbench_config.n_episodes_eval
        ), "Number of environments should match eval length."

        assert isinstance(
            env, gym.vector.SyncVectorEnv
        ), "DACBench environment should be a SyncVectorEnv."
        assert isinstance(
            eval_env(), gym.vector.SyncVectorEnv
        ), "Eval environment should be a SyncVectorEnv."

        bench = getattr(benchmarks, self.dacbench_config.env)()
        for k in self.dacbench_config.env_kwargs:
            bench.config[k] = self.dacbench_config.env_kwargs[k]
        assert isinstance(
            env.envs[0], type(bench.get_environment())
        ), "Environment should have correct type."
        assert isinstance(
            eval_env().envs[0], type(bench.get_environment())
        ), "Eval environment should have correct type."
        assert (
            env.envs[0].config.instance_set_path
            == self.dacbench_config.env_kwargs.instance_set_path
        ), "Environment should have correct instance set."
        assert eval_env().envs[0].test, "Eval environment should be in test mode."

    def test_make_dacbench_benchmark_mode(self):
        """Test env creation with make_dacbench_env in benchmark mode."""
        env, eval_env, eval_default = make_dacbench_env(self.dacbench_config_benchmark)
        self.check_vector_env(env)
        self.check_vector_env(eval_env())
        assert (
            eval_default
            == len(env.envs[0].instance_set.keys())
            * self.dacbench_config_benchmark.n_episodes_eval
        ), "Default number of eval episodes should instance set size times evaluation episodes."
        assert (
            len(env.envs) == self.dacbench_config_benchmark.num_envs
        ), "Number of environments should match config."
        assert (
            len(eval_env().envs)
            == len(env.envs[0].instance_set.keys())
            * self.dacbench_config_benchmark.n_episodes_eval
        ), "Number of environments should match eval length."

        assert isinstance(
            env, gym.vector.SyncVectorEnv
        ), "DACBench environment should be a SyncVectorEnv."
        assert isinstance(
            eval_env(), gym.vector.SyncVectorEnv
        ), "Eval environment should be a SyncVectorEnv."

        benchmark_kwargs = OmegaConf.to_container(
            self.dacbench_config_benchmark.env_kwargs, resolve=True
        )
        del benchmark_kwargs["benchmark"]
        del benchmark_kwargs["config_space"]
        bench = getattr(benchmarks, self.dacbench_config_benchmark.env)()
        benchmark_env = bench.get_benchmark(**benchmark_kwargs)
        assert isinstance(
            env.envs[0], type(benchmark_env)
        ), "Environment should have correct type."
        assert isinstance(
            eval_env().envs[0], type(benchmark_env)
        ), "Eval environment should have correct type."
        assert eval_env().envs[0].test, "Eval environment should be in test mode."
        for k in env.envs[0].config.keys():
            if k == "observation_space_args":
                continue
            elif k == "instance_set" or k == "test_set":
                for i in range(len(env.envs[0].config[k])):
                    assert (
                        env.envs[0].config[k][i].functions[0].a
                        == benchmark_env.config[k][i].functions[0].a
                    ), f"Environment should have matching instances, mismatch for function parameter a at instance {i}: {env.envs[0].config[k][i].functions[0].a} != {benchmark_env.config[k][i].functions[0].a}"
                    assert (
                        env.envs[0].config[k][i].functions[0].b
                        == benchmark_env.config[k][i].functions[0].b
                    ), f"Environment should have matching instances, mismatch for function parameter b at instance {i}: {env.envs[0].config[k][i].functions[0].b} != {benchmark_env.config[k][i].functions[0].b}"
                    assert (
                        env.envs[0].config[k][i].omit_instance_type
                        == benchmark_env.config[k][i].omit_instance_type
                    ), f"Environment should have matching instances, mismatch for omit_instance_type at instance {i}: {env.envs[0].config[k][i].omit_instance_type} != {benchmark_env.config[k][i].omit_instance_type}"
            else:
                assert (
                    env.envs[0].config[k] == benchmark_env.config[k]
                ), f"Environment should have correct config, mismatch at {k}: {env.envs[0].config[k]} != {benchmark_env.config[k]}"

    def test_make_carl_env(self):
        """Test env creation with make_carl_env."""
        env, eval_env, eval_default = make_carl_env(self.carl_config)
        self.check_vector_env(env)
        self.check_vector_env(eval_env())
        assert eval_default == self.carl_config.n_episodes_eval * len(
            env.envs[0].contexts.keys()
        ), "Default number of eval episodes should match config"

        env_class = getattr(carl.envs, self.carl_config.env)
        assert isinstance(
            env.envs[0], env_class
        ), "Environment should have the correct type."
        assert isinstance(
            eval_env().envs[0], env_class
        ), "Eval environment should have the correct type."

        assert isinstance(
            env, CARLVectorEnvSimulator
        ), "CARL environment should be wrapped."
        assert isinstance(
            eval_env(), CARLVectorEnvSimulator
        ), "CARL eval environment should be wrapped."

    def test_make_carl_context(self):
        """Test env creation with make_carl_env."""
        env, eval_env, eval_default = make_carl_env(self.carl_config_context)
        self.check_vector_env(env)
        self.check_vector_env(eval_env())
        assert eval_default == self.carl_config_context.n_episodes_eval * len(
            env.envs[0].contexts.keys()
        ), "Default number of eval episodes should match config"

        train_contexts = env.envs[0].contexts
        eval_contexts = eval_env().envs[0].contexts
        assert (
            len(train_contexts) == self.carl_config_context.env_kwargs.num_contexts
        ), "Number of training contexts should match config."
        assert (
            len(eval_contexts) == 100
        ), "Number of eval contexts should match default."

        assert not all(
            [
                train_contexts[i]["target_distance"]
                == train_contexts[i + 1]["target_distance"]
                for i in range(len(train_contexts) - 1)
            ]
        ), "Contexts should be varied in target distance."
        assert not all(
            [
                train_contexts[i]["target_direction"]
                == train_contexts[i + 1]["target_direction"]
                for i in range(len(train_contexts) - 1)
            ]
        ), "Contexts should be varied in target direction."
        assert not all(
            [
                train_contexts[i]["friction"] == train_contexts[i + 1]["friction"]
                for i in range(len(train_contexts) - 1)
            ]
        ), "Contexts should be varied in friction."
        assert not all(
            [
                train_contexts[i]["gravity"] == train_contexts[i + 1]["gravity"]
                for i in range(len(train_contexts) - 1)
            ]
        ), "Contexts should be varied in gravity."

        assert not all(
            [
                eval_contexts[i]["target_distance"]
                == eval_contexts[i + 1]["target_distance"]
                for i in range(len(eval_contexts) - 1)
            ]
        ), "Eval contexts should be varied in target distance."
        assert not all(
            [
                eval_contexts[i]["target_direction"]
                == eval_contexts[i + 1]["target_direction"]
                for i in range(len(eval_contexts) - 1)
            ]
        ), "Eval contexts should be varied in target direction."
        assert not all(
            [
                eval_contexts[i]["friction"] == eval_contexts[i + 1]["friction"]
                for i in range(len(eval_contexts) - 1)
            ]
        ), "Eval contexts should be varied in friction."
        assert not all(
            [
                eval_contexts[i]["gravity"] == eval_contexts[i + 1]["gravity"]
                for i in range(len(eval_contexts) - 1)
            ]
        ), "Eval contexts should be varied in gravity."

        assert all(
            [
                train_contexts[i]["target_direction"] in [1, 2, 3, 4]
                for i in range(len(train_contexts))
            ]
        ), "Contexts lie within distribution of target direction."
        assert all(
            [train_contexts[i]["friction"] <= 10 for i in range(len(train_contexts))]
        ), "Contexts lie below upper bound for friction."
        assert all(
            [train_contexts[i]["friction"] >= 0 for i in range(len(train_contexts))]
        ), "Contexts lie above lower bound for friction."
        assert all(
            [train_contexts[i]["gravity"] <= 5 for i in range(len(train_contexts))]
        ), "Contexts lie below upper bound for gravity."
        assert all(
            [train_contexts[i]["gravity"] >= -5 for i in range(len(train_contexts))]
        ), "Contexts lie above lower bound for gravity."

        assert all(
            [
                eval_contexts[i]["target_direction"] in [1, 2, 3, 4]
                for i in range(len(eval_contexts))
            ]
        ), "Eval contexts lie within distribution of target direction."
        assert all(
            [eval_contexts[i]["friction"] <= 10 for i in range(len(eval_contexts))]
        ), "Eval contexts lie below upper bound for friction."
        assert all(
            [eval_contexts[i]["friction"] >= 0 for i in range(len(eval_contexts))]
        ), "Eval contexts lie above lower bound for friction."
        assert all(
            [eval_contexts[i]["gravity"] <= 5 for i in range(len(eval_contexts))]
        ), "Eval contexts lie below upper bound for gravity."
        assert all(
            [eval_contexts[i]["gravity"] >= -5 for i in range(len(eval_contexts))]
        ), "Eval contexts lie above lower bound for gravity."

    def test_make_procgen_env(self):
        """Test env creation with make_procgen_env."""
        if PROCGEN:
            env, eval_env, eval_default = make_procgen_env(self.procgen_config)
            assert (
                eval_default == self.procgen_config.n_episodes_eval
            ), "Default number of eval episodes should match config"
            self.check_vector_env(env)
            self.check_vector_env(eval_env())
            if ENVPOOL:
                assert isinstance(
                    env, envpool.VectorEnv
                ), "Environment should be an envpool env if we create a gym env with envpool installed."
            else:
                assert isinstance(
                    env, ProcgenVecEnv
                ), "Environment should be ProcGen env if we create a gym env without envpool installed."
            assert isinstance(
                eval_env(), ProcgenVecEnv
            ), "Eval env should be a ProcGen env."
        else:
            Warning("Procgen not installed, skipping test.")

    def test_make_pufferlib_env(self):
        """Test env creation with make_pufferlib_env."""
        env, eval_env, eval_default = make_pufferlib_env(self.pufferlib_config)
        self.check_vector_env(env)
        self.check_vector_env(eval_env())
        assert (
            eval_default == self.pufferlib_config.n_episodes_eval
        ), "Default number of eval episodes should match config"
        assert (
            len(env.envs) == self.pufferlib_config.num_envs
        ), "Number of environments should match config."
        assert (
            len(eval_env().envs) == self.pufferlib_config.n_episodes_eval
        ), "Number of environments should match config."

        domain = ".".join(self.pufferlib_config.env.split(".")[:-1])
        name = self.pufferlib_config.env.split(".")[-1]
        get_env_func = importlib.import_module(domain).env_creator
        make_env = get_env_func(name)(**self.pufferlib_config.env_kwargs)
        assert isinstance(
            env.envs[0], type(make_env)
        ), "Environment should have correct type."
        assert isinstance(
            eval_env().envs[0], type(make_env)
        ), "Eval environment should have correct type."

        assert isinstance(
            env, PufferlibToGymAdapter
        ), "Pufferlib env should be wrapped."
        assert isinstance(
            eval_env(), PufferlibToGymAdapter
        ), "Pufferlib eval env should be wrapped."

    def test_make_mighty_env(self):
        """Test correct typing of environments when creating with make_mighty_env."""
        env, eval_env, eval_default = make_mighty_env(self.gym_config)
        assert (
            eval_default == self.gym_config.n_episodes_eval
        ), "Default number of eval episodes should match config"
        self.check_vector_env(env)
        self.check_vector_env(eval_env())
        if ENVPOOL:
            assert isinstance(
                env, envpool.VectorEnv
            ), "Mighty environment should be an envpool env if we create a gym env with envpool installed."
            assert isinstance(
                eval_env(), gym.vector.SyncVectorEnv
            ), "Eval env should be a SyncVectorEnv env if we create a gym env with envpool installed."
        else:
            Warning("Envpool not installed, skipping test.")
            assert isinstance(
                env, gym.vector.SyncVectorEnv
            ), "Mighty environment should be a SyncVectorEnv if we create a gym env without envpool installed."
            assert isinstance(
                env, gym.vector.SyncVectorEnv
            ), "Eval environment should be a SyncVectorEnv if we create a gym env without envpool installed."

        for config in [
            self.dacbench_config,
            self.carl_config,
            self.carl_config_context,
            self.pufferlib_config,
        ]:
            env, eval_env, _ = make_mighty_env(config)
            self.check_vector_env(env)
            self.check_vector_env(eval_env())

        if PROCGEN:
            env, eval_env, _ = make_mighty_env(self.procgen_config)
            self.check_vector_env(env)
            self.check_vector_env(eval_env())
