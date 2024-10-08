"""Environment creation utilities."""

from __future__ import annotations

import importlib
from functools import partial
from typing import TYPE_CHECKING, Tuple, Callable, Any

import gymnasium as gym
from omegaconf import OmegaConf
from mighty.mighty_utils.wrappers import (
    PufferlibToGymAdapter,
    ContextualVecEnv,
    CARLVectorEnvSimulator,
    ProcgenVecEnv,
)

try:
    import envpool  # type: ignore

    ENVPOOL = True
except ImportError:
    ENVPOOL = False

if TYPE_CHECKING:
    from omegaconf import DictConfig


def make_dacbench_env(cfg: DictConfig) -> Tuple[ContextualVecEnv, Callable, int]:
    """Make dacbench environment."""
    from dacbench import benchmarks  # type: ignore
    import ConfigSpace as CS

    bench = getattr(benchmarks, cfg.env)()

    use_benchmark = False
    if "benchmark" in cfg.env_kwargs:
        use_benchmark = cfg.env_kwargs["benchmark"]

    if use_benchmark:
        benchmark_kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True)
        del benchmark_kwargs["benchmark"]  # type: ignore
        if "config_space" in benchmark_kwargs:  # type: ignore
            del benchmark_kwargs["config_space"]  # type: ignore
        make_env = partial(bench.get_benchmark, **benchmark_kwargs)  # type: ignore
    else:
        for k in cfg.env_kwargs:
            if k == "config_space":
                space = CS.ConfigurationSpace()
                for name, desc in cfg.env_kwargs[k].items():
                    if desc["type"] == "int":
                        hp = CS.UniformIntegerHyperparameter(
                            name, lower=desc["lower"], upper=desc["upper"]
                        )
                    elif desc["type"] == "float":
                        hp = CS.UniformFloatHyperparameter(
                            name, lower=desc["lower"], upper=desc["upper"]
                        )
                    elif desc["type"] == "cat":
                        hp = CS.CategoricalHyperparameter(name, desc["choices"])
                    space.add_hyperparameter(hp)
                print(bench.config)
                print(space)
                bench.config.config_space = space
            else:
                bench.config[k] = cfg.env_kwargs[k]
        print(bench.config)
        print(bench.config.keys())
        make_env = bench.get_environment

    def make_eval_env(make_env: Callable) -> Any:
        eval_env = make_env()
        eval_env.use_test_set()
        return eval_env

    env = ContextualVecEnv([make_env for _ in range(cfg.num_envs)])
    eval_env = partial(
        ContextualVecEnv,
        [
            partial(make_eval_env, make_env)
            for _ in range(cfg.n_episodes_eval * len(env.envs[0].instance_set.keys()))  # type: ignore
        ],
    )
    eval_default = len(env.envs[0].instance_set.keys()) * cfg.n_episodes_eval  # type: ignore
    return env, eval_env, eval_default


def make_carl_env(
    cfg: DictConfig,
) -> Tuple[type[CARLVectorEnvSimulator], Callable, int]:
    """Make carl environment."""
    import carl
    from carl import envs
    from carl.context.sampler import ContextSampler

    env_kwargs = OmegaConf.to_container(cfg.env_kwargs, resolve=True)

    if "num_contexts" not in env_kwargs:  # type: ignore
        env_kwargs["num_contexts"] = 100  # type: ignore
    if "num_evaluation_contexts" not in env_kwargs:  # type: ignore
        env_kwargs["num_evaluation_contexts"] = 100  # type: ignore
    if "context_feature_args" not in env_kwargs:  # type: ignore
        env_kwargs["context_feature_args"] = {}  # type: ignore
    if "context_sample_seed" not in env_kwargs:  # type: ignore
        env_kwargs["context_sample_seed"] = 0  # type: ignore
    if "evaluation_context_sample_seed" not in env_kwargs:  # type: ignore
        env_kwargs["evaluation_context_sample_seed"] = 1  # type: ignore

    env_class = getattr(envs, cfg.env)

    if len(env_kwargs["context_feature_args"].keys()) > 0:  # type: ignore
        context_distributions = []
        for context_feature, dist_args in env_kwargs["context_feature_args"].items():  # type: ignore
            if dist_args[0] == "uniform-int":
                dist = carl.context.context_space.UniformIntegerContextFeature(
                    context_feature, lower=dist_args[1], upper=dist_args[2]
                )
            elif dist_args[0] == "uniform-float":
                dist = carl.context.context_space.UniformFloatContextFeature(
                    context_feature, lower=dist_args[1], upper=dist_args[2]
                )
            elif dist_args[0] == "normal":
                dist = carl.context.context_space.NormalFloatContextFeature(
                    context_feature,
                    mu=dist_args[3],
                    sigma=dist_args[4],
                    lower=dist_args[1],
                    upper=dist_args[2],
                )
            elif dist_args[0] == "categorical":
                dist = carl.context.context_space.CategoricalContextFeature(
                    context_feature, choices=dist_args[1]
                )
            else:
                raise ValueError(
                    "Unknown context distribution type. Valid types are: uniform-int, uniform-float, normal, categorical."
                )
            context_distributions.append(dist)

        context_sampler = ContextSampler(
            context_distributions,
            context_space=env_class.get_context_space(),
            seed=env_kwargs["context_sample_seed"],  # type: ignore
        )
        contexts = context_sampler.sample_contexts(env_kwargs["num_contexts"])  # type: ignore
        context_sampler.seed(env_kwargs["evaluation_context_sample_seed"])  # type: ignore
        eval_contexts = context_sampler.sample_contexts(
            env_kwargs["num_evaluation_contexts"]  # type: ignore
        )
    else:
        contexts = {0: env_class.get_default_context()}
        eval_contexts = {0: env_class.get_default_context()}

    env = env_class(contexts=contexts)
    eval_env = env_class(contexts=eval_contexts)

    env = CARLVectorEnvSimulator(env)
    eval_env = partial(CARLVectorEnvSimulator, eval_env)
    eval_default = len(eval_contexts) * cfg.n_episodes_eval
    return env, eval_env, eval_default


def make_procgen_env(cfg: DictConfig) -> Tuple[type[ProcgenVecEnv], Callable, int]:
    """Make procgen environment."""
    from procgen import ProcgenEnv  # type: ignore

    if ENVPOOL:
        env = envpool.make(ProcgenEnv, env_type="gym", **cfg.env_kwargs)
    else:
        env = ProcgenVecEnv(
            ProcgenEnv(num_envs=cfg.num_envs, env_name=cfg.env.split(":")[-1])
        )
    eval_base = ProcgenEnv(
        num_envs=cfg.n_episodes_eval, env_name=cfg.env.split(":")[-1]
    )
    eval_env = partial(ProcgenVecEnv, eval_base)
    eval_default = cfg.n_episodes_eval
    return env, eval_env, eval_default


def make_pufferlib_env(cfg: DictConfig) -> Tuple[PufferlibToGymAdapter, Callable, int]:
    """Make pufferlib environment."""
    import pufferlib  # type: ignore
    import pufferlib.vector  # type: ignore

    domain = ".".join(cfg.env.split(".")[:-1])
    name = cfg.env.split(".")[-1]
    get_env_func = importlib.import_module(domain).env_creator
    make_env = partial(get_env_func(name), **cfg.env_kwargs)
    env = PufferlibToGymAdapter(pufferlib.vector.make(make_env, num_envs=cfg.num_envs))

    def get_eval() -> PufferlibToGymAdapter:
        env = pufferlib.vector.make(make_env, num_envs=cfg.n_episodes_eval)
        return PufferlibToGymAdapter(env)

    eval_default = cfg.n_episodes_eval
    return env, get_eval, eval_default


def make_gym_env(
    cfg: DictConfig,
) -> Tuple[gym.vector.SyncVectorEnv, partial[gym.vector.SyncVectorEnv], int]:
    """Make gymnasium environment."""
    make_env = partial(gym.make, cfg.env, **cfg.env_kwargs)
    env = gym.vector.SyncVectorEnv([make_env for _ in range(cfg.num_envs)])
    eval_env = partial(
        gym.vector.SyncVectorEnv, [make_env for _ in range(cfg.n_episodes_eval)]
    )
    eval_default = cfg.n_episodes_eval
    return env, eval_env, eval_default


def make_mighty_env(cfg: DictConfig) -> Tuple[ContextualVecEnv, Callable, int]:
    """Return environment according to the configuration."""
    if cfg.env.endswith("Benchmark"):
        env, eval_env, eval_default = make_dacbench_env(cfg)
    elif cfg.env.startswith("CARL"):
        env, eval_env, eval_default = make_carl_env(cfg)  # type: ignore
    elif cfg.env.startswith("procgen"):
        env, eval_env, eval_default = make_procgen_env(cfg)  # type: ignore
    elif cfg.env.startswith("pufferlib"):
        env, eval_env, eval_default = make_pufferlib_env(cfg)  # type: ignore
    elif ENVPOOL:
        env = envpool.make(cfg.env, env_type="gym", **cfg.env_kwargs)
        make_env = partial(gym.make, cfg.env, **cfg.env_kwargs)
        eval_env = partial(
            gym.vector.SyncVectorEnv, [make_env for _ in range(cfg.n_episodes_eval)]
        )
        eval_default = cfg.n_episodes_eval
    else:
        env, eval_env, eval_default = make_gym_env(cfg)  # type: ignore
    return env, eval_env, eval_default
