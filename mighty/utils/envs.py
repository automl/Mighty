"""Environment creation utilities."""
from __future__ import annotations

import importlib
from functools import partial

import gymnasium as gym
from mighty.utils.wrappers import PufferlibToGymAdapter

try:
    import envpool

    ENVPOOL = True
except ImportError:
    ENVPOOL = False


def make_dacbench_env(cfg):
    """Make dacbench environment."""
    from dacbench import benchmarks

    bench = getattr(benchmarks, cfg.env)()

    use_benchmark = False
    if "benchmark" in cfg.env_kwargs:
        use_benchmark = cfg.env_kwargs["benchmark"]

    if use_benchmark:
        del cfg.env_kwargs["benchmark"]
        env = bench.get_benchmark(**cfg.env_kwargs)
        eval_env = bench.get_benchmark(**cfg.env_kwargs)
    else:
        for k in cfg.env_kwargs:
            bench.config[k] = cfg.env_kwargs[k]
        env = bench.get_environment()
        eval_env = bench.get_environment()
    eval_default = len(eval_env.instance_set.keys())
    return env, eval_env, eval_default


def make_carl_env(cfg):
    """Make carl environment."""
    import carl
    from carl.context.sampling import sample_contexts

    if "num_contexts" not in cfg.env_kwargs:
        cfg.env_kwargs["num_contexts"] = 100
    if "context_feature_args" not in cfg.env_kwargs:
        cfg.env_kwargs["context_feature_args"] = []

    contexts = sample_contexts(cfg.env, **cfg.env_kwargs)
    eval_contexts = sample_contexts(cfg.env, **cfg.env_kwargs)

    env_class = getattr(carl.envs, cfg.env)
    env = env_class(contexts=contexts)
    eval_env = env_class(contexts=eval_contexts)
    eval_default = len(eval_contexts)
    return env, eval_env, eval_default


def make_procgen_env(cfg):
    """Make procgen environment."""
    from procgen import ProcgenGym3Env

    if ENVPOOL:
        env = envpool.make(ProcgenGym3Env, env_type="gym", **cfg.env_kwargs)
    else:
        env = ProcgenGym3Env(num=cfg.num_envs, env_name=cfg.env.split(":")[-1])
    # TODO: needs a partial here
    eval_env = ProcgenGym3Env(num=cfg.n_episodes_eval, env_name=cfg.env.split(":")[-1])
    eval_default = cfg.n_episodes_eval
    return env, eval_env, eval_default


def make_pufferlib_env(cfg):
    """Make pufferlib environment."""
    import pufferlib
    import pufferlib.vector

    domain = ".".join(cfg.env.split(".")[:-1])
    name = cfg.env.split(".")[-1]
    get_env_func = importlib.import_module(domain).env_creator
    make_env = partial(get_env_func(name), **cfg.env_kwargs)
    if cfg.debug:
        env = make_env()
    else:
        env = PufferlibToGymAdapter(pufferlib.vector.make(make_env, num_envs=cfg.num_envs))

    def get_eval():
        env = pufferlib.vector.make(make_env, num_envs=cfg.n_episodes_eval)
        return PufferlibToGymAdapter(env)

    eval_default = cfg.n_episodes_eval
    return env, get_eval, eval_default


def make_gym_env(cfg):
    """Make gymnasium environment."""
    make_env = partial(gym.make, cfg.env, **cfg.env_kwargs)
    env = gym.vector.SyncVectorEnv([make_env for _ in range(cfg.num_envs)])
    eval_env = partial(
        gym.vector.SyncVectorEnv, [make_env for _ in range(cfg.n_episodes_eval)]
    )
    eval_default = cfg.n_episodes_eval
    return env, eval_env, eval_default


def make_mighty_env(cfg):
    """Return environment according to the configuration."""
    if cfg.env.endswith("Benchmark"):
        env, eval_env, eval_default = make_dacbench_env(cfg)
    elif cfg.env.startswith("CARL"):
        env, eval_env, eval_default = make_carl_env(cfg)
    elif cfg.env.startswith("procgen"):
        env, eval_env, eval_default = make_procgen_env(cfg)
    elif cfg.env.startswith("pufferlib"):
        env, eval_env, eval_default = make_pufferlib_env(cfg)
    elif ENVPOOL:
        env = envpool.make(cfg.env, env_type="gym", **cfg.env_kwargs)
        eval_env = gym.make(cfg.env, **cfg.env_kwargs)
        eval_default = 1
    else:
        env, eval_env, eval_default = make_gym_env(cfg)
    return env, eval_env, eval_default
