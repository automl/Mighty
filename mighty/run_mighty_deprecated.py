import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
import logging

import hydra
from hydra.utils import get_class
from omegaconf import DictConfig
from rich import print
import numpy as np

from mighty.agent.factory_deprecated import get_agent_class
from mighty.utils.logger_deprecated import Logger


@hydra.main("./configs", "base", version_base=None)
def main(cfg: DictConfig) -> float:
    """Parse config and run Mighty agent"""
    seed = cfg.seed

    # Initialize Logger
    logger = Logger(
        experiment_name=f"{cfg.experiment_name}_{seed}",
        output_path=cfg.output_dir,
        step_write_frequency=100,
        episode_write_frequency=None,
        log_to_wandb=cfg.wandb_project,
        log_to_tensorboad=cfg.tensorboard_file,
        hydra_config=cfg,
        cli_log_lvl=logging.INFO,
    )
    logger.info(f"Output will be written to {logger.log_dir}")

    # Check whether env is from DACBench, CARL or gym
    # Make train and eval env
    if cfg.env.endswith("Benchmark"):
        from dacbench import benchmarks

        bench = getattr(benchmarks, cfg.env)()

        use_benchmark = False
        if "benchmark" in cfg.env_kwargs.keys():
            use_benchmark = cfg.env_kwargs["benchmark"]

        if use_benchmark:
            del cfg.env_kwargs["benchmark"]
            env = bench.get_benchmark(**cfg.env_kwargs)
            eval_env = bench.get_benchmark(**cfg.env_kwargs)
        else:
            for k in cfg.env_kwargs.keys():
                bench.config[k] = cfg.env_kwargs[k]
            env = bench.get_environment()
            eval_env = bench.get_environment()
        eval_default = len(eval_env.instance_set.keys())
    elif cfg.env.startswith("CARL"):
        import carl
        from carl.context.sampling import sample_contexts

        if "num_contexts" not in cfg.env_kwargs.keys():
            cfg.env_kwargs["num_contexts"] = 100
        if "context_feature_args" not in cfg.env_kwargs.keys():
            cfg.env_kwargs["context_feature_args"] = []

        contexts = sample_contexts(cfg.env, **cfg.env_kwargs)
        eval_contexts = sample_contexts(cfg.env, **cfg.env_kwargs)

        env_class = getattr(carl.envs, cfg.env)
        env = env_class(contexts=contexts)
        eval_env = env_class(contexts=eval_contexts)
        eval_default = len(eval_contexts)
    else:
        import gymnasium as gym

        env = gym.make(cfg.env, **cfg.env_kwargs)
        eval_env = gym.make(cfg.env, **cfg.env_kwargs)
        eval_default = 1

    for w in cfg.env_wrappers:
        if "wrapper_kwargs" in cfg.keys():
            wkwargs = cfg.wrapper_kwargs
        else:
            wkwargs = {}
        cls = get_class(w)
        env = cls(env, **wkwargs)
        eval_env = cls(eval_env, **wkwargs)

    # Setup agent
    agent_class = get_agent_class(cfg.algorithm)
    args_agent = dict(cfg.algorithm_kwargs)
    agent = agent_class(
        env=env,
        eval_env=eval_env,
        logger=logger,
        **args_agent,
    )

    n_episodes_eval = cfg.n_episodes_eval if cfg.n_episodes_eval else eval_default
    eval_every_n_steps = cfg.eval_every_n_steps

    # Load checkpoint if one is given
    if cfg.checkpoint is not None:
        agent.load(cfg.checkpoint)
        print("#" * 80)
        print(f"Loading checkpoint at {cfg.checkpoint}")

    # Train
    print("#" * 80)
    print(f'Using agent type "{agent}" to learn')
    print("#" * 80)
    agent.train(
        n_steps=cfg.num_steps,
        n_episodes_eval=n_episodes_eval,
        eval_every_n_steps=eval_every_n_steps,
        save_model_every_n_steps=cfg.save_model_every_n_steps
    )

    # Final evaluation
    eval_metrics_list = agent.evaluate(n_episodes_eval=n_episodes_eval)
    logger.close()

    # When optimizing we minimize

    # Get performance mean across instances (if any)
    performance = np.mean([d["mean_eval_reward"] for d in eval_metrics_list])
    return -performance


if __name__ == "__main__":
    main()