import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
import logging
from rich import print
import importlib

from mighty.agent.factory import get_agent_class
from mighty.utils.logger import Logger

from omegaconf import DictConfig
import hydra


@hydra.main("./configs", "base", version_base=None)
def main(cfg: DictConfig):
    """Parse config and run Mighty agent"""
    seed = cfg.seed

    #Initialize Logger
    logger = Logger(
        experiment_name=f"{cfg.experiment_name}_{seed}",
        output_path=cfg.output_dir,
        step_write_frequency=100,
        episode_write_frequency=None,
        log_to_wandb=cfg.wandb_project,
        log_to_tensorboad=cfg.tensorboard_file,
        hydra_config=cfg,
        cli_log_lvl=logging.INFO
    )
    logger.info(f'Output will be written to {logger.log_dir}')

    # Check whether env is from DACBench, CARL or gym
    # Make train and eval env
    if cfg.env.endswith("Benchmark"):
        from dacbench import benchmarks
        bench = getattr(benchmarks, cfg.env)()
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

        env = gym.make(cfg.env)
        eval_env = gym.make(cfg.env)
        eval_default = 1

    for w in cfg.env_wrappers:
        class_name = w.split('.')[-1]
        import_from = importlib.import_module('.'.join(w.split('.')[:-1]))
        env = getattr(import_from, class_name)(env)
        eval_env = getattr(import_from, class_name)(eval_env)

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
    )
    logger.close()


if __name__ == "__main__":
    main()
