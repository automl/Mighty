import os
from pathlib import Path
from rich import print

from dacbench import benchmarks
from dacbench.wrappers import PerformanceTrackingWrapper

from mighty.agent.factory import get_agent_class
from mighty.utils.logger import Logger

import importlib
import mighty.utils.main_parser
importlib.reload(mighty.utils.main_parser)

from omegaconf import DictConfig
import hydra


@hydra.main("./configs", "base")
def main(cfg: DictConfig):
    out_dir = os.getcwd()  # working directory changes to hydra.run.dir
    seed = cfg.seed

    logger = Logger(
        experiment_name=f"{cfg.experiment_name}_{seed}",
        step_write_frequency=10,
        episode_write_frequency=None,
        log_to_wandb=cfg.wandb_project,
        log_to_tensorboad=cfg.tensorboard_file,
        hydra_config=cfg
    )

    if cfg.env in dir(benchmarks):
        bench = getattr(benchmarks, cfg.env)()
        env = bench.get_environment()
        eval_env = bench.get_environment()
        eval_default = len(eval_env.instance_set.keys())
    elif cfg.env.startswith("CARL"):
        from carl.context.sampling import sample_contexts

        if "num_contexts" not in cfg.env_kwargs.keys():
            cfg.env_kwargs["num_contexts"] = 100
        if "context_feature_args" not in cfg.env_kwargs.keys():
            cfg.env_kwargs["context_feature_args"] = []

        contexts = sample_contexts(cfg.env, **cfg.env_kwargs)
        eval_contexts = sample_contexts(cfg.env, **cfg.env_kwargs)

        env_class = getattr(carl.envs, cfg.env)
        env = env_class(contexts)
        eval_env = env_class(eval_contexts)
        eval_default = len(eval_contexts)
    else:
        import gym
        env = gym.make(cfg.env)
        eval_env = gym.make(cfg.env)
        eval_default = 1

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

    if not cfg.checkpoint is None:
        agent.load(cfg.checkpoint)
        print('#' * 80)
        print(f"Loading checkpoint at {cfg.checkpoint}")

    print('#' * 80)
    print(f'Using agent type "{agent}" to learn')
    print('#' * 80)
    num_eval_episodes = 100
    agent.train(
        n_steps=cfg.num_steps,
        n_episodes_eval=n_episodes_eval,
        eval_every_n_steps=eval_every_n_steps,
    )
    logger.close()

if __name__ == "__main__":
    main()
