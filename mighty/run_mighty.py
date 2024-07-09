"""Run Mighty agent."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import hydra
from hydra.utils import get_class
from mighty.mighty_agents.factory import get_agent_class
from mighty.utils.envs import make_mighty_env
from mighty.utils.logger import Logger
from rich import print


warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main("./configs", "base", version_base=None)
def main(cfg: DictConfig) -> float:
    """Parse config and run Mighty agent."""
    seed = cfg.seed

    # Initialize Logger
    logger = Logger(
        experiment_name=f"{cfg.experiment_name}_{seed}",
        output_path=cfg.output_dir,
        step_write_frequency=100,
        episode_write_frequency=None,
        hydra_config=cfg,
        cli_log_lvl=logging.INFO,
    )
    logger.info(f"Output will be written to {logger.log_dir}")

    # Check whether env is from DACBench, CARL or gym
    # Make train and eval env
    env, base_eval_env, eval_default = make_mighty_env(cfg)

    # TODO: move wrapping to env handling?
    wrapper_classes = []
    for w in cfg.env_wrappers:
        wkwargs = cfg.wrapper_kwargs if "wrapper_kwargs" in cfg else {}
        cls = get_class(w)
        env = cls(env, **wkwargs)
        wrapper_classes.append((cls, wkwargs))

    def wrap_eval():
        wrapped_env = base_eval_env()
        for cls, wkwargs in wrapper_classes:
            wrapped_env = cls(wrapped_env, **wkwargs)
        return wrapped_env

    eval_env = wrap_eval

    # Setup agent
    agent_class = get_agent_class(cfg.algorithm)
    args_agent = dict(cfg.algorithm_kwargs)

    # Update args_agent with obs_shape and action_dim
    if cfg.algorithm == "PPO":
        args_agent["rollout_buffer_kwargs"]["obs_shape"] = (
            env.single_observation_space.shape
        )
        args_agent["rollout_buffer_kwargs"]["act_dim"] = int(env.single_action_space.n)
        args_agent["rollout_buffer_kwargs"]["n_envs"] = cfg.num_envs

    agent = agent_class(
        env=env,
        eval_env=eval_env,
        logger=logger,
        seed=cfg.seed,
        **args_agent,
    )

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

    agent.run(
        n_steps=cfg.num_steps,
        n_episodes_eval=cfg.n_episodes_eval,
        eval_every_n_steps=eval_every_n_steps,
        save_model_every_n_steps=cfg.save_model_every_n_steps,
    )

    # Final evaluation
    eval_metrics_list = agent.evaluate()
    logger.close()

    # When optimizing we minimize
    # Get performance mean across instances (if any)
    performance = eval_metrics_list["mean_eval_reward"]
    return -performance


if __name__ == "__main__":
    main()
