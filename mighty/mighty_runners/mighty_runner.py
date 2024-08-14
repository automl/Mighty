from __future__ import annotations
from abc import ABC

import logging
import warnings
from typing import TYPE_CHECKING, Tuple, Dict, Any

from hydra.utils import get_class
from mighty.mighty_agents.factory import get_agent_class
from mighty.mighty_utils.envs import make_mighty_env
from mighty.mighty_utils.logger import Logger

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MightyRunner(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        """Parse config and run Mighty agent."""
        seed = cfg.seed

        # Initialize Logger
        self.logger = Logger(
            experiment_name=f"{cfg.experiment_name}_{seed}",
            output_path=cfg.output_dir,
            step_write_frequency=100,
            episode_write_frequency=None,
            hydra_config=cfg,
            cli_log_lvl=logging.INFO,
        )
        self.logger.info(f"Output will be written to {self.logger.log_dir}")

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

        def wrap_eval():  # type: ignore
            wrapped_env = base_eval_env()
            for cls, wkwargs in wrapper_classes:
                wrapped_env = cls(wrapped_env, **wkwargs)
            return wrapped_env

        eval_env = wrap_eval()

        # Setup agent
        agent_class = get_agent_class(cfg.algorithm)
        args_agent = dict(cfg.algorithm_kwargs)
        self.agent = agent_class(  # type: ignore
            env=env,
            eval_env=eval_env,
            logger=self.logger,
            seed=cfg.seed,
            **args_agent,
        )

        self.eval_every_n_steps = cfg.eval_every_n_steps
        self.num_steps = cfg.num_steps

        # Load checkpoint if one is given
        if cfg.checkpoint is not None:
            self.agent.load(cfg.checkpoint)
            self.logger.info("#" * 80)
            self.logger.info(f"Loading checkpoint at {cfg.checkpoint}")

        # Train
        self.logger.info("#" * 80)
        self.logger.info(f'Using agent type "{self.agent}" to learn')
        self.logger.info("#" * 80)

    def train(self, num_steps: int, env=None) -> Any:  # type: ignore
        return self.agent.run(
            n_steps=num_steps, env=env, eval_every_n_steps=self.eval_every_n_steps
        )

    def evaluate(self, eval_env=None) -> Any:  # type: ignore
        return self.agent.evaluate(eval_env)

    def close(self) -> None:
        self.logger.close()

    def run(self) -> Tuple[Dict, Dict]:
        raise NotImplementedError
