"""Base agent template."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict
from abc import ABC
import numpy as np
import torch
import wandb

from mighty.mighty_replay import MightyReplay, MightyRolloutBuffer
from mighty.mighty_utils.env_handling import CARLENV, DACENV, MIGHTYENV
from mighty.mighty_exploration import MightyExplorationPolicy
from mighty.mighty_utils.agent_handling import retrieve_class
from omegaconf import DictConfig
from rich import print
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

if TYPE_CHECKING:
    from mighty.mighty_utils.logger import Logger
    from mighty.mighty_utils.types import TypeKwargs


class MightyAgent(ABC):
    """Base agent for RL implementations."""

    def __init__(  # noqa: PLR0915, PLR0912
        self,
        env: MIGHTYENV,  # type: ignore
        logger: Logger,
        seed: int | None = None,
        eval_env: MIGHTYENV | None = None,  # type: ignore
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        batch_size: int = 64,
        learning_starts: int = 1,
        render_progress: bool = True,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        wandb_kwargs: dict | None = None,
        replay_buffer_class: str
        | DictConfig
        | type[MightyReplay]
        | type[MightyRolloutBuffer]
        | None = None,
        replay_buffer_kwargs: TypeKwargs | None = None,
        meta_methods: list[str | type] | None = None,
        meta_kwargs: list[TypeKwargs] | None = None,
        verbose: bool = True,
    ):
        """Base agent initialization.

        Creates all relevant class variables and calls agent-specific init function

        :param env: Train environment
        :param logger: Mighty logger
        :param eval_env: Evaluation environment
        :param learning_rate: Learning rate for training
        :param epsilon: Exploration factor for training
        :param batch_size: Batch size for training
        :param render_progress: Render progress
        :param log_tensorboard: Log to tensorboard as well as to file
        :param log_wandb: Whether to log to wandb
        :param wandb_kwargs: Kwargs for wandb.init, e.g. including the project name
        :param replay_buffer_class: Replay buffer class from coax replay buffers
        :param replay_buffer_kwargs: Arguments for the replay buffer
        :param tracer_class: Reward tracing class from coax tracers
        :param tracer_kwargs: Arguments for the reward tracer
        :param meta_methods: Class names or types of mighty meta learning modules to use
        :param meta_kwargs: List of kwargs for the meta learning modules
        :return:
        """
        if meta_kwargs is None:
            meta_kwargs = []
        if meta_methods is None:
            meta_methods = []
        if wandb_kwargs is None:
            wandb_kwargs = {}
        self.learning_rate = learning_rate
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._learning_starts = learning_starts

        self.buffer: MightyReplay | None = None
        self.policy: MightyExplorationPolicy | None = None

        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            torch.manual_seed(seed)
        else:
            self.rng = np.random.default_rng()

        # Replay Buffer
        replay_buffer_class = retrieve_class(
            cls=replay_buffer_class,
            default_cls=MightyReplay,  # type: ignore
        )
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {  # type: ignore
                "capacity": 1_000_000,
            }
        self.buffer_class = replay_buffer_class
        self.buffer_kwargs = replay_buffer_kwargs

        output_dir = logger.log_dir if logger is not None else None
        self.verbose = verbose

        self.env = env
        if eval_env is None:
            self.eval_env = self.env
        else:
            self.eval_env = eval_env

        self.logger = logger
        self.render_progress = render_progress
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.model_dir = Path(self.output_dir) / Path("models")

        # Create meta modules
        self.meta_modules = {}
        for i, m in enumerate(meta_methods):
            meta_class = retrieve_class(cls=m, default_cls=None)  # type: ignore
            assert (
                meta_class is not None
            ), f"Class {m} not found, did you specify the correct loading path?"
            kwargs: Dict = {}
            if len(meta_kwargs) > i:
                kwargs = meta_kwargs[i]
            self.meta_modules[meta_class.__name__] = meta_class(**kwargs)

        self.logger.log("Meta modules", list(self.meta_modules.keys()))

        self.last_state = None
        self.total_steps = 0

        self.writer = None
        if log_tensorboard and output_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(output_dir)
            self.writer.add_scalar("hyperparameter/learning_rate", self.learning_rate)
            self.writer.add_scalar("hyperparameter/batch_size", self._batch_size)
            self.writer.add_scalar("hyperparameter/policy_epsilon", self._epsilon)

        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(**wandb_kwargs)

        self.initialize_agent()
        self.steps = 0

    def _initialize_agent(self) -> None:
        """Agent/algorithm specific initializations."""
        raise NotImplementedError

    def process_transition(  # type: ignore
        self, curr_s, action, reward, next_s, dones, log_prob=None, metrics=None
    ) -> Dict:
        """Agent/algorithm specific transition operations."""
        raise NotImplementedError

    def initialize_agent(self) -> None:
        """General initialization of tracer and buffer for all agents.

        Algorithm specific initialization like policies etc.
        are done in _initialize_agent
        """
        self._initialize_agent()
        self.buffer = self.buffer_class(**self.buffer_kwargs)  # type: ignore

    def update_agent(self) -> Dict:
        """Policy/value function update."""
        raise NotImplementedError

    def adapt_hps(self, metrics: Dict) -> None:
        """Set hyperparameters."""
        self.learning_rate = metrics["hp/lr"]
        self._epsilon = metrics["hp/pi_epsilon"]
        self._batch_size = metrics["hp/batch_size"]
        self._learning_starts = metrics["hp/learning_starts"]

    def make_checkpoint_dir(self, t: int) -> None:
        """Checkpoint model.

        :param T: Current timestep
        :return:
        """
        logdir = self.logger.log_dir
        self.upper_checkpoint_dir = Path(logdir) / Path("checkpoints")
        if not self.upper_checkpoint_dir.exists():
            Path(self.upper_checkpoint_dir).mkdir()
        self.checkpoint_dir = self.upper_checkpoint_dir / f"{t}"
        if not self.checkpoint_dir.exists():
            Path(self.checkpoint_dir).mkdir()

    def __del__(self) -> None:
        """Close wandb upon deletion."""
        self.env.close()  # type: ignore
        if self.log_wandb:
            wandb.finish()

    def step(self, observation: torch.Tensor, metrics: Dict) -> torch.Tensor:
        for k in self.meta_modules.keys():
            self.meta_modules[k].pre_step(metrics)

        self.adapt_hps(metrics)
        return self.policy(observation, metrics=metrics, return_logp=True)  # type: ignore

    def update(self, metrics: Dict, update_kwargs: Dict) -> Dict:
        """Update agent."""
        for k in self.meta_modules:
            self.meta_modules[k].pre_update(metrics)

        agent_update_metrics = self.update_agent(**update_kwargs)
        metrics.update(agent_update_metrics)
        metrics = {k: np.array(v) for k, v in metrics.items()}
        metrics["step"] = self.steps

        if self.writer is not None:
            self.writer.add_scalars("training_metrics", metrics, global_step=self.steps)

        if self.log_wandb:
            wandb.log(metrics)

        metrics["env"] = self.env
        metrics["vf"] = self.value_function  # type: ignore
        metrics["policy"] = self.policy
        for k in self.meta_modules:
            self.meta_modules[k].post_update(metrics)
        return metrics

    def run(  # noqa: PLR0915
        self,
        n_steps: int,
        eval_every_n_steps: int = 1_000,
        human_log_every_n_steps: int = 5000,
        save_model_every_n_steps: int | None = 5000,
        env: MIGHTYENV = None,  # type: ignore
    ) -> Dict:
        """Run agent."""
        episodes = 0
        if env is not None:
            self.env = env
        # FIXME: can we add the eval result here? Else the evals spam the command line in a pretty ugly way
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "Remaining:",
            TimeRemainingColumn(),
            "Elapsed:",
            TimeElapsedColumn(),
            disable=not self.render_progress,
        ) as progress:
            steps_task = progress.add_task(
                "Train Steps",
                total=n_steps - self.steps,
                start=False,
                visible=False,
            )
            steps_since_eval = 0
            progress.start_task(steps_task)
            # FIXME: this is more of a question: are there cases where we don't want to reset this completely?
            # I can't think of any, can you? If yes, we should maybe add this as an optional argument
            metrics = {
                "env": self.env,
                "vf": self.value_function,  # type: ignore
                "policy": self.policy,
                "step": self.steps,
                "hp/lr": self.learning_rate,
                "hp/pi_epsilon": self._epsilon,
                "hp/batch_size": self._batch_size,
                "hp/learning_starts": self._learning_starts,
            }

            # Reset env and initialize reward sum
            curr_s, _ = self.env.reset()  # type: ignore
            if len(curr_s.squeeze().shape) == 0:
                episode_reward = [0]
            else:
                episode_reward = np.zeros(curr_s.squeeze().shape[0])  # type: ignore

            last_episode_reward = episode_reward
            progress.update(steps_task, visible=True)

            # Main loop: rollouts, training and evaluation
            while self.steps < n_steps:
                metrics["episode_reward"] = episode_reward

                # TODO Remove
                progress.stop()

                action, log_prob = self.step(curr_s, metrics)

                next_s, reward, terminated, truncated, _ = self.env.step(action)  # type: ignore
                dones = np.logical_or(terminated, truncated)

                transition_metrics = self.process_transition(
                    curr_s, action, reward, next_s, dones, log_prob, metrics
                )

                metrics.update(transition_metrics)

                episode_reward += reward

                # Log everything
                t = {
                    "seed": self.seed,
                    "step": self.steps,
                    "reward": reward.tolist(),
                    "action": action.tolist(),
                    "state": curr_s.tolist(),
                    "next_state": next_s.tolist(),
                    "terminated": terminated.tolist(),
                    "truncated": truncated.tolist(),
                    "episode_reward": last_episode_reward,
                }
                metrics["episode_reward"] = episode_reward
                self.logger.log_dict(t)
                if self.writer is not None:
                    self.writer.add_scalars("transition", t, global_step=self.steps)

                if self.log_wandb:
                    wandb.log(t)

                for k in self.meta_modules:
                    self.meta_modules[k].post_step(metrics)

                self.steps += len(action)
                metrics["step"] = self.steps
                steps_since_eval += len(action)
                for _ in range(len(action)):
                    progress.advance(steps_task)

                # Update agent
                if (
                    len(self.buffer) >= self._batch_size  # type: ignore
                    and self.steps >= self._learning_starts
                ):
                    update_kwargs = {"next_s": next_s, "dones": dones}

                    metrics = self.update(metrics, update_kwargs)

                # End step
                self.last_state = curr_s
                curr_s = next_s
                self.logger.next_step()

                # Evaluate
                if eval_every_n_steps and steps_since_eval >= eval_every_n_steps:
                    steps_since_eval = 0
                    self.evaluate()

                # Log to command line
                if self.steps % human_log_every_n_steps == 0 and self.verbose:
                    mean_last_ep_reward = np.round(
                        np.mean(last_episode_reward), decimals=2
                    )
                    mean_last_step_reward = np.round(
                        np.mean(mean_last_ep_reward / len(last_episode_reward)),
                        decimals=2,
                    )
                    print(
                        f"""Steps: {self.steps}, Latest Episode Reward: {mean_last_ep_reward}, Latest Step Reward: {mean_last_step_reward}"""  # noqa: E501
                    )

                # Save
                if (
                    save_model_every_n_steps
                    and self.steps % save_model_every_n_steps == 0
                ):
                    self.save(self.steps)

                if np.any(dones):
                    last_episode_reward = np.where(  # type: ignore
                        dones, episode_reward, last_episode_reward
                    )
                    episode_reward = np.where(dones, 0, episode_reward)  # type: ignore
                    # End episode
                    if isinstance(self.env, DACENV) or isinstance(self.env, CARLENV):
                        instance = self.env.instance  # type: ignore
                    else:
                        instance = None
                    self.logger.next_episode(instance)
                    episodes += 1
                    for k in self.meta_modules:
                        self.meta_modules[k].post_episode(metrics)

                    # Remove rollout data from last episode
                    # TODO: only do this for finished envs
                    # FIXME: open todo, I think we need to use dones as a mask here
                    # Proposed fix: metrics[k][:, dones] = 0
                    # I don't think this is correct masking and I think we have to check the size of zeros
                    for k in list(metrics.keys()):
                        if "rollout" in k:
                            del metrics[k]

                    # Meta Module hooks
                    for k in self.meta_modules:
                        self.meta_modules[k].pre_episode(metrics)
        return metrics

    def apply_config(self, config: Dict) -> None:
        """Apply config to agent."""
        for n in config:
            algo_name = n.split(".")[-1]
            if hasattr(self, algo_name):
                setattr(self, algo_name, config[n])
            elif hasattr(self, "_" + algo_name):
                setattr(self, "_" + algo_name, config[n])
            elif n in ["architecture", "n_units", "n_layers", "size"]:
                pass
            else:
                print(f"Trying to set hyperparameter {algo_name} which does not exist.")

    # FIXME: as above, logging down here is ugly and we should add it to the progress bar instead
    def evaluate(self, eval_env: MIGHTYENV | None = None) -> Dict:  # type: ignore
        """Eval agent on an environment. (Full rollouts).

        :param env: The environment to evaluate on
        :param episodes: The number of episodes to evaluate
        :return:
        """

        self.logger.set_eval(True)
        terminated, truncated = False, False
        options: Dict = {}
        if eval_env is None:
            eval_env = self.eval_env

        state, _ = eval_env.reset(options=options)  # type: ignore
        rewards = np.zeros(eval_env.num_envs)  # type: ignore
        steps = np.zeros(eval_env.num_envs)  # type: ignore
        mask = np.zeros(eval_env.num_envs)  # type: ignore
        while not np.all(mask):
            action = self.policy(state, evaluate=True)  # type: ignore
            state, reward, terminated, truncated, _ = eval_env.step(action)  # type: ignore
            rewards += reward * (1 - mask)
            steps += 1 * (1 - mask)
            dones = np.logical_or(terminated, truncated)
            mask = np.where(dones, 1, mask)
            self.logger.next_step()

        eval_env.close()  # type: ignore

        if isinstance(self.eval_env, DACENV) or isinstance(self.env, CARLENV):
            instance = eval_env.instance  # type: ignore
        else:
            instance = None
        self.logger.next_episode(instance)
        self.logger.write()

        eval_metrics = {
            "step": self.steps,
            "eval_episodes": np.array(rewards) / steps,
            "mean_eval_step_reward": np.mean(rewards) / steps,
            "mean_eval_reward": np.mean(rewards),
        }
        if instance is not None:
            eval_metrics["instance"] = instance

        # FIXME: this is the ugly I'm talking about
        if self.verbose:
            print("")
            print(
                "------------------------------------------------------------------------------"
            )
            print(
                f"""Evaluation performance after {self.steps} steps:
                {np.round(np.mean(rewards), decimals=2)}"""
            )
            print(
                f"""Evaluation performance per step after {self.steps} steps:
                {np.round(np.mean(rewards/ steps), decimals=2)}"""
            )
            print(
                "------------------------------------------------------------------------------"
            )
            print("")

        self.logger.log_dict(eval_metrics)

        if self.writer is not None:
            self.writer.add_scalars("eval", eval_metrics)

        if self.log_wandb:
            wandb.log(eval_metrics)

        self.logger.set_eval(False)
        return eval_metrics

    def save(self, t: int) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
