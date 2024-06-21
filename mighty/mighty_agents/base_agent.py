"""Base agent template."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import hydra
import numpy as np
import torch
import wandb
from mighty.mighty_replay import MightyReplay, TransitionBatch
from mighty.utils.env_handling import CARLENV, DACENV, MIGHTYENV
from omegaconf import DictConfig
from rich import print
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

if TYPE_CHECKING:
    from mighty.utils.logger import Logger
    from mighty.utils.types import TypeKwargs


def retrieve_class(cls: str | DictConfig | type, default_cls: type) -> type:
    """Get coax or mighty class."""
    if cls is None:
        cls = default_cls
    elif isinstance(cls, DictConfig):
        cls = hydra.utils.get_class(cls._target_)
    elif isinstance(cls, str):
        cls = hydra.utils.get_class(cls)
    return cls


class MightyAgent:
    """Base agent for Coax RL implementations."""

    def __init__(  # noqa: PLR0915, PLR0912
        self,
        env: MIGHTYENV,
        logger: Logger,
        seed: int | None = None,
        eval_env: MIGHTYENV | None = None,
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        batch_size: int = 64,
        learning_starts: int = 1,
        render_progress: bool = True,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        wandb_kwargs: dict | None = None,
        replay_buffer_class: str | DictConfig | type[MightyReplay] | None = None,
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

        self.replay_buffer: MightyReplay | None = None
        self.policy = None

        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            torch.manual_seed(seed)
        else:
            self.rng = np.random.default_rng()

        # Replay Buffer
        replay_buffer_class = retrieve_class(
            cls=replay_buffer_class, default_cls=MightyReplay
        )
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {
                "capacity": 1_000_000,
            }
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs

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
            meta_class = retrieve_class(cls=m, default_cls=None)
            assert (
                meta_class is not None
            ), f"Class {m} not found, did you specify the correct loading path?"
            kwargs = {}
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

    def _initialize_agent(self):
        """Agent/algorithm specific initializations."""
        raise NotImplementedError

    def initialize_agent(self):
        """General initialization of tracer and buffer for all agents.

        Algorithm specific initialization like policies etc.
        are done in _initialize_agent
        """
        self.replay_buffer = self.replay_buffer_class(**self.replay_buffer_kwargs)

        self._initialize_agent()

    def update_agent(self):
        """Policy/value function update."""
        raise NotImplementedError

    def adapt_hps(self, metrics):
        """Set hyperparameters."""
        self.learning_rate = metrics["hp/lr"]
        self._epsilon = metrics["hp/pi_epsilon"]
        self._batch_size = metrics["hp/batch_size"]
        self._learning_starts = metrics["hp/learning_starts"]
        return metrics

    def make_checkpoint_dir(self, t):
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

    def __del__(self):
        """Close wandb upon deletion."""
        self.env.close()
        if self.log_wandb:
            wandb.finish()

    def step(self, observation, metrics):
        """This is a util function for the runner,
        combining meta_modules and prediction.
        """
        for k in self.meta_modules.keys():
            self.meta_modules[k].pre_step(metrics)

        metrics = self.adapt_hps(metrics)
        return self.policy(observation, metrics=metrics)

    def update(self, metrics):
        """Update agent."""
        for k in self.meta_modules:
            self.meta_modules[k].pre_update(metrics)

        agent_update_metrics = self.update_agent()
        metrics.update(agent_update_metrics)
        metrics = {k: np.array(v) for k, v in metrics.items()}
        metrics["step"] = self.steps

        if self.writer is not None:
            self.writer.add_scalars("training_metrics", metrics, global_step=self.steps)

        if self.log_wandb:
            wandb.log(metrics)

        metrics["env"] = self.env
        metrics["vf"] = self.value_function
        metrics["policy"] = self.policy
        for k in self.meta_modules:
            self.meta_modules[k].post_update(metrics)
        return metrics

    def run(  # noqa: PLR0915
        self,
        n_steps: int,
        n_episodes_eval: int,
        eval_every_n_steps: int = 1_000,
        human_log_every_n_steps: int = 5000,
        save_model_every_n_steps: int | None = 5000,
    ):
        """Run agent."""
        episodes = 0
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
                "Train Steps", total=n_steps - self.steps, start=False, visible=False
            )
            steps_since_eval = 0
            progress.start_task(steps_task)
            metrics = {
                "env": self.env,
                "vf": self.value_function,
                "policy": self.policy,
                "step": self.steps,
                "hp/lr": self.learning_rate,
                "hp/pi_epsilon": self._epsilon,
                "hp/batch_size": self._batch_size,
                "hp/learning_starts": self._learning_starts,
            }
            s, _ = self.env.reset()
            if len(s.squeeze().shape) == 0:
                episode_reward = [0]
            else:
                episode_reward = np.zeros(s.squeeze().shape[0])
            last_episode_reward = episode_reward
            progress.update(steps_task, visible=True)
            while self.steps < n_steps:
                metrics["episode_reward"] = episode_reward
                action = self.step(s, metrics)
                next_s, reward, terminated, truncated, _ = self.env.step(action)
                dones = np.logical_or(terminated, truncated)
                transition = TransitionBatch(s, action, reward, next_s, dones)
                transition_metrics = self.get_transition_metrics(transition, metrics)
                metrics.update(transition_metrics)
                self.replay_buffer.add(transition, metrics)
                episode_reward += reward

                # Log everything
                t = {
                    "seed": self.seed,
                    "step": self.steps,
                    "reward": reward.tolist(),
                    "action": action.tolist(),
                    "state": s.tolist(),
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
                    len(self.replay_buffer) >= self._batch_size
                    and self.steps >= self._learning_starts
                ):
                    metrics = self.update(metrics)

                # End step
                self.last_state = s
                s = next_s
                self.logger.next_step()

                # Evaluate
                if eval_every_n_steps and steps_since_eval >= eval_every_n_steps:
                    steps_since_eval = 0
                    self.evaluate(n_eval_episodes=n_episodes_eval)

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
                    last_episode_reward = np.where(
                        dones, episode_reward, last_episode_reward
                    )
                    episode_reward = np.where(dones, 0, episode_reward)
                    # End episode
                    if isinstance(self.env, DACENV):
                        instance = self.env.instance
                    elif isinstance(self.env, CARLENV):
                        instance = self.env.context
                    else:
                        instance = None
                    self.logger.next_episode(instance)
                    episodes += 1
                    for k in self.meta_modules:
                        self.meta_modules[k].post_episode(metrics)

                    # Remove rollout data from last episode
                    # TODO: only do this for finished envs
                    for k in list(metrics.keys()):
                        if "rollout" in k:
                            del metrics[k]

                    for k in self.meta_modules:
                        self.meta_modules[k].pre_episode(metrics)
        return metrics

    def apply_config(self, config):
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

    def evaluate(self, n_eval_episodes):
        """Eval agent on an environment. (Full rollouts).

        :param env: The environment to evaluate on
        :param episodes: The number of episodes to evaluate
        :return:
        """
        self.logger.set_eval(True)
        terminated, truncated = False, False
        options = {}
        eval_env = self.eval_env()

        state, _ = eval_env.reset(options=options)
        rewards = np.zeros(n_eval_episodes)
        steps = np.zeros(n_eval_episodes)
        mask = np.zeros(n_eval_episodes)
        while not np.all(mask):
            action = self.policy(state, evaluate=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            rewards += reward * (1 - mask)
            steps += 1 * (1 - mask)
            dones = np.logical_or(terminated, truncated)
            mask = np.where(dones, 1, mask)
            self.logger.next_step()

        eval_env.close()

        if isinstance(self.eval_env, DACENV):
            instance = eval_env.instance
        elif isinstance(self.eval_env, CARLENV):
            instance = eval_env.context
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
