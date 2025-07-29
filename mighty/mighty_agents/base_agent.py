"""Base agent template."""

from __future__ import annotations

import os
import random
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import pandas as pd
import torch
import wandb
import json
from omegaconf import DictConfig
from rich import print
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from uniplot import plot_to_string

from mighty.mighty_exploration import MightyExplorationPolicy
from mighty.mighty_replay import MightyReplay, MightyRolloutBuffer, PrioritizedReplay
from mighty.mighty_utils.mighty_types import CARLENV, DACENV, MIGHTYENV, retrieve_class

if TYPE_CHECKING:
    from mighty.mighty_utils.mighty_types import TypeKwargs

import gymnasium as gym
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward


def seed_everything(seed: int, env: gym.Env | None = None):
    """
    Seed Python, NumPy, Torch (including cuDNN), plus Gym (action_space, observation_space,
    and the environment's own RNG). If `env` is vectorized (has `env.envs`), we dig into each
    sub-env. Always call this before ANY network-building or RNG usage.
    """
    # 1) Python/NumPy/Torch/Hash
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # enforce strictly deterministic ops
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) Gym environment seeding
    if env is not None:
        # If the env is wrapped, try to unwrap to the core
        try:
            core_env = env.unwrapped
            core_env.seed(seed)
        except Exception:
            pass

        # If vectorized (e.g. SyncVectorEnv), seed each sub‐env separately
        if hasattr(env, "envs") and isinstance(env.envs, list):
            sub_seeds = [seed for _ in range(len(env.envs))]
            for sub_seed, subenv in zip(sub_seeds, env.envs):
                subenv.action_space.seed(sub_seed)
                subenv.observation_space.seed(sub_seed)
                try:
                    subenv.unwrapped.seed(sub_seed)
                except Exception:
                    pass
            # Reset the vectorized env with explicit seeds
            env.reset(seed=sub_seeds)
        else:
            # Single environment
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            try:
                env.unwrapped.seed(seed)
            except Exception:
                pass
            env.reset(seed=seed)


def update_buffer(buffer, new_data):
    for k in buffer.keys():
        buffer[k].append(new_data[k])
    return buffer


def log_to_file(output_dir, result_buffer, hp_buffer, eval_buffer, loss_buffer):
    if loss_buffer is not None:
        loss_df = pd.DataFrame(loss_buffer)
        if (Path(output_dir) / "losses.csv").exists():
            (Path(output_dir) / "losses.csv").unlink()
        loss_df.to_csv(Path(output_dir) / "losses.csv")

    if (Path(output_dir) / "results.npz").exists():
        (Path(output_dir) / "results.npz").unlink()
    np.savez(Path(output_dir) / "results.npz", result_buffer)
    result_df = pd.DataFrame(result_buffer)
    if (Path(output_dir) / "results.csv").exists():
        (Path(output_dir) / "results.csv").unlink()
    result_df.to_csv(Path(output_dir) / "results.csv")

    hp_df = pd.DataFrame(hp_buffer)
    if (Path(output_dir) / "hyperparameters.csv").exists():
        (Path(output_dir) / "hyperparameters.csv").unlink()
    hp_df.to_csv(Path(output_dir) / "hyperparameters.csv")

    eval_df = pd.DataFrame(eval_buffer)
    if (Path(output_dir) / "eval_results.csv").exists():
        (Path(output_dir) / "eval_results.csv").unlink()
    eval_df.to_csv(Path(output_dir) / "eval_results.csv")


def log_to_wandb(metrics: Dict) -> None:
    """Wandb logging"""
    # Only log relevant, serializable keys
    log_keys = [
        "step",
        "episode_reward",
        "Update/policy_loss",
        "Update/value_loss",
        "Update/entropy",
        "Update/approx_kl",
    ]
    serializable_metrics = {}
    for k in log_keys:
        if k in metrics:
            v = metrics[k]
            # Convert numpy arrays to scalars or lists
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    v = v.item()
                else:
                    v = v.tolist()
            # Convert torch tensors to scalars or lists
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.cpu().numpy().tolist()
            # Try to serialize, skip if not possible
            try:
                json.dumps(v)
                serializable_metrics[k] = v
            except TypeError:
                print(f"Skipping non-serializable metric: {k}")

    wandb.log(serializable_metrics, step=metrics["step"])


class MightyAgent(ABC):
    """Base agent for RL implementations."""

    def __init__(  # noqa: PLR0915, PLR0912
        self,
        output_dir,
        env: MIGHTYENV,  # type: ignore
        seed: int | None = None,
        eval_env: MIGHTYENV | None = None,  # type: ignore
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        batch_size: int = 64,
        learning_starts: int = 1,
        n_gradient_steps: int = 1,
        render_progress: bool = True,
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
        normalize_obs: bool = False,  # ← NEW
        normalize_reward: bool = False,  # ← NEW (optional)
    ):
        """Base agent initialization.

        Creates all relevant class variables and calls agent-specific init function

        :param env: Train environment
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
        self.n_gradient_steps = n_gradient_steps

        self.buffer: MightyReplay | None = None
        self.policy: MightyExplorationPolicy | None = None

        self.seed = seed
        if self.seed is not None:
            seed_everything(self.seed, env=None)  # type: ignore
            # Re-seed Python/NumPy/Torch here again.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["PYTHONHASHSEED"] = str(seed)

            # Also seed any RNGs you will use later (e.g. for buffer / policy):
            self.rng = np.random.default_rng(seed)  # for any numpy-based sampling
            # If you ever use torch.Generator for sampling actions, you could do:
            self.torch_gen = torch.Generator().manual_seed(seed)

            # If you use Python’s random inside your policy, call random.seed(seed)
            random.seed(seed)

        else:
            self.rng = np.random.default_rng()
            self.seed = 0

        # Replay Buffer
        replay_buffer_class = retrieve_class(
            cls=replay_buffer_class,
            default_cls=MightyReplay,  # type: ignore
        )

        if replay_buffer_kwargs is None or len(replay_buffer_kwargs) == 0:
            if issubclass(replay_buffer_class, MightyReplay):
                replay_buffer_kwargs = {  # type: ignore
                    "capacity": 1_000_000,
                }
            else:
                replay_buffer_kwargs = {}

        self.buffer_class = replay_buffer_class
        self.buffer_kwargs = replay_buffer_kwargs

        self.output_dir = output_dir
        self.verbose = verbose

        if normalize_obs:
            env = NormalizeObservation(env)
            if eval_env is not None:
                eval_env = NormalizeObservation(eval_env)

        if normalize_reward:
            env = NormalizeReward(env)
            if eval_env is not None:
                eval_env = NormalizeReward(eval_env)

        self.env = env
        if eval_env is None:
            self.eval_env = self.env
        else:
            self.eval_env = eval_env

        self.render_progress = render_progress
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.model_dir = Path(self.output_dir) / Path("models")

        # Create meta modules
        self.meta_modules = {}
        for i, m in enumerate(meta_methods):
            meta_class = retrieve_class(cls=m, default_cls=None)  # type: ignore
            assert meta_class is not None, (
                f"Class {m} not found, did you specify the correct loading path?"
            )
            kwargs: Dict = {}
            if len(meta_kwargs) > i:
                kwargs = meta_kwargs[i]
            self.meta_modules[meta_class.__name__] = meta_class(**kwargs)

        self.last_state = None
        self.total_steps = 0

        self.result_buffer = {
            "seed": [],
            "step": [],
            "reward": [],
            "action": [],
            "state": [],
            "next_state": [],
            "terminated": [],
            "truncated": [],
            "mean_episode_reward": [],
        }

        self.eval_buffer = {
            "step": [],
            "seed": [],
            "eval_episodes": [],
            "mean_eval_step_reward": [],
            "mean_eval_reward": [],
            "instance": [],
        }

        self.hp_buffer = {
            "step": [],
            "hp/lr": [],
            "hp/pi_epsilon": [],
            "hp/batch_size": [],
            "hp/learning_starts": [],
            "meta_modules": [],
        }
        self.loss_buffer = None
        starting_hps = {
            "step": 0,
            "hp/lr": self.learning_rate,
            "hp/pi_epsilon": self._epsilon,
            "hp/batch_size": self._batch_size,
            "hp/learning_starts": self._learning_starts,
            "meta_modules": list(self.meta_modules.keys()),
        }
        self.hp_buffer = update_buffer(self.hp_buffer, starting_hps)

        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(**wandb_kwargs)
            wandb.log(starting_hps)

        self.initialize_agent()
        self.steps = 0

        seed_everything(self.seed, self.env)  # type: ignore
        seed_everything(self.seed, self.eval_env)  # type: ignore

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

        if isinstance(self.buffer_class, type) and issubclass(
            self.buffer_class, PrioritizedReplay
        ):
            # 1) Get observation-space shape
            try:
                obs_space = self.env.single_observation_space
                obs_shape = tuple(obs_space.shape)
            except Exception:
                # Fallback: call env.reset() once and infer shape from returned numpy/torch array
                first_obs, _ = self.env.reset(seed=self.seed)
                obs_shape = tuple(np.array(first_obs).shape)

            # 2) Get action-space shape (if discrete, .n is number of actions)
            action_space = self.env.single_action_space
            if hasattr(action_space, "n"):
                # Discrete action space → action_shape = () (scalar), but Q-net will expect a single integer
                # We store it as a zero-length tuple, and treat it as int later.
                action_shape = ()
            else:
                # Continuous action space, e.g. Box(shape=(3,)), so we store that tuple
                action_shape = tuple(action_space.shape)

            # 3) Overwrite the YAML placeholders (null → actual)
            self.buffer_kwargs["obs_shape"] = obs_shape
            self.buffer_kwargs["action_shape"] = action_shape

        self.buffer = self.buffer_class(**self.buffer_kwargs)  # type: ignore

    def update_agent(self) -> Dict:
        """Policy/value function update."""
        raise NotImplementedError

    def adapt_hps(self, metrics: Dict) -> None:
        """Set hyperparameters."""
        old_hps = {
            "step": self.steps,
            "hp/lr": self.learning_rate,
            "hp/pi_epsilon": self._epsilon,
            "hp/batch_size": self._batch_size,
            "hp/learning_starts": self._learning_starts,
            "meta_modules": list(self.meta_modules.keys()),
        }
        self.learning_rate = metrics["hp/lr"]
        self._epsilon = metrics["hp/pi_epsilon"]
        self._batch_size = metrics["hp/batch_size"]
        self._learning_starts = metrics["hp/learning_starts"]

        updated_hps = {
            "step": self.steps,
            "hp/lr": self.learning_rate,
            "hp/pi_epsilon": self._epsilon,
            "hp/batch_size": self._batch_size,
            "hp/learning_starts": self._learning_starts,
            "meta_modules": list(self.meta_modules.keys()),
        }

        if any(old_hps[k] != updated_hps[k] for k in old_hps.keys()):
            self.hp_buffer = update_buffer(self.hp_buffer, updated_hps)

    def make_checkpoint_dir(self, t: int) -> None:
        """Checkpoint model.

        :param T: Current timestep
        :return:
        """
        self.upper_checkpoint_dir = Path(self.output_dir) / Path("checkpoints")
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

        batches = []
        for batches_left in reversed(range(self.n_gradient_steps)):
            batch = self.buffer.sample(self._batch_size)
            agent_update_metrics = self.update_agent(
                transition_batch=batch, batches_left=batches_left, **update_kwargs
            )

            metrics.update(agent_update_metrics)
            metrics["step"] = self.steps

            if self.log_wandb:
                log_to_wandb(metrics=metrics)

            metrics["env"] = self.env
            metrics["vf"] = self.value_function  # type: ignore
            metrics["policy"] = self.policy
            batches.append(batch)

        metrics["update_batches"] = batches
        for k in self.meta_modules:
            self.meta_modules[k].post_update(metrics)
        del metrics["update_batches"]
        return metrics

    def make_logging_table(
        self, step, episode_reward, step_reward, evaluation_reward, actions
    ) -> Table:
        table = Table(title="Training Stats")
        table.add_column("Metric", style="yellow", justify="center")
        table.add_column("At step", style="magenta", justify="center")
        table.add_column("Latest Mean", style="yellow", justify="center")
        table.add_column("Latest Std", style="yellow", justify="center")

        table.add_row(
            "Mean Episode Reward",
            f"{step}",
            str(np.round(np.mean(episode_reward), decimals=2)),
            str(np.round(np.std(episode_reward), decimals=2)),
        )
        table.add_row(
            "Mean Step Reward",
            f"{step}",
            str(np.round(np.mean(step_reward), decimals=2)),
            str(np.round(np.std(step_reward), decimals=2)),
        )
        table.add_row(
            "Mean Eval Reward",
            f"{step}",
            str(np.round(np.mean(evaluation_reward), decimals=2)),
            str(np.round(np.std(evaluation_reward), decimals=2)),
        )
        table.add_row(
            "Mean Action",
            f"{step}",
            str(np.round(np.mean(actions), decimals=2)),
            str(np.round(np.std(actions), decimals=2)),
        )
        return table

    def get_plot(self, steps, performances: list, label) -> Panel:
        return Panel(
            plot_to_string(
                xs=steps,
                ys=performances,
                title=label,
                height=8,
                x_min=0,
                x_unit="steps",
                # y_min=0,
                y_unit="reward",
                lines=True,
            )
        )

    def make_logging_layout(self, n_steps: int) -> Layout:
        logging_layout = Layout()
        logging_layout.split_column(
            Layout(name="upper"), Layout(name="middle"), Layout(name="lower")
        )
        logging_layout["middle"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        logging_layout["lower"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        metrics_table = self.make_logging_table(0, [0], [0], [0], [0])  # type: ignore
        training_detail_table = Table(title="Training Details")
        training_detail_table.add_column("Name", style="cyan", justify="center")
        training_detail_table.add_column("Value", style="cyan", justify="center")
        training_detail_table.add_row("Training Steps", str(n_steps))
        training_detail_table.add_row("Seed", str(self.seed))
        training_detail_table.add_row("Learning Rate", str(self.learning_rate))
        training_detail_table.add_row("Batch Size", str(self._batch_size))

        logging_layout["middle"]["left"].update(metrics_table)
        logging_layout["middle"]["right"].update(training_detail_table)

        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "Remaining:",
            TimeRemainingColumn(),
            "Elapsed:",
            TimeElapsedColumn(),
            disable=not self.render_progress,
        )
        steps_task = progress.add_task(
            "Train Steps",
            total=n_steps - self.steps,
            start=False,
            visible=False,
        )
        progress.start_task(steps_task)

        progress_table = Table.grid(expand=True)
        progress_table.add_row(
            Panel(
                progress,
                title="Training Progress",
                border_style="green",
                padding=(2, 2),
            ),
        )
        logging_layout["upper"].update(progress_table)
        return logging_layout, progress, steps_task

    def run(  # noqa: PLR0915
        self,
        n_steps: int,
        eval_every_n_steps: int = 1_000,
        save_model_every_n_steps: int | None = 5000,
        env: MIGHTYENV = None,  # type: ignore
    ) -> Dict:
        """Run agent."""
        episodes = 0
        if env is not None:
            self.env = env

        logging_layout, progress, steps_task = self.make_logging_layout(n_steps)
        update_multiplier = 0

        with Live(logging_layout, refresh_per_second=10, vertical_overflow="visible"):
            steps_since_eval = 0
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
            curr_s, _ = self.env.reset(seed=self.seed)  # type: ignore
            if len(curr_s.squeeze().shape) == 0:
                episode_reward = [0]
            else:
                episode_reward = np.zeros(curr_s.squeeze().shape[0])  # type: ignore

            last_episode_reward = episode_reward
            if not torch.is_tensor(last_episode_reward):
                last_episode_reward = torch.tensor(last_episode_reward).float()

            recent_episode_reward = []
            recent_step_reward = []
            recent_actions = []
            evaluation_reward = []

            # Start logging
            eval_curve = [0]
            learning_curve = [0]
            curve_xs = [0]
            progress.update(steps_task, visible=True)
            logging_layout["lower"]["left"].update(
                self.get_plot(curve_xs, learning_curve, "Training Reward")
            )
            logging_layout["lower"]["right"].update(
                self.get_plot(curve_xs, eval_curve, "Evaluation Reward")
            )

            # Main loop: rollouts, training and evaluation
            while self.steps < n_steps:
                metrics["episode_reward"] = episode_reward

                action, log_prob = self.step(curr_s, metrics)
                next_s, reward, terminated, truncated, _ = self.env.step(action)  # type: ignore
                dones = np.logical_or(terminated, truncated)

                episode_reward += reward

                # Log everything
                t = {
                    "seed": self.seed,
                    "step": self.steps,
                    "reward": reward,
                    "action": action,
                    "state": curr_s,
                    "next_state": next_s,
                    "terminated": terminated.astype(int),
                    "truncated": truncated.astype(int),
                    "dones": dones.astype(int),
                    "mean_episode_reward": last_episode_reward.mean()
                    .cpu()
                    .numpy()
                    .item(),
                }
                metrics["log_prob"] = log_prob.detach().cpu().numpy()
                metrics["episode_reward"] = episode_reward
                metrics["transition"] = t

                recent_actions.append(np.mean(action))
                if len(recent_actions) > 100:
                    recent_actions.pop(0)

                for k in self.meta_modules:
                    self.meta_modules[k].post_step(metrics)

                transition_metrics = self.process_transition(
                    metrics["transition"]["state"],
                    metrics["transition"]["action"],
                    metrics["transition"]["reward"],
                    metrics["transition"]["next_state"],
                    metrics["transition"]["dones"],
                    metrics["log_prob"],
                    metrics,
                )
                metrics.update(transition_metrics)
                self.result_buffer = update_buffer(self.result_buffer, t)

                if self.log_wandb:
                    wandb.log(t)

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

                # Evaluate
                if eval_every_n_steps and steps_since_eval >= eval_every_n_steps:
                    steps_since_eval = 0
                    eval_metrics = self.evaluate()
                    evaluation_reward = eval_metrics["eval_rewards"]

                # Log to command line via rich layout
                if self.steps >= 1000 * update_multiplier:
                    metrics_table = self.make_logging_table(
                        self.steps,
                        recent_episode_reward,
                        recent_step_reward,
                        evaluation_reward,
                        recent_actions,
                    )
                    logging_layout["middle"]["left"].update(metrics_table)
                    eval_curve.append(np.mean(evaluation_reward))
                    learning_curve.append(np.mean(recent_episode_reward))
                    curve_xs.append(self.steps)

                    logging_layout["lower"]["left"].update(
                        self.get_plot(curve_xs, learning_curve, "Training Reward")
                    )
                    logging_layout["lower"]["right"].update(
                        self.get_plot(curve_xs, eval_curve, "Evaluation Reward")
                    )
                    update_multiplier += 1

                # Save model & metrics
                if (
                    save_model_every_n_steps
                    and self.steps % save_model_every_n_steps == 0
                ):
                    self.save(self.steps)
                    log_to_file(
                        self.output_dir,
                        self.result_buffer,
                        self.hp_buffer,
                        self.eval_buffer,
                        self.loss_buffer,
                    )

                # Perform resets as necessary
                if np.any(dones):
                    last_episode_reward = np.where(  # type: ignore
                        dones, episode_reward, last_episode_reward
                    )
                    recent_episode_reward.append(np.mean(last_episode_reward))
                    recent_step_reward.append(
                        np.mean(last_episode_reward) / len(last_episode_reward)
                    )
                    last_episode_reward = torch.tensor(last_episode_reward).float()
                    if len(recent_episode_reward) > 10:
                        recent_episode_reward.pop(0)
                        recent_step_reward.pop(0)
                    episode_reward = np.where(dones, 0, episode_reward)  # type: ignore
                    # End episode
                    if isinstance(self.env, DACENV) or isinstance(self.env, CARLENV):
                        instance = self.env.instance  # type: ignore
                    else:
                        instance = None
                    metrics["instance"] = instance
                    episodes += 1
                    for k in self.meta_modules:
                        self.meta_modules[k].post_episode(metrics)

                    if "rollout_values" in metrics:
                        del metrics["rollout_values"]

                    if "rollout_logits" in metrics:
                        del metrics["rollout_logits"]

                    # Meta Module hooks
                    for k in self.meta_modules:
                        self.meta_modules[k].pre_episode(metrics)

        # Final logging
        log_to_file(
            self.output_dir,
            self.result_buffer,
            self.hp_buffer,
            self.eval_buffer,
            self.loss_buffer,
        )
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

    def evaluate(self, eval_env: MIGHTYENV | None = None) -> Dict:  # type: ignore
        """Eval agent on an environment. (Full rollouts).

        :param env: The environment to evaluate on
        :param episodes: The number of episodes to evaluate
        :return:
        """

        terminated, truncated = False, False
        options: Dict = {}
        if eval_env is None:
            eval_env = self.eval_env

        state, _ = eval_env.reset(options=options, seed=self.seed)  # type: ignore
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

        eval_env.close()  # type: ignore

        if isinstance(self.eval_env, DACENV) or isinstance(self.env, CARLENV):
            instance = eval_env.instance  # type: ignore
        else:
            instance = "None"

        eval_metrics = {
            "step": self.steps,
            "seed": self.seed,
            "eval_episodes": np.array(rewards) / steps,
            "mean_eval_step_reward": np.mean(rewards) / steps,
            "mean_eval_reward": np.mean(rewards),
            "instance": instance,
            "eval_rewards": rewards,
        }
        self.eval_buffer = update_buffer(self.eval_buffer, eval_metrics)

        if self.log_wandb:
            wandb.log(eval_metrics)

        return eval_metrics

    def save(self, t: int) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
