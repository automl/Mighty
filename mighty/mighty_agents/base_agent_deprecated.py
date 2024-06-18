import os
from pathlib import Path
from typing import List, Optional, Type, Union

import coax
import hydra
import jax.numpy as jnp
import numpy as np
import wandb
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from jax import vmap
from omegaconf import DictConfig
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from mighty.env.env_handling_deprecated import CARLENV, DACENV, MIGHTYENV
from mighty.mighty_replay import MightyReplay
from mighty.utils.logger_deprecated import Logger
from mighty.utils.types_deprecated import TypeKwargs


def retrieve_class(cls: Union[str, DictConfig, Type], default_cls: Type) -> Type:
    """Get coax or mighty class."""
    if cls is None:
        cls = default_cls
    elif type(cls) is DictConfig:
        cls = hydra.utils.get_class(cls._target_)
    elif type(cls) is str:
        cls = hydra.utils.get_class(cls)
    return cls


class MightyAgent(object):
    """Base agent for Coax RL implementations."""

    def __init__(
        self,
        env: MIGHTYENV,
        logger: Logger,
        eval_env: Optional[MIGHTYENV] = None,
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        batch_size: int = 64,
        learning_starts: int = 1,
        render_progress: bool = True,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        wandb_kwargs: dict = {},
        replay_buffer_class: Optional[
            Union[str, DictConfig, Type[BaseReplayBuffer]]
        ] = None,
        replay_buffer_kwargs: Optional[TypeKwargs] = None,
        tracer_class: Optional[Union[str, DictConfig, Type[BaseRewardTracer]]] = None,
        tracer_kwargs: Optional[TypeKwargs] = None,
        meta_methods: Optional[List[Union[str, Type]]] = [],
        meta_kwargs: Optional[list[TypeKwargs]] = [],
    ):
        """
        Base agent initialization.

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

        self.learning_rate = learning_rate
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._learning_starts = learning_starts

        self.replay_buffer: Optional[BaseReplayBuffer] = None
        self.tracer: Optional[BaseRewardTracer] = None
        self.policy = None

        # Replay Buffer
        replay_buffer_class = retrieve_class(
            cls=replay_buffer_class, default_cls=MightyReplay
        )
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {
                "capacity": 1_000_000,
                "random_seed": None,
            }
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs

        # Reward Tracer
        tracer_class = retrieve_class(
            cls=tracer_class, default_cls=coax.reward_tracing.NStep
        )
        if tracer_kwargs is None:
            tracer_kwargs = {
                "n": 1,
                "gamma": 0.9,
            }
        self.tracer_class = tracer_class
        self.tracer_kwargs = tracer_kwargs

        if logger is not None:
            output_dir = logger.log_dir
        else:
            output_dir = None

        self.env = env
        if eval_env is None:
            self.eval_env = self.env
        else:
            self.eval_env = eval_env

        self.evals = []
        if isinstance(self.eval_env, DACENV) or isinstance(self.eval_env, CARLENV):
            eval_instance_ids = (
                self.eval_env.instance_id_list
                if isinstance(self.eval_env, DACENV)
                else list(self.eval_env.contexts.keys())
            )
            for i in eval_instance_ids:
                self.evals.append(make_eval(self.eval_env, i, logger))
        else:
            self.evals.append(make_eval(self.eval_env, None, logger))

        self.logger = logger
        self.render_progress = render_progress
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.model_dir = os.path.join(self.output_dir, "models")

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

    def _initialize_agent(self):
        """Agent/algorithm specific initializations."""

        raise NotImplementedError

    def initialize_agent(self):
        """
        General initialization of tracer and buffer for all agents.

        Algorithm specific initialization like policies etc. are done in _initialize_agent
        """

        self.tracer = self.tracer_class(
            **self.tracer_kwargs
        )  # specify how to trace the transitions
        self.replay_buffer = self.replay_buffer_class(**self.replay_buffer_kwargs)

        self._initialize_agent()

    def update_agent(self, step):
        """Policy/value function update"""
        raise NotImplementedError

    def adapt_hps(self, metrics):
        """Set hyperparameters."""
        self.learning_rate = metrics["hp/lr"]
        self._epsilon = metrics["hp/pi_epsilon"]

    def train(
        self,
        n_steps: int,
        n_episodes_eval: int,
        eval_every_n_steps: int = 1_000,
        human_log_every_n_steps: int = 5000,
        save_model_every_n_steps: int | None = 5000,
    ):
        """
        Trains the agent for n steps.

        Evaluation is done for the given number of episodes each evaluation interval.

        :param n_steps: The number of training steps
        :param n_episodes_eval: The number of episodes to evaluate
        :param eval_every_n_steps: Evaluation intervall
        :param human_log_every_n_episodes: Intervall for human readable logging to the command line
        :param save_mode_every_n_episodes: Intervall for model checkpointing
        :return:
        """

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
                "Train Steps", total=n_steps, start=False, visible=False
            )
            progress.start_task(steps_task)
            self.steps = 0
            steps_since_eval = 0
            log_reward_buffer = []
            metrics = {
                "env": self.env,
                "vf": self.value_function,
                "policy": self.policy,
                "step": self.steps,
                "hp/lr": self.learning_rate,
                "hp/pi_epsilon": self._epsilon,
            }
            while self.steps < n_steps:
                # Remove rollout data from last episode
                for k in list(metrics.keys()):
                    if "rollout" in k:
                        del metrics[k]

                for k in self.meta_modules.keys():
                    self.meta_modules[k].pre_episode(metrics)

                progress.update(steps_task, visible=True)
                s, info = self.env.reset()
                terminated, truncated = False, False
                episode_reward = 0
                metrics["episode_reward"] = episode_reward
                while not (terminated or truncated) and self.steps < n_steps:
                    for k in self.meta_modules.keys():
                        self.meta_modules[k].pre_step(metrics)
                    self.adapt_hps(metrics)

                    a = self.policy(s, metrics=metrics)
                    s_next, r, terminated, truncated, info = self.env.step(a)
                    episode_reward += r

                    self.logger.log("reward", r)
                    self.logger.log("action", a)
                    self.logger.log("next_state", s_next)
                    self.logger.log("state", s)
                    self.logger.log("terminated", terminated)
                    self.logger.log("truncated", truncated)
                    t = {
                        "step": self.steps,
                        "reward": r,
                        "action": a,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    }
                    metrics["episode_reward"] = episode_reward

                    if self.writer is not None:
                        self.writer.add_scalars("transition", t, global_step=self.steps)

                    if self.log_wandb:
                        wandb.log(t)

                    for k in self.meta_modules.keys():
                        self.meta_modules[k].post_step(metrics)

                    log_reward_buffer.append(r)
                    self.steps += 1
                    steps_since_eval += 1
                    progress.advance(steps_task)

                    # add transition to buffer
                    self.tracer.add(s, a, r, terminated or truncated)
                    while self.tracer:
                        transition = self.tracer.pop()
                        transition_metrics = self.get_transition_metrics(
                            transition, metrics
                        )
                        metrics.update(transition_metrics)
                        if isinstance(transition.extra_info, dict):
                            transition.extra_info.update(info)
                        else:
                            transition.extra_info = info
                        self.replay_buffer.add(transition, transition_metrics)

                    # update
                    if (
                        len(self.replay_buffer) >= self._batch_size
                        and self.steps >= self._learning_starts
                    ):
                        for k in self.meta_modules.keys():
                            self.meta_modules[k].pre_step(metrics)

                        agent_update_metrics = self.update_agent(self.steps)
                        metrics.update(agent_update_metrics)
                        metrics = {k: np.array(v) for k, v in metrics.items()}
                        metrics["step"] = self.steps

                        if self.writer is not None:
                            self.writer.add_scalars(
                                "training_metrics", metrics, global_step=self.steps
                            )

                        if self.log_wandb:
                            wandb.log(metrics)

                        metrics["env"] = self.env
                        metrics["vf"] = self.value_function
                        metrics["policy"] = self.policy
                        for k in self.meta_modules.keys():
                            self.meta_modules[k].pre_step(metrics)

                    self.last_state = s
                    s = s_next
                    self.logger.next_step()

                    if eval_every_n_steps:
                        if steps_since_eval >= eval_every_n_steps:
                            steps_since_eval = 0
                            eval_metrics_list = self.evaluate(n_episodes_eval=n_episodes_eval)

                    if self.steps % human_log_every_n_steps == 0:
                        print(
                            f"Steps: {self.steps}, Reward: {sum(log_reward_buffer) / len(log_reward_buffer)}"
                        )
                        log_reward_buffer = []

                    if save_model_every_n_steps:
                        if self.steps % save_model_every_n_steps == 0:
                            self.save(self.steps)

                if isinstance(self.env, DACENV):
                    instance = self.env.instance
                elif isinstance(self.env, CARLENV):
                    instance = self.env.context
                else:
                    instance = None
                self.logger.next_episode(instance)
                episodes += 1
                for k in self.meta_modules.keys():
                    self.meta_modules[k].post_episode(metrics)

        # At the end make sure logger writes buffer to file
        self.logger.write()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def evaluate(self, n_episodes_eval: int) -> list[dict]:
        self.logger.set_eval(True)
        eval_metrics_list = []
        for e in self.evals:
            eval_metrics = vmap(e, in_axes=(None, None, 0), out_axes=0)(
                self.policy,
                self.steps,
                jnp.arange(n_episodes_eval),
            )
            eval_metrics_list.append(eval_metrics)

            if self.writer is not None:
                self.writer.add_scalars("eval", eval_metrics)

            if self.log_wandb:
                wandb.log(eval_metrics)

        self.logger.set_eval(False)
        return eval_metrics_list

    def get_state(self):
        """Return internal state for checkpointing."""
        raise NotImplementedError

    def set_state(self, state):
        """Set internal state after loading."""
        raise NotImplementedError

    def load(self, path):
        """
        Load checkpointed model.

        :param path: Model path
        :return:
        """
        state = coax.utils.load(path)
        self.set_state(state=state)

    def save(self, T):
        """
        Checkpoint model.

        :param T: Current timestep
        :return:
        """
        logdir = self.logger.log_dir
        filepath = Path(os.path.join(logdir, "checkpoints", f"checkpoint_{T}.pkl.lz4"))
        if not filepath.is_file() or True:  # TODO build logic
            state = self.get_state()
            coax.utils.dump(state, str(filepath))
            print(f"Saved checkpoint to {filepath}")

    def __del__(self):
        """Close wandb upon deletion."""
        if self.log_wandb:
            wandb.finish()


def make_eval(env, instance_id, logger):
    """Eval constructor function."""
    def eval(policy, steps, _):
        """
        Eval agent on an environment. (Full rollouts)

        :param env: The environment to evaluate on
        :param episodes: The number of episodes to evaluate
        :return:
        """

        rewards = []
        terminated, truncated = False, False
        options = {}
        if instance_id is not None:
            options = {"instance_id": instance_id}
        state, _ = env.reset(options=options)
        r = 0
        while not (terminated or truncated):
            action = policy(state, eval=True)
            state, reward, terminated, truncated, _ = env.step(action)
            r += reward
            logger.next_step()
        rewards.append(r)

        if isinstance(env, DACENV):
            instance = env.instance
        elif isinstance(env, CARLENV):
            instance = env.context
        else:
            instance = None
        logger.next_episode(instance)
        logger.write()

        eval_metrics = {
            "step": steps,
            "eval_episodes": np.array(rewards),
            "mean_eval_reward": np.mean(rewards),
        }
        if instance is not None:
            eval_metrics["instance"] = instance
        return eval_metrics

    return eval