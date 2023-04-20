from pathlib import Path
import os
from typing import Optional, Union, Type, List
import hydra
from omegaconf import DictConfig
import numpy as np
import wandb

import jax.numpy as jnp
from jax import vmap
import coax
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from typing import Optional
from rich.progress import Progress, TimeRemainingColumn, TimeElapsedColumn, BarColumn

from mighty.env.env_handling import MIGHTYENV, DACENV, CARLENV
from mighty.utils.logger import Logger
from mighty.utils.types import TypeKwargs
from mighty.mighty_replay import MightyReplay


def retrieve_class(cls: Union[str, DictConfig, Type], default_cls: Type) -> Type:
    if cls is None:
        cls = default_cls
    elif type(cls) == DictConfig:
        cls = hydra.utils.get_class(cls._target_)
    elif type(cls) == str:
        cls = hydra.utils.get_class(cls)
    return cls


class MightyAgent(object):
    """
    Base agent for Coax RL implementations
    """

    def __init__(
        self,
        env: MIGHTYENV,
        logger: Logger,
        eval_env: Optional[MIGHTYENV] = None,
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        batch_size: int = 64,
        render_progress: bool = True,  # FIXME: Does this actually do anything or can we take it out?
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
        Base agent initialization
        Creates all relevant class variables and calls agent-specific init function

        :param env: Train environment
        :param logger: Mighty logger
        :param eval_env: Evaluation environment
        :param learning_rate: Learning rate for training
        :param epsilon: Exploration factor for training
        :param batch_size: Batch size for training
        :param render_progress: Render progress
        :param log_tensorboard: Log to tensorboard as well as to file
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

        self.replay_buffer: Optional[BaseReplayBuffer] = None
        self.tracer: Optional[BaseRewardTracer] = None
        self.policy: Optional = None

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

        if logger is not None:
            output_dir = logger.log_dir
        else:
            output_dir = None

        self.env = env
        if eval_env is None:
            self.eval_env = self.env
        else:
            self.eval_env = eval_env

        self.logger = logger
        self.render_progress = render_progress
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.model_dir = os.path.join(self.output_dir, "models")

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

    def train(
        self,
        n_steps: int,
        n_episodes_eval: int,
        eval_every_n_steps: int = 1_000,
        human_log_every_n_steps: int = 5000,
        save_model_every_n_steps: int = 5000,
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
        step_progress = 1 / n_steps
        episodes = 0
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "Remaining:",
            TimeRemainingColumn(),
            "Elapsed:",
            TimeElapsedColumn(),
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
                "vf": self.vf,
                "policy": self.policy,
                "step": self.steps,
            }
            while self.steps < n_steps:
                # Remove rollout data from last episode
                for k in list(metrics.keys()):
                    if "rollout" in k:
                        del metrics[k]

                for k in self.meta_modules.keys():
                    self.meta_modules[k].pre_episode(metrics)

                progress.update(steps_task, visible=True)
                s, _ = self.env.reset()
                terminated, truncated = False, False
                episode_reward = 0
                metrics["episode_reward"] = episode_reward
                while not (terminated or truncated):
                    for k in self.meta_modules.keys():
                        self.meta_modules[k].pre_step(metrics)

                    a = self.policy(s, metrics=metrics)
                    s_next, r, terminated, truncated, _ = self.env.step(a)
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
                        self.replay_buffer.add(transition, transition_metrics)

                    # update
                    if len(self.replay_buffer) >= self._batch_size:
                        for k in self.meta_modules.keys():
                            self.meta_modules[k].pre_step(metrics)

                        metrics.update(self.update_agent(self.steps))
                        metrics = {k: np.array(v) for k, v in metrics.items()}
                        metrics["step"] = self.steps

                        if self.writer is not None:
                            self.writer.add_scalars(
                                "training_metrics", metrics, global_step=self.steps
                            )

                        if self.log_wandb:
                            wandb.log(metrics)

                        metrics["env"] = self.env
                        metrics["vf"] = self.vf
                        metrics["policy"] = self.policy
                        for k in self.meta_modules.keys():
                            self.meta_modules[k].pre_step(metrics)

                    self.last_state = s
                    s = s_next
                    self.logger.next_step()

                    if steps_since_eval >= eval_every_n_steps:
                        steps_since_eval = 0
                        # TODO: make it work with CARL
                        if isinstance(self.eval_env, DACENV):
                            eval_instance_ids = self.eval_env.instance_id_list
                            vmap(self.eval, in_axes=(None, 0), out_axes=0)(
                                n_episodes_eval, jnp.array(eval_instance_ids)
                            )
                        else:
                            self.eval(n_episodes_eval)

                    if self.steps % human_log_every_n_steps == 0:
                        print(
                            f"Steps: {self.steps}, Reward: {sum(log_reward_buffer) / len(log_reward_buffer)}"
                        )
                        log_reward_buffer = []

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
        self.writer.flush()
        self.writer.close()

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

    def eval(self, episodes: int, instance_id=None):
        """
        Eval agent on an environment. (Full rollouts)
        :param env: The environment to evaluate on
        :param episodes: The number of episodes to evaluate
        :return:
        """
        self.logger.set_eval(True)
        rewards = []
        for _ in range(episodes):
            terminated, truncated = False, False
            # TODO: this doesn't work for CARL, can we change that?
            if instance_id is not None:
                state, _ = self.eval_env.reset(options={"instance_id": instance_id})
            else:
                state, _ = self.eval_env.reset()
            r = 0
            while not (terminated or truncated):
                action = self.policy(state, eval=True)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                r += reward
                self.logger.next_step()
            rewards.append(r)

            if isinstance(self.eval_env, DACENV):
                instance = self.eval_env.instance
            elif isinstance(self.eval_env, CARLENV):
                instance = self.eval_env.context
            else:
                instance = None
            self.logger.next_episode(instance)

        self.logger.write()
        self.logger.set_eval(False)

        eval_metrics = {
            "step": self.steps,
            "eval_episodes": np.array(rewards),
            "mean_eval_reward": np.mean(rewards),
        }
        if instance_id is not None:
            eval_metrics["instance_id"] = instance_id
        if self.writer is not None:
            self.writer.add_scalars("eval", eval_metrics)

        if self.log_wandb:
            wandb.log(eval_metrics)
