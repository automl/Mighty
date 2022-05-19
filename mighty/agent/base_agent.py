from pathlib import Path
import os
from typing import Optional, Dict, Any, Union, Tuple, Type
import hydra
from omegaconf import DictConfig

import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp
from coax.experience_replay._simple import BaseReplayBuffer
from coax.experience_replay import SimpleReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from typing import Optional
from rich.progress import Progress, TimeRemainingColumn, TimeElapsedColumn, BarColumn

from mighty.env.env_handling import MIGHTYENV, DACENV, CARLENV
from mighty.utils.logger import Logger
from mighty.utils.types import TypeKwargs


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
        render_progress: bool = True, #FIXME: Does this actually do anything or can we take it out?
        log_tensorboard: bool = False,
        replay_buffer_class: Optional[
            Union[str, DictConfig, Type[BaseReplayBuffer]]
        ] = None,
        replay_buffer_kwargs: Optional[TypeKwargs] = None,
        tracer_class: Optional[Union[str, DictConfig, Type[BaseRewardTracer]]] = None,
        tracer_kwargs: Optional[TypeKwargs] = None,
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
            cls=replay_buffer_class, default_cls=SimpleReplayBuffer
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

        self.logger = logger
        self.render_progress = render_progress
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.model_dir = os.path.join(self.output_dir, "models")

        self.last_state = None
        self.total_steps = 0

        self.writer = None
        if log_tensorboard and output_dir is not None:
            self.writer = SummaryWriter(output_dir)
            self.writer.add_scalar("hyperparameter/learning_rate", self.learning_rate)
            self.writer.add_scalar("hyperparameter/batch_size", self._batch_size)
            self.writer.add_scalar("hyperparameter/policy_epsilon", self._epsilon)

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
        human_log_every_n_episodes: int = 100,
        save_model_every_n_episodes: int = 100,
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
            steps = 0
            steps_since_eval = 0
            log_reward_buffer = []
            while steps < n_steps:
                progress.update(steps_task, visible=True)
                s = self.env.reset()
                done = False
                while not done:
                    a = self.policy(s)
                    s_next, r, done, info = self.env.step(a)

                    self.logger.log("reward", r)
                    self.logger.log("action", a)
                    self.logger.log("next_state", s_next)
                    self.logger.log("state", s)
                    self.logger.log("done", done)

                    log_reward_buffer.append(r)
                    steps += 1
                    steps_since_eval += 1
                    progress.advance(steps_task)

                    # add transition to buffer
                    self.tracer.add(s, a, r, done)
                    while self.tracer:
                        if isinstance(self.replay_buffer, coax.experience_replay.PrioritizedReplayBuffer):
                            transition = self.tracer.pop()
                            self.replay_buffer.add(transition, self.qlearning.td_error(transition))
                        else:
                            self.replay_buffer.add(self.tracer.pop())

                    # update
                    if len(self.replay_buffer) >= self._batch_size:
                        self.update_agent(steps)

                    self.last_state = s
                    s = s_next
                    self.logger.next_step()

                if isinstance(self.env, DACENV):
                    instance = self.env.instance
                elif isinstance(self.env, CARLENV):
                    instance = self.env.context
                else:
                    instance = None
                self.logger.next_episode(instance)
                episodes += 1

                if steps_since_eval >= eval_every_n_steps:
                    steps_since_eval = 0
                    self.eval(self.eval_env, n_episodes_eval)

                if episodes % human_log_every_n_episodes == 0:
                    print(
                        f"Steps: {steps}, Reward: {sum(log_reward_buffer) / len(log_reward_buffer)}"
                    )
                    log_reward_buffer = []

                if episodes % save_model_every_n_episodes == 0:
                    self.save(episodes)

        # At the end make sure logger writes buffer to file
        self.logger.write()

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
        logdir = os.getcwd()
        filepath = Path(os.path.join(logdir, "checkpoints", f"checkpoint_{T}.pkl.lz4"))
        if not filepath.is_file() or True:  # TODO build logic
            state = self.get_state()
            coax.utils.dump(state, str(filepath))
            print(f"Saved checkpoint to {filepath}")

    def eval(self, env: MIGHTYENV, episodes: int):
        """
        Eval agent on an environment. (Full rollouts)
        :param env: The environment to evaluate on
        :param episodes: The number of episodes to evaluate
        :return:
        """
        self.logger.set_eval(True)
        for e in range(episodes):
            done = False
            state = env.reset()
            while not done:
                action = self.policy(state)
                state, reward, done, _ = env.step(action)
                self.logger.next_step()

            if isinstance(self.env, DACENV):
                instance = self.env.instance
            elif isinstance(self.env, CARLENV):
                instance = self.env.context
            else:
                instance = None
            self.logger.next_episode(instance)

        self.logger.write()
        self.logger.set_eval(False)
