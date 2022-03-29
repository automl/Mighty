from typing import Optional, Union, Type

import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from coax._core.value_based_policy import BaseValueBasedPolicy

from omegaconf import DictConfig

from mighty.agent.base_agent import MightyAgent, retrieve_class
from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger
from mighty.utils.types import TypeKwargs

# DQN with possible extension to DDQN as first proposed in "Playing Atari with Deep Reinforcement Learn#DQN with possible extension to DDQN as first proposed in "Playing Atari with Deep Reinforcement Learning" by Mnih et al. in 2013
# DDQN was proposed by van Hasselt et al. in 2016's "Deep Reinforcement Learning with Double Q-learning"


class DQNAgent(MightyAgent):
    """
    Simple double DQN Agent
    """

    def __init__(
        self,
        # MightyAgent Args
        env: DACENV,
        logger: Logger,
        eval_env: DACENV = None,
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        batch_size: int = 64,
        render_progress: bool = True,
        log_tensorboard: bool = False,
        replay_buffer_class: Optional[
            Union[str, DictConfig, Type[BaseReplayBuffer]]
        ] = None,
        replay_buffer_kwargs: Optional[TypeKwargs] = None,
        tracer_class: Optional[Union[str, DictConfig, Type[BaseRewardTracer]]] = None,
        tracer_kwargs: Optional[TypeKwargs] = None,
        # DDQN Specific Args
        n_units: int = 8,
        soft_update_weight: float = 1.0,  # TODO which default value?
        policy_class: Optional[
            Union[str, DictConfig, Type[BaseValueBasedPolicy]]
        ] = None,
        policy_kwargs: Optional[TypeKwargs] = None,
        td_update_class: Optional[
            Union[
                Type[coax.td_learning.QLearning],
                Type[coax.td_learning.DoubleQLearning],
                Type[coax.td_learning.SoftQLearning],
                Type[coax.td_learning.ClippedDoubleQLearning],
                Type[coax.td_learning.SoftClippedDoubleQLearning],
            ]
        ] = None,
        td_update_kwargs: Optional[TypeKwargs] = None,
    ):
        self.n_units = n_units
        assert 0.0 <= soft_update_weight <= 1.0
        self.soft_update_weight = soft_update_weight

        # Placeholder variables which are filled in self.initialize_agent
        self.q: Optional[coax.Q] = None
        self.policy: Optional[BaseValueBasedPolicy] = None
        self.q_target: Optional[coax.Q] = None
        self.qlearning: Optional[coax.td_learning.DoubleQLearning] = None

        # Policy Class
        policy_class = retrieve_class(cls=policy_class, default_cls=coax.EpsilonGreedy)
        if policy_kwargs is None:
            policy_kwargs = {"epsilon": 0.1}
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs

        self.td_update_class = retrieve_class(
            cls=td_update_class, default_cls=coax.td_learning.DoubleQLearning
        )
        if td_update_kwargs is None:
            td_update_kwargs = {"q_targ": None, "optimizer": optax.adam(learning_rate)}
        self.td_update_kwargs = td_update_kwargs

        super().__init__(
            env=env,
            logger=logger,
            eval_env=eval_env,
            learning_rate=learning_rate,
            epsilon=epsilon,
            batch_size=batch_size,
            render_progress=render_progress,
            log_tensorboard=log_tensorboard,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            tracer_class=tracer_class,
            tracer_kwargs=tracer_kwargs,
        )

    def q_function(self, S, is_training):
        """ Q-function base """
        seq = hk.Sequential(
            (
                hk.Linear(self.n_units),
                jax.nn.relu,
                hk.Linear(self.n_units),
                jax.nn.relu,
                hk.Linear(self.n_units),
                jax.nn.relu,
                hk.Linear(
                    self.env.action_space.n, w_init=jnp.zeros
                ),  # TODO check if this spec is needed. haiku automatically determines sizes
            )
        )
        return seq(S)

    def _initialize_agent(self):
        """ Initialize DQN specific things like q-function """

        self.q = coax.Q(self.q_function, self.env)
        self.policy = self.policy_class(q=self.q, **self.policy_kwargs)

        # target network
        self.q_target = self.q.copy()

        # specify how to update value function
        self.qlearning = self.td_update_class(self.q, **self.td_update_kwargs)

        print("Initialized agent.")

    def update_agent(self, step):
        transition_batch = self.replay_buffer.sample(batch_size=self._batch_size)
        metrics_q = self.qlearning.update(transition_batch)
        # TODO: log these properly
        # env.record_metrics(metrics_q)

        # periodically sync target models
        if step % 10 == 0:
            self.q_target.soft_update(self.q, tau=self.soft_update_weight)

    def get_state(self):
        return (
            self.q.params,
            self.q.function_state,
            self.q_target.params,
            self.q_target.function_state,
        )

    def set_state(self, state):
        (
            self.q.params,
            self.q.function_state,
            self.q_target.params,
            self.q_target.function_state,
        ) = state

    def eval(self, env: DACENV, episodes: int):
        """
        Eval agent on an environment. (Full evaluation)
        :param env:
        :param episodes:
        :return:
        """
        raise NotImplementedError
