from typing import Optional, Union, Type

import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from numpy import prod

from omegaconf import DictConfig

from mighty.agent.base_agent import MightyAgent, retrieve_class
from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger
from mighty.utils.types import TKwargs


class SACAgent(MightyAgent):
    """
    SAC Agent
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
            replay_buffer_class: Optional[Union[str, DictConfig, Type[BaseReplayBuffer]]] = None,
            replay_buffer_kwargs: Optional[TKwargs] = None,
            tracer_class: Optional[Union[str, DictConfig, Type[BaseRewardTracer]]] = None,
            tracer_kwargs: Optional[TKwargs] = None,
            # SAC Specific Args
            n_policy_units: int = 8,
            n_critic_units: int = 8,
            soft_update_weight: float = 1.,  # TODO which default value?
            td_update_class: Optional[Union[Type[coax.td_learning.QLearning],
                                            Type[coax.td_learning.DoubleQLearning],
                                            Type[coax.td_learning.SoftQLearning],
                                            Type[coax.td_learning.ClippedDoubleQLearning],
                                            Type[coax.td_learning.SoftClippedDoubleQLearning]]] = None,
            td_update_kwargs: Optional[TKwargs] = None
    ):
        assert 0. <= soft_update_weight <= 1.
        self.soft_update_weight = soft_update_weight
        self.n_policy_units = n_policy_units
        self.n_critic_units = n_critic_units

        # Placeholder variables which are filled in self.initialize_agent
        self.q1: Optional[coax.Q] = None
        self.q2: Optional[coax.Q] = None
        self.q1_target: Optional[coax.Q] = None
        self.q2_target: Optional[coax.Q] = None
        self.qlearning1: Optional[coax.td_learning.DoubleQLearning] = None
        self.qlearning2: Optional[coax.td_learning.DoubleQLearning] = None
        self.soft_pg: Optional[coax.policy_objectives.SoftPG] = None
        self.buffer: Optional[coax.experience_replay.SimpleReplayBuffer] = None
        self.policy_regularizer: Optional[coax.regularizers.NStepEntropyRegularizer] = None

        self.td_update_class = retrieve_class(cls=td_update_class, default_cls=coax.td_learning.DoubleQLearning)
        if td_update_kwargs is None:
            td_update_kwargs = {
                "q_targ": None,
                "optimizer": optax.adam(learning_rate)
            }
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

    def _initialize_agent(self):
        def func_pi(S, is_training):
            seq = hk.Sequential((
                hk.Linear(self.n_policy_units), jax.nn.relu,
                hk.Linear(self.n_policy_units), jax.nn.relu,
                hk.Linear(self.n_policy_units), jax.nn.relu,
                hk.Linear(prod(self.env.action_space.shape) * 2, w_init=jnp.zeros),
                hk.Reshape((*self.env.action_space.shape, 2)),
            ))
            x = seq(S)
            mu, logvar = x[..., 0], x[..., 1]
            return {'mu': mu, 'logvar': logvar}

        def func_q(S, A, is_training):
            seq = hk.Sequential((
                hk.Linear(self.n_critic_units), jax.nn.relu,
                hk.Linear(self.n_critic_units), jax.nn.relu,
                hk.Linear(self.n_critic_units), jax.nn.relu,
                hk.Linear(1, w_init=jnp.zeros), jnp.ravel
            ))
            X = jnp.concatenate((S, A), axis=-1)
            return seq(X)

        # main function approximators
        self.policy = coax.Policy(func_pi, self.env)
        self.q1 = coax.Q(func_q, self.env, action_preprocessor=self.policy.proba_dist.preprocess_variate)
        self.q2 = coax.Q(func_q, self.env, action_preprocessor=self.policy.proba_dist.preprocess_variate)

        # target network
        self.q1_target = self.q1.copy()
        self.q2_target = self.q2.copy()

        # regularizer
        alpha = 0.2
        self.policy_regularizer = coax.regularizers.NStepEntropyRegularizer(self.policy,
                                                                       beta=alpha / self.tracer.n,
                                                                       gamma=self.tracer.gamma,
                                                                       n=[self.tracer.n])

        # updaters (use current pi to update the q-functions and use sampled action in contrast to TD3)
        self.qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
            self.q1,
            pi_targ_list=[self.policy],
            q_targ_list=[self.q1_target, self.q2_target],
            loss_function=coax.value_losses.mse,
            optimizer=optax.adam(self.learning_rate),
            policy_regularizer=self.policy_regularizer)
        self.qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
            self.q2,
             pi_targ_list=[self.policy],
              q_targ_list=[self.q1_target, self.q2_target],
            loss_function=coax.value_losses.mse,
             optimizer=optax.adam(self.learning_rate),
            policy_regularizer=self.policy_regularizer)
        self.soft_pg = coax.policy_objectives.SoftPG(self.policy, [self.q1_target, self.q2_target], optimizer=optax.adam(
            1e-3), regularizer=coax.regularizers.NStepEntropyRegularizer(self.policy,
                                                                         beta=alpha / self.tracer.n,
                                                                         gamma=self.tracer.gamma,
                                                                         n=jnp.arange(self.tracer.n)))
        print("Initialized agent.")

    def update_agent(self, step):
        transition_batch = self.replay_buffer.sample(batch_size=self._batch_size)
        metrics = {}
        # flip a coin to decide which of the q-functions to update
        qlearning = self.qlearning1 if jax.random.bernoulli(self.q1.rng) else self.qlearning2
        metrics.update(qlearning.update(transition_batch))
        metrics.update(self.soft_pg.update(transition_batch))

        #TODO: log metrics
        #env.record_metrics(metrics)

        # sync target networks
        self.q1_target.soft_update(self.q1, tau=self.soft_update_weight)
        self.q2_target.soft_update(self.q2, tau=self.soft_update_weight)

    def get_state(self):
        return self.policy.proba_dist, self.policy.function_state,\
               self.q1.params, self.q1.function_state, \
               self.q2.params, self.q2.function_state, \
               self.q1_target.params, self.q1_target.function_state, \
               self.q2_target.params, self.q2_target.function_state,

    def set_state(self, state):
        self.policy.proba_dist, self.policy.function_state, \
            self.q1.params, self.q1.function_state, \
            self.q2.params, self.q2.function_state, \
            self.q1_target.params, self.q1_target.function_state, \
            self.q2_target.params, self.q2_target.function_state, = state

    def eval(self, env: DACENV, episodes: int):
        """
        Eval agent on an environment. (Full evaluation)
        :param env:
        :param episodes:
        :return:
        """
        raise NotImplementedError
