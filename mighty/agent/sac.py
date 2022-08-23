from typing import Optional, Union, Type

import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from coax.reward_tracing import NStep
from numpy import prod

from omegaconf import DictConfig

from mighty.agent.base_agent import MightyAgent, retrieve_class
from mighty.env.env_handling import MIGHTYENV
from mighty.utils.logger import Logger
from mighty.utils.types import TypeKwargs


class MightySACAgent(MightyAgent):
    """
    Mighty SAC agent

    This agent implements the SAC algorithm from Haarnoja et al.'s "Soft Actor-Critic: Off-Policy
    Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" Paper at ICML 2018.
    Like all Mighty agents, it's supposed to be called via the train method.
    Policy and Q-function architectures can be altered by overwriting the policy_function and q_function with a suitable
    haiku/coax architecture.
    """
    def __init__(
        self,
        # MightyAgent Args
        env: MIGHTYENV,
        logger: Logger,
        eval_env: MIGHTYENV = None,
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        batch_size: int = 64,
        render_progress: bool = True,
        log_tensorboard: bool = False,
        replay_buffer_class: Optional[
            Union[str, DictConfig, Type[BaseReplayBuffer]]
        ] = None,
        replay_buffer_kwargs: Optional[TypeKwargs] = None,
        tracer_class: Optional[Union[str, DictConfig, Type[BaseRewardTracer]]] = NStep,
        tracer_kwargs: Optional[TypeKwargs] = {"record_extra_info": True, "n": 5},
        # SAC Specific Args
        n_policy_units: int = 8,
        n_critic_units: int = 8,
        soft_update_weight: float = 1.0,  # TODO which default value?
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
        """
        SAC initialization
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
        :param n_policy_units: Number of units for policy network
        :param n_critic_units: Number of units for critic network
        :param soft_update_weight: Size of soft updates for target network
        :param td_update_class: Kind of TD update used from coax TD updates
        :param td_update_kwargs: Arguments for the TD update
        :return:
        """

        assert 0.0 <= soft_update_weight <= 1.0
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
        self.policy_regularizer: Optional[
            coax.regularizers.NStepEntropyRegularizer
        ] = None

        self.td_update_class = retrieve_class(
            cls=td_update_class, default_cls=coax.td_learning.DoubleQLearning
        )
        if td_update_kwargs is None:
            td_update_kwargs = {"q_targ": None, "optimizer": optax.adam(learning_rate)}
        self.td_update_kwargs = td_update_kwargs

        tracer_kwargs["gamma"] = 0.9

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

    def policy_function(self, S, is_training):
        """Policy base function"""
        seq = hk.Sequential(
            (
                hk.Linear(self.n_policy_units),
                jax.nn.relu,
                hk.Linear(self.n_policy_units),
                jax.nn.relu,
                hk.Linear(self.n_policy_units),
                jax.nn.relu,
                hk.Linear(prod(self.env.action_space.shape) * 2, w_init=jnp.zeros),
                hk.Reshape((*self.env.action_space.shape, 2)),
            )
        )
        x = seq(S)
        mu, logvar = x[..., 0], x[..., 1]
        return {"mu": mu, "logvar": logvar}

    def q_function(self, S, A, is_training):
        """Q-function base for critic"""
        seq = hk.Sequential(
            (
                hk.Linear(self.n_critic_units),
                jax.nn.relu,
                hk.Linear(self.n_critic_units),
                jax.nn.relu,
                hk.Linear(self.n_critic_units),
                jax.nn.relu,
                hk.Linear(1, w_init=jnp.zeros),
                jnp.ravel,
            )
        )
        X = jnp.concatenate((S, A), axis=-1)
        return seq(X)

    def _initialize_agent(self):
        """Initialize algorithm components like policy and critic"""
        # main function approximators
        self.policy = coax.Policy(self.policy_function, self.env)
        self.q1 = coax.Q(
            self.q_function,
            self.env,
            action_preprocessor=self.policy.proba_dist.preprocess_variate,
        )
        self.q2 = coax.Q(
            self.q_function,
            self.env,
            action_preprocessor=self.policy.proba_dist.preprocess_variate,
        )

        # target network
        self.q1_target = self.q1.copy()
        self.q2_target = self.q2.copy()

        # regularizer
        alpha = 0.2
        self.policy_regularizer = coax.regularizers.NStepEntropyRegularizer(
            self.policy,
            beta=alpha / self.tracer.n,
            gamma=self.tracer.gamma,
            n=[self.tracer.n],
        )

        # updaters (use current pi to update the q-functions and use sampled action in contrast to TD3)
        self.qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
            self.q1,
            pi_targ_list=[self.policy],
            q_targ_list=[self.q1_target, self.q2_target],
            loss_function=coax.value_losses.mse,
            optimizer=optax.adam(self.learning_rate),
            policy_regularizer=self.policy_regularizer,
        )
        self.qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
            self.q2,
            pi_targ_list=[self.policy],
            q_targ_list=[self.q1_target, self.q2_target],
            loss_function=coax.value_losses.mse,
            optimizer=optax.adam(self.learning_rate),
            policy_regularizer=self.policy_regularizer,
        )
        self.soft_pg = coax.policy_objectives.SoftPG(
            self.policy,
            [self.q1_target, self.q2_target],
            optimizer=optax.adam(1e-3),
            regularizer=coax.regularizers.NStepEntropyRegularizer(
                self.policy,
                beta=alpha / self.tracer.n,
                gamma=self.tracer.gamma,
                n=jnp.arange(self.tracer.n),
            ),
        )
        print("Initialized agent.")

    def update_agent(self, step):
        """
        Compute and apply SAC update
        :param step: Current training step
        :return:
        """
        transition_batch = self.replay_buffer.sample(batch_size=self._batch_size)
        metrics = {}
        # flip a coin to decide which of the q-functions to update
        qlearning = (
            self.qlearning1 if jax.random.bernoulli(self.q1.rng) else self.qlearning2
        )
        metrics.update(qlearning.update(transition_batch))
        metrics.update(self.soft_pg.update(transition_batch))

        # TODO: log metrics
        # env.record_metrics(metrics)

        # sync target networks
        self.q1_target.soft_update(self.q1, tau=self.soft_update_weight)
        self.q2_target.soft_update(self.q2, tau=self.soft_update_weight)

    def get_state(self):
        """
        Return current agent state, e.g. for saving.
        For SAC, this consists of:
        - the policy action probability distribution
        - the policy function state
        - the first Q network's parameters
        - the first Q network's function state
        - the second Q network's parameters
        - the second Q network's function state
        - the first target network's parameters
        - the first target network's function state
        - the second target network's parameters
        - the second target network's function state
        :return: Agent state
        """
        return (
            self.policy.proba_dist,
            self.policy.function_state,
            self.q1.params,
            self.q1.function_state,
            self.q2.params,
            self.q2.function_state,
            self.q1_target.params,
            self.q1_target.function_state,
            self.q2_target.params,
            self.q2_target.function_state,
        )

    def set_state(self, state):
        """Set the internal state of the agent, e.g. after loading"""
        (
            self.policy.proba_dist,
            self.policy.function_state,
            self.q1.params,
            self.q1.function_state,
            self.q2.params,
            self.q2.function_state,
            self.q1_target.params,
            self.q1_target.function_state,
            self.q2_target.params,
            self.q2_target.function_state,
        ) = state
