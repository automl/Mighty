from typing import Optional, Union, Type, List

import jax
import coax
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer

from omegaconf import DictConfig

from mighty.agent.base_agent import MightyAgent, retrieve_class
from mighty.env.env_handling import MIGHTYENV
from mighty.utils.logger import Logger
from mighty.utils.types import TypeKwargs
from mighty.mighty_exploration import MightyExplorationPolicy, EpsilonGreedy


class MightyDQNAgent(MightyAgent):
    """
    Mighty DQN agent

    This agent implements the DQN algorithm and extension as first proposed in "Playing Atari with
    Deep Reinforcement Learning" by Mnih et al. in 2013.
    DDQN was proposed by van Hasselt et al. in 2016's "Deep Reinforcement Learning with Double Q-learning".
    Like all Mighty agents, it's supposed to be called via the train method.
    The Q-function architecture can be altered by overwriting the q_function with a suitable haiku/coax architecture.
    By default, this agent uses an epsilon-greedy policy.
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
        # DDQN Specific Args
        n_units: int = 8,
        soft_update_weight: float = 1.0,  # TODO which default value?
        policy_class: Optional[
            Union[str, DictConfig, Type[MightyExplorationPolicy]]
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
        """
        DQN initialization.

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
        :param n_units: Number of units for Q network
        :param soft_update_weight: Size of soft updates for target network
        :param policy_class: Policy class from coax value-based policies
        :param policy_kwargs: Arguments for the policy
        :param td_update_class: Kind of TD update used from coax TD updates
        :param td_update_kwargs: Arguments for the TD update
        :return:
        """

        self.n_units = n_units
        assert 0.0 <= soft_update_weight <= 1.0
        self.soft_update_weight = soft_update_weight

        # Placeholder variables which are filled in self.initialize_agent
        self.q: Optional[coax.Q] = None
        self.policy: Optional[MightyExplorationPolicy] = None
        self.q_target: Optional[coax.Q] = None
        self.qlearning: Optional[coax.td_learning.DoubleQLearning] = None

        # Policy Class
        policy_class = retrieve_class(cls=policy_class, default_cls=EpsilonGreedy)
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
            learning_starts=learning_starts,
            render_progress=render_progress,
            log_tensorboard=log_tensorboard,
            log_wandb=log_wandb,
            wandb_kwargs=wandb_kwargs,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            tracer_class=tracer_class,
            tracer_kwargs=tracer_kwargs,
            meta_methods=meta_methods,
            meta_kwargs=meta_kwargs,
        )

    @property
    def value_function(self):
        """Q-function."""

        return self.q

    def q_function(self, S, is_training):
        """Q-function base"""

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
        """Initialize DQN specific things like q-function"""

        self.q = coax.Q(self.q_function, self.env)
        self.policy = self.policy_class(algo="q", func=self.q, **self.policy_kwargs)

        # target network
        self.q_target = self.q.copy()

        # specify how to update value function
        self.qlearning = self.td_update_class(self.q, **self.td_update_kwargs)

        print("Initialized agent.")

    def update_agent(self, step):
        """
        Compute and apply TD update.

        :param step: Current training step
        :return:
        """

        transition_batch = self.replay_buffer.sample(batch_size=self._batch_size)
        metrics_q = self.qlearning.update(transition_batch)
        metrics_q = {
            f"Q-Update/{k.split('/')[-1]}": metrics_q[k] for k in metrics_q.keys()
        }

        # periodically sync target models
        if step % 10 == 0:
            self.q_target.soft_update(self.q, tau=self.soft_update_weight)
        return metrics_q

    def get_transition_metrics(self, transition, metrics):
        """
        Get metrics per transition.

        :param transition: Current transition
        :param metrics: Current metrics dict
        :return:
        """

        if "rollout_errors" not in metrics.keys():
            metrics["rollout_errors"] = np.empty(0)
            metrics["rollout_values"] = np.empty(0)

        metrics["td_error"] = self.qlearning.td_error(transition)
        metrics["rollout_errors"] = np.append(
            metrics["rollout_errors"], self.qlearning.td_error(transition)
        )
        metrics["rollout_values"] = np.append(
            metrics["rollout_values"], self.value_function(transition.S)
        )
        return metrics

    def get_state(self):
        """
        Return current agent state, e.g. for saving.

        For DQN, this consists of:
        - the Q network parameters
        - the Q network function state
        - the target network parameters
        - the target network function state

        :return: Agent state
        """

        return (
            self.q.params,
            self.q.function_state,
            self.q_target.params,
            self.q_target.function_state,
        )

    def set_state(self, state):
        """Set the internal state of the agent, e.g. after loading."""

        (
            self.q.params,
            self.q.function_state,
            self.q_target.params,
            self.q_target.function_state,
        ) = state
