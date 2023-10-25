from typing import List, Optional, Type, Union

import coax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from numpy import prod
from omegaconf import DictConfig

from mighty.agent.base_agent import MightyAgent, retrieve_class
from mighty.env.env_handling import MIGHTYENV
from mighty.mighty_exploration import MightyExplorationPolicy
from mighty.utils.logger import Logger
from mighty.utils.types import TypeKwargs


class MightyPPOAgent(MightyAgent):
    """
    Mighty PPO agent

    This agent implements the PPO algorithm first proposed by Schulman et al. in "Proximal Policy Optimization Algorithms" in 2017.
    Like all Mighty agents, it's supposed to be called via the train method.
    Policy and value architectures can be altered by overwriting the policy_function and value_function with a suitable
    haiku/coax architecture.
    """

    def __init__(
        self,
        # MightyAgent Args
        env: MIGHTYENV,
        logger: Logger,
        eval_env: MIGHTYENV = None,
        learning_rate: float = 0.01,  # TODO the default learning rate is pretty high, intended?
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
        policy_class: Optional[
            Union[str, DictConfig, Type[MightyExplorationPolicy]]
        ] = None,
        policy_kwargs: Optional[TypeKwargs] = None,
        meta_methods: Optional[List[Union[str, Type]]] = [],
        meta_kwargs: Optional[list[TypeKwargs]] = [],
        # PPO Specific Args
        n_policy_units: int = 8,
        n_critic_units: int = 8,
        soft_update_weight: float = 1.0,  # TODO which default value?
    ):
        """
        PPO initialization.

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
        :return:
        """

        self.n_policy_units = n_policy_units
        self.n_critic_units = n_critic_units

        assert 0.0 <= soft_update_weight <= 1.0
        self.soft_update_weight = soft_update_weight

        # Placeholder variables which are filled in self.initialize_agent
        self.v: Optional[coax.V] = None
        self.policy: Optional[coax.Policy] = None
        self.v_targ: Optional[coax.V] = None
        self.pi_old: Optional[coax.Policy] = None
        self.td_update: Optional[coax.td_learning.SimpleTD] = None
        self.ppo_clip: Optional[coax.policy_objectives.PPOClip] = None

        self.policy_class = retrieve_class(
            cls=policy_class, default_cls=MightyExplorationPolicy
        )
        if policy_kwargs is None:
            policy_kwargs = {"func": self.policy_function, "env": env}
        self.policy_kwargs = policy_kwargs

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
        """Value function."""
        return self.v

    def policy_function(self, S, is_training):
        """Policy base."""

        shared = hk.Sequential(
            (
                hk.Linear(self.n_policy_units),
                jax.nn.relu,
                hk.Linear(self.n_policy_units),
                jax.nn.relu,
            )
        )
        mu = hk.Sequential(
            (
                shared,
                hk.Linear(self.n_policy_units),
                jax.nn.relu,
                hk.Linear(prod(self.env.action_space.shape), w_init=jnp.zeros),
                hk.Reshape(self.env.action_space.shape),
            )
        )
        logvar = hk.Sequential(
            (
                shared,
                hk.Linear(self.n_policy_units),
                jax.nn.relu,
                hk.Linear(prod(self.env.action_space.shape), w_init=jnp.zeros),
                hk.Reshape(self.env.action_space.shape),
            )
        )
        return {"mu": mu(S), "logvar": logvar(S)}

    def vf(self, S, is_training):
        """value base."""

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
        return seq(S)

    def _initialize_agent(self):
        """Initialize PPO specific components."""

        self.policy = self.policy_class("ppo", **self.policy_kwargs)
        self.v = coax.V(self.vf, self.env)

        # targets
        self.pi_old = self.policy.copy()
        self.v_targ = self.v.copy()

        # update
        self.td_update = coax.td_learning.SimpleTD(
            self.v, self.v_targ, optimizer=optax.adam(0.02)
        )
        self.ppo_clip = coax.policy_objectives.PPOClip(
            self.policy, optimizer=optax.adam(0.01)
        )

        print("Initialized agent.")

    def update_agent(self, step):
        """
        Compute and apply PPO update.

        :param step: Current training step
        :return:
        """

        transition_batch = self.replay_buffer.sample(batch_size=self._batch_size)
        td_metrics, td_error = self.td_update.update(
            transition_batch, return_td_error=True
        )
        td_metrics = {
            f"ValueUpdate/{k.split('/')[-1]}": td_metrics[k] for k in td_metrics.keys()
        }
        pg_metrics = self.ppo_clip.update(transition_batch, td_error)
        pg_metrics = {
            f"PolicyUpdate/{k.split('/')[-1]}": pg_metrics[k] for k in pg_metrics.keys()
        }
        td_metrics.update(pg_metrics)

        # sync target networks
        self.v_targ.soft_update(self.v, tau=self.soft_update_weight)
        self.pi_old.soft_update(self.policy, tau=self.soft_update_weight)
        return td_metrics

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
            metrics["rollout_logits"] = np.empty(0)

        metrics["td_error"] = self.td_update.td_error(transition)
        metrics["rollout_errors"] = np.append(
            metrics["rollout_errors"], self.td_update.td_error(transition)
        )
        metrics["rollout_values"] = np.append(
            metrics["rollout_values"], self.value_function(transition.S)
        )
        _, logprobs = self.policy(transition.S, return_logp=True)
        metrics["rollout_logits"] = np.append(metrics["rollout_logits"], logprobs)
        return metrics

    def get_state(self):
        """
        Return current agent state, e.g. for saving.

        For PPO, this consists of:
        - the value network parameters
        - the value network function state
        - the value target parameters
        - the value target function state
        - the policy network parameters
        - the policy network function state
        - the policy target parameters
        - the policy target function state
        :return: Agent state
        """

        return (
            self.v.params,
            self.v.function_state,
            self.v_targ.params,
            self.v_targ.function_state,
            self.policy.params,
            self.policy.function_state,
            self.pi_old.params,
            self.pi_old.function_state,
        )

    def set_state(self, state):
        """Set the internal state of the agent, e.g. after loading."""

        (
            self.v.params,
            self.v.function_state,
            self.v_targ.params,
            self.v_targ.function_state,
            self.policy.params,
            self.policy.function_state,
            self.pi_old.params,
            self.pi_old.function_state,
        ) = state
