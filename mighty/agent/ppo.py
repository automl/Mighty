from typing import Optional, Union, Type

import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from coax._core.value_based_policy import BaseValueBasedPolicy
from numpy import prod

from omegaconf import DictConfig

from mighty.agent.base_agent import MightyAgent
from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger
from mighty.utils.types import TKwargs


class DDQNAgent(MightyAgent):
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
            replay_buffer_class: Optional[Union[str, DictConfig, Type[BaseReplayBuffer]]] = None,
            replay_buffer_kwargs: Optional[TKwargs] = None,
            tracer_class: Optional[Union[str, DictConfig, Type[BaseRewardTracer]]] = None,
            tracer_kwargs: Optional[TKwargs] = None,
            # PPO Specific Args
            n_policy_units: int = 8,
            n_critic_units: int = 8,
            soft_update_weight: float = 1.,  # TODO which default value?
    ):
        self.n_policy_units = n_policy_units
        self.n_critic_units = n_critic_units

        assert 0. <= soft_update_weight <= 1.
        self.soft_update_weight = soft_update_weight

        # Placeholder variables which are filled in self.initialize_agent
        self.q: Optional[coax.Q] = None
        self.policy: Optional[BaseValueBasedPolicy] = None
        self.q_target: Optional[coax.Q] = None
        self.qlearning: Optional[coax.td_learning.DoubleQLearning] = None

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
            shared = hk.Sequential((
                hk.Linear(self.n_policy_units), jax.nn.relu,
                hk.Linear(self.n_policy_units), jax.nn.relu,
            ))
            mu = hk.Sequential((
                shared,
                hk.Linear(self.n_policy_units), jax.nn.relu,
                hk.Linear(prod(self.env.action_space.shape), w_init=jnp.zeros),
                hk.Reshape(self.env.action_space.shape),
            ))
            logvar = hk.Sequential((
                shared,
                hk.Linear(self.n_policy_units), jax.nn.relu,
                hk.Linear(prod(self.env.action_space.shape), w_init=jnp.zeros),
                hk.Reshape(self.env.action_space.shape),
            ))
            return {'mu': mu(S), 'logvar': logvar(S)}

        def func_v(S, is_training):
            seq = hk.Sequential((
                hk.Linear(self.n_critic_units), jax.nn.relu,
                hk.Linear(self.n_critic_units), jax.nn.relu,
                hk.Linear(self.n_critic_units), jax.nn.relu,
                hk.Linear(1, w_init=jnp.zeros), jnp.ravel
            ))
            return seq(S)

        self.pi = coax.Policy(func_pi, self.env)
        self.v = coax.V(func_v, self.env)

        # targets
        self.pi_old = self.pi.copy()
        self.v_targ = self.v.copy()

        # update
        self.td_update = coax.td_learning.SimpleTD(self.v, self.v_targ, optimizer=optax.adam(0.02))
        self.ppo_clip = coax.policy_objectives.PPOClip(self.pi, optimizer=optax.adam(0.01))

        print("Initialized agent.")

    def update_agent(self, step):
        transition_batch = self.tracer.pop()
        _, td_error = self.td_update.update(transition_batch, return_td_error=True)
        self.ppo_clip.update(transition_batch, td_error)

        # sync target networks
        self.v_targ.soft_update(self.v, tau=self.soft_update_weight)
        self.pi_old.soft_update(self.pi, tau=self.soft_update_weight)

    def get_state(self):
        return self.v.params, self.v.function_state, \
                self.v_targ.params, self.v_targ.function_state, \
                self.pi.params, self.pi.function_state, \
                self.pi_old.params, self.pi_old.function_state

    def set_state(self, state):
        self.v.params, self.v.function_state, \
            self.v_targ.params, self.v_targ.function_state, \
            self.pi.params, self.pi.function_state, \
            self.pi_old.params, self.pi_old.function_state = state

    def eval(self, env: DACENV, episodes: int):
        """
        Eval agent on an environment. (Full evaluation)
        :param env:
        :param episodes:
        :return:
        """
        raise NotImplementedError






