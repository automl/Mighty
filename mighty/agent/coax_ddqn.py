import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, Type

import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp
from coax.experience_replay._simple import BaseReplayBuffer
from coax.reward_tracing._base import BaseRewardTracer
from coax.experience_replay import SimpleReplayBuffer
from coax._core.value_based_policy import BaseValueBasedPolicy

import hydra
from omegaconf import DictConfig

from mighty.agent.coax_agent import MightyAgent
from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger


def retrieve_class(cls: Union[str, DictConfig, Type], default_cls: Type) -> Type:
    if cls is None:
        cls = default_cls
    elif type(cls) == DictConfig:
        cls = hydra.utils.get_class(cls._target_)
    elif type(cls) == str:
        cls = hydra.utils.get_class(cls)
    return cls


class DDQNAgent(MightyAgent):
    """
    Simple double DQN Agent
    """

    def __init__(
            self,
            env: DACENV,
            logger: Logger,
            eval_env: DACENV = None,
            learning_rate: float = 0.01,
            epsilon: float = 0.1,
            batch_size: int = 64,
            render_progress: bool = True,
            log_tensorboard: bool = False,
            n_units: int = 8,
            replay_buffer_class: Optional[Union[str, DictConfig, Type[BaseReplayBuffer]]] = None,
            replay_buffer_kwargs: Optional[Union[Dict[str, Any], DictConfig]] = None,
            tracer_class: Optional[Union[str, DictConfig, Type[BaseRewardTracer]]] = None,
            tracer_kwargs: Optional[Union[Dict[str, Any], DictConfig]] = None,
    ):
        self.n_units = n_units

        # Replay Buffer
        replay_buffer_class = retrieve_class(cls=replay_buffer_class, default_cls=SimpleReplayBuffer)
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {
                "capacity": 1_000_000,
                "random_seed": None,
            }
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs

        # Reward Tracer
        # TODO create dac tracer receiving instance as additional info
        tracer_class = retrieve_class(cls=tracer_class, default_cls=coax.reward_tracing.NStep)
        if tracer_kwargs is None:
            tracer_kwargs = {
                "n": 1,
                "gamma": 0.9,
            }
        self.tracer_class = tracer_class
        self.tracer_kwargs = tracer_kwargs

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
            log_tensorboard=log_tensorboard
        )

    def initialize_agent(self):

        def func_q(S, is_training):
            """ type-2 q-function: s -> q(s,.) """
            seq = hk.Sequential((
                hk.Linear(self.n_units), jax.nn.relu,
                hk.Linear(self.n_units), jax.nn.relu,
                hk.Linear(self.n_units), jax.nn.relu,
                hk.Linear(self.env.action_space.n, w_init=jnp.zeros)  # TODO check if this spec is needed. haiku automatically determines sizes
            ))
            return seq(S)

        self.q = coax.Q(func_q, self.env)
        self.policy = coax.EpsilonGreedy(self.q, epsilon=self._epsilon)

        # target network
        self.q_target = self.q.copy()

        # specify how to update value function
        self.qlearning = coax.td_learning.DoubleQLearning(self.q, q_targ=self.q_target, optimizer=optax.adam(self.learning_rate))

        # specify how to trace the transitions
        self.tracer = self.tracer_class(**self.tracer_kwargs)
        self.replay_buffer = self.replay_buffer_class(**self.replay_buffer_kwargs)
        print("Initialized agent.")

    def update_agent(self, step):
        transition_batch = self.replay_buffer.sample(batch_size=self._batch_size)
        metrics_q = self.qlearning.update(transition_batch)
        # TODO: log these properly
        # env.record_metrics(metrics_q)

        # periodically sync target models
        if step % 10 == 0:
            self.q_target.soft_update(self.q, tau=1.0)

    def load(self, path):
        """ Load checkpointed model. """
        self.q, self.q_target, self.qlearning = coax.utils.load(path)

    def save(self):
        """ Checkpoint model. """
        # path = os.path.join(self.model_dir, 'checkpoint.pkl.lz4')
        # #For some reason there's an error here to do with pickle. Pickling this outside of the class works, though.
        # #coax.utils.dump((self.q, self.q_target, self.qlearning), path)

        logdir = os.getcwd()
        T = 0  # TODO get automatic checkpoint IDs
        filepath = Path(os.path.join(logdir, "checkpoints", f"checkpoint_{T}.pkl.lz4"))
        if not filepath.is_file() or True:  # TODO build logic
            state = self.get_state()
            coax.utils.dump(state, str(filepath))
            print(f"Saved checkpoint to {filepath}")

    def get_state(self):
        return self.q.params, self.q.function_state, self.q_target.params, self.q_target.function_state

    def set_state(self, state):
        self.q.params, self.q.function_state, self.q_target.params, self.q_target.function_state = state







