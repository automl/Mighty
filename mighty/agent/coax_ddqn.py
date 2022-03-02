import os
import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp

from mighty.agent.coax_agent import MightyAgent
from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger


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
    ):
        self.n_units = n_units
        super().__init__(env, logger, eval_env, learning_rate, epsilon, batch_size, render_progress, log_tensorboard)

    def initialize_agent(self):

        def func_q(S, is_training):
            """ type-2 q-function: s -> q(s,.) """
            seq = hk.Sequential((
                hk.Linear(self.n_units), jax.nn.relu,
                hk.Linear(self.n_units), jax.nn.relu,
                hk.Linear(self.n_units), jax.nn.relu,
                hk.Linear(self.env.action_space.n, w_init=jnp.zeros)
            ))
            return seq(S)

        self.q = coax.Q(func_q, self.env)
        self.policy = coax.EpsilonGreedy(self.q, epsilon=self._epsilon)

        # target network
        self.q_target = self.q.copy()

        # specify how to update value function
        self.qlearning = coax.td_learning.DoubleQLearning(self.q, q_targ=self.q_target, optimizer=optax.adam(self.learning_rate))

        # specify how to trace the transitions
        self.tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)
        print("Initialized agent.")

    def update_agent(self, step):
        transition_batch = self.buffer.sample(batch_size=self._batch_size)
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
        path = os.path.join(self.model_dir, 'checkpoint.pkl.lz4')
        #For some reason there's an error here to do with pickle. Pickling this outside of the class works, though.
        #coax.utils.dump((self.q, self.q_target, self.qlearning), path)

