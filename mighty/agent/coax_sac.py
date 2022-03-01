import os
import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp
from numpy import prod

from mighty.agent.coax_agent import MightyAgent
from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger


class SACAgent(MightyAgent):
    """
    SAC Agent
    """

    def __init__(
            self,
            env: DACENV,
            logger: Logger,
            eval_env: DACENV = None,
            lr: float = 0.01,
            epsilon: float = 0.1,
            batch_size: int = 64,
            render_progress: bool = True,
            log_tensorboard: bool = False,
            n_policy_units: int = 8,
            n_policy_layers: int = 3,
            n_critic_units: int = 8,
            n_critic_layers: int = 3,
    ):
        self.n_policy_units = n_policy_units
        self.n_policy_layers = n_policy_layers
        self.n_critic_units = n_critic_units
        self.n_critic_layers = n_critic_layers
        super().__init__(env, logger, eval_env, lr, epsilon, batch_size, render_progress, log_tensorboard)

    def initialize_agent(self):
        def func_pi(S, is_training):
            layers = [hk.Linear(self.n_critic_units), jax.nn.relu] * self.n_critic_layers
            layers.append(hk.Linear(prod(self.env.action_space.shape) * 2, w_init=jnp.zeros))
            layers.append(hk.Reshape((*self.env.action_space.shape, 2)))
            seq = hk.Sequential(tuple(layers))
            x = seq(S)
            mu, logvar = x[..., 0], x[..., 1]
            return {'mu': mu, 'logvar': logvar}

        def func_q(S, A, is_training):
            layers = [hk.Linear(self.n_critic_units), jax.nn.relu] * self.n_critic_layers
            layers.append(hk.Linear(1, w_init=jnp.zeros))
            layers.append(jnp.ravel)
            seq = hk.Sequential(tuple(layers))
            X = jnp.concatenate((S, A), axis=-1)
            return seq(X)

        # main function approximators
        self.policy = coax.Policy(func_pi, self.env)
        self.q1 = coax.Q(func_q, self.env, action_preprocessor=self.policy.proba_dist.preprocess_variate)
        self.q2 = coax.Q(func_q, self.env, action_preprocessor=self.policy.proba_dist.preprocess_variate)

        # target network
        self.q1_target = self.q1.copy()
        self.q2_target = self.q2.copy()

        # experience tracer
        self.tracer = coax.reward_tracing.NStep(n=5, gamma=0.9, record_extra_info=True)
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=25000)
        alpha = 0.2
        self.policy_regularizer = coax.regularizers.NStepEntropyRegularizer(self.policy,
                                                                       beta=alpha / self.tracer.n,
                                                                       gamma=self.tracer.gamma,
                                                                       n=[self.tracer.n])

        # updaters (use current pi to update the q-functions and use sampled action in contrast to TD3)
        self.qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
            q1, pi_targ_list=[self.policy], q_targ_list=[self.q1_target, self.q2_target],
            loss_function=coax.value_losses.mse, optimizer=optax.adam(1e-3),
            policy_regularizer=self.policy_regularizer)
        self.qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
            q2, pi_targ_list=[self.policy], q_targ_list=[self.q1_target, self.q2_target],
            loss_function=coax.value_losses.mse, optimizer=optax.adam(1e-3),
            policy_regularizer=self.policy_regularizer)
        self.soft_pg = coax.policy_objectives.SoftPG(self.policy, [self.q1_target, self.q2_target], optimizer=optax.adam(
            1e-3), regularizer=coax.regularizers.NStepEntropyRegularizer(pi,
                                                                         beta=alpha / self.tracer.n,
                                                                         gamma=self.tracer.gamma,
                                                                         n=jnp.arange(self.tracer.n)))
        print("Initialized agent.")

    def update_agent(self, step):
        transition_batch = buffer.sample(batch_size=self._batch_size)
        metrics = {}
        # flip a coin to decide which of the q-functions to update
        qlearning = self.qlearning1 if jax.random.bernoulli(self.q1.rng) else self.qlearning2
        metrics.update(qlearning.update(transition_batch))
        metrics.update(self.soft_pg.update(transition_batch))

        #TODO: log metrics
        #env.record_metrics(metrics)

        # sync target networks
        self.q1_target.soft_update(self.q1, tau=0.001)
        self.q2_target.soft_update(self.q2, tau=0.001)

    def load(self, path):
        """ Load checkpointed model. """
        self.policy, self.q1, self.q2, self.q1_target, self.q2_target, self.qlearning1, self.qlearning2, self.soft_pg = coax.utils.load(path)

    def save(self):
        """ Checkpoint model. """
        path = os.path.join(self.model_dir, 'checkpoint.pkl.lz4')
        #For some reason there's an error here to do with pickle. Pickling this outside of the class works, though.
        #coax.utils.dump((self.policy, self.q1, self.q2, self.q1_target, self.q2_target, self.qlearning1, self.qlearning2, self.soft_pg), path)
