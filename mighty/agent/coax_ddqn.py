import gym
import coax
import optax
import haiku as hk
import jax.numpy as jnp

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import copy
import math
from gym.wrappers import TimeLimit
from typing import Optional, Dict

from rich import print, box
import logging
from rich.logging import RichHandler
from rich.progress import Progress, TimeRemainingColumn, TimeElapsedColumn, BarColumn
from rich.theme import Theme

from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina_examples.rl.dqn.double_dqn.dqn import soft_update_params, _state_dict
from salina.rl.replay_buffer import ReplayBuffer
from salina.logger import TFLogger
import salina.rl.functional as RLF

from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger
from mighty.train.dacbench_salina_agent import AutoResetDACBenchAgent
from mighty.agent.policies import DQNMLPAgent


# pick environment
env = gym.make(...)
env = coax.wrappers.TrainMonitor(env)

import torch
import torch.nn as nn
from salina import TAgent, instantiate_class
from salina_examples.rl.dqn.agents import MLP



class DQNMLPAgent(TAgent):
    def __init__(self, pi, noise = None):
        super().__init__()
        self.pi = pi
        self.noise = noise

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))
        input_with_noise = self.noise(input)
        action = self.pi(input_with_noise)
        self.set(("action", t), action)


def func_pi(t, is_training):
    #FIXME: read state given t
    # custom haiku function (for continuous actions in this example)
    mu = hk.Sequential([...])(S)  # mu.shape: (batch_size, *action_space.shape)
    #FIXME: write to space instead of return
    return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}  # deterministic policy


def func_q(S, A, is_training):
    # custom haiku function
    value = hk.Sequential([...])
    return value(S)  # output shape: (batch_size,)


# define function approximator
pi = coax.Policy(func_pi, env)
q = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)

#FIXME: make remote eval workers from policy

# target networks
pi_targ = pi.copy()
q_targ = q.copy()


# specify how to update policy and value function
determ_pg = coax.policy_objectives.DeterministicPG(pi, q, optimizer=optax.adam(0.001))
qlearning = coax.td_learning.QLearning(q, pi_targ, q_targ, optimizer=optax.adam(0.002))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


# action noise
noise = coax.utils.OrnsteinUhlenbeckNoise(mu=0., sigma=0.2, theta=0.15)


for ep in range(100):
    s = env.reset()
    noise.reset()
    noise.sigma *= 0.99  # slowly decrease noise scale

    for t in range(env.spec.max_episode_steps):
        a = noise(pi(s))
        s_next, r, done, info = env.step(a)

        # add transition to buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # update
        transition_batch = buffer.sample(batch_size=32)
        metrics_q = qlearning.update(transition_batch)
        metrics_pi = determ_pg.update(transition_batch)
        env.record_metrics(metrics_q)
        env.record_metrics(metrics_pi)

        # periodically sync target models
        if ep % 10 == 0:
            pi_targ.soft_update(pi, tau=1.0)
            q_targ.soft_update(q, tau=1.0)

        if done:
            break

        s = s_next


# This is a copy of the selina only agent
class DDQNAgent(object):
    """
        Simple double DQN Agent
        """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tt(self, ndarray):
        """
        Helper Function to cast observation to correct type/device
        """
        if self.device == "cuda":
            return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
        else:
            return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

    def func_pi(t, is_training):
        # FIXME: read state given t
        # custom haiku function (for continuous actions in this example)
        mu = hk.Sequential([...])(S)  # mu.shape: (batch_size, *action_space.shape)
        # FIXME: write to space instead of return
        return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}  # deterministic policy

    def func_q(S, A, is_training):
        # custom haiku function
        value = hk.Sequential([...])
        return value(S)  # output shape: (batch_size,)

    def __init__(
            self,
            env_kwargs: Dict,
            logger: TFLogger,
            env_eval_kwargs: Optional[Dict] = None,
            discount_factor: float = 0.99,
            epsilon: float = 0.2,
            batch_size: int = 64,
            learning_rate: float = 0.001,  # TODO pass general optimizer args, pass optimizer class
            replay_buffer_size: int = 1_000_000,
            replay_buffer_time_size: int = 2,
            initial_replay_buffer_size: int = 1000,
            begin_updating_weights: int = 1,
            soft_update_weight: float = 0.01,
            max_env_time_steps: int = 10_000_000,
            checkpoint_mode: str = 'latest',
            log_tensorboard: bool = True,
            n_acq_processes: int = 8,  # TODO test with > 0
            n_eval_processes: int = 1,
            seed: int = 42,
            args: argparse.Namespace = None,  # from AgentConfigParser
            epsilon_start: float = -1,
            epsilon_final: float = -1,
            burning_timesteps: int = 0,
            inner_epochs: int = 1,
            clip_grad: float = 2,  # TODO: data type float? what default value?
            update_target_epochs: int = 1000,  # TODO default value?
            overlapping_timesteps: int = 1,  # TODO default value?
            # TODO pass general training (loop) args
            n_train_timesteps: int = 2,  # TODO default value?
            eval_episode_cutoff: int = 100,  # TODO does this make sense?
            n_envs: Optional[int] = None,
            render_progress: bool = True,
    ):
        if args:
            # overwrite defaults
            discount_factor = args.discount_factor
            epsilon = args.epsilon
            batch_size = args.batch_size
            learning_rate = args.learning_rate
            replay_buffer_size = args.replay_buffer_size
            begin_updating_weights = args.begin_updating_weights
            soft_update_weight = args.soft_update_weight
            max_env_time_steps = args.max_env_time_steps
            checkpoint_mode = args.checkpoint_mode
            n_acq_processes = args.n_acq_processes
            n_eval_processes = args.n_eval_processes
            seed = args.seed
            initial_replay_buffer_size = args.initial_replay_buffer_size
            epsilon_start = args.epsilon_start
            epsilon_final = args.epsilon_final
            burning_timesteps = args.burning_timesteps
            inner_epochs = args.inner_epochs
            clip_grad = args.clip_grad
            update_target_epochs = args.update_target_epochs
            replay_buffer_time_size = args.replay_buffer_time_size
            overlapping_timesteps = args.overlapping_timesteps
            n_train_timesteps = args.n_train_timesteps
            eval_episode_cutoff = args.eval_episode_cutoff  # TODO rename arg n_eval_steps to this in parser
            n_envs = args.n_envs

        if logger is not None:
            output_dir = logger.log_dir
        else:
            output_dir = None

        self.env_kwargs = env_kwargs
        if env_eval_kwargs is None:
            self.env_eval_kwargs = self.env_kwargs
        else:
            self.env_eval_kwargs = env_eval_kwargs
        self.n_acq_processes = n_acq_processes
        self.n_eval_processes = n_eval_processes
        if n_envs is None:
            n_envs = self.n_acq_processes
        self.n_envs = n_envs
        self.discount_factor = discount_factor
        self.logger = logger
        self.checkpoint_mode = checkpoint_mode
        self.seed = seed
        self.loss_device = self.device
        self.eval_device = self.device
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer_time_size = replay_buffer_time_size
        self.initial_replay_buffer_size = initial_replay_buffer_size
        self.burning_timesteps = burning_timesteps
        self.inner_epochs = inner_epochs
        self.clip_grad = clip_grad
        self.overlapping_timesteps = overlapping_timesteps
        self.n_train_timesteps = n_train_timesteps
        self.eval_episode_cutoff = eval_episode_cutoff
        # TODO: make util function that can detect this correctly for all kinds of gym spaces and use it here
        # try:
        #     self._action_dim = self.env.action_space.n
        # except AttributeError:
        #     self._action_dim = self.env.action_space.shape[0]
        # self._state_shape = self.env.observation_space.shape[0]

        self.render_progress = render_progress

        self.max_env_time_steps = max_env_time_steps  # type: int

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.model_dir = os.path.join(self.output_dir, 'models')

        self.last_state = None
        self.total_steps = 0

        self._mapping_save_components = None  # type: Optional[Dict[str, Any]]

        self._loss_function = nn.MSELoss()
        self.lr = learning_rate

        self._epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_schedule = self.epsilon_static
        if self.epsilon_start >= 0 and self.epsilon_final >= 0:
            self.epsilon_schedule = self.epsilon_schedule_decay

        self._batch_size = batch_size
        self._begin_updating_weights = begin_updating_weights
        self._soft_update_weight = soft_update_weight  # type: float
        self.hard_target_update = False
        if self._soft_update_weight < 0:
            self.hard_target_update = True  # TODO is this correct?
        self.update_target_epochs = update_target_epochs

        self.writer = None
        if log_tensorboard and output_dir is not None:
            self.writer = SummaryWriter(output_dir)
            self.writer.add_scalar('hyperparameter/lr', self.lr)
            self.writer.add_scalar('hyperparameter/batch_size', self._batch_size)
            self.writer.add_scalar('hyperparameter/policy_epsilon', self._epsilon)

        print("Create salina agents.")
        # define function approximator
        self.policy = coax.Policy(self.func_pi, env)
        self.q_agent = coax.Q(self.func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)

        # FIXME: make sure the remote worker creation works

        self.acq_remote_agent, self.acq_workspace = NRemoteAgent.create(
            self.q_agent,
            num_processes=self.n_acq_processes,
            t=0,  # start time step
            n_steps=self.n_train_timesteps,
            epsilon=1.0,  # exploration fraction (set to 1 for initial random filling of the replay buffer)
            # stochastic=False,  # TODO or stochastic = True?
        )

        self.eval_remote_agent, self.eval_workspace = NRemoteAgent.create(
            self.q_agent,
            num_processes=self.n_eval_processes,
            t=0,
            n_steps=self.eval_episode_cutoff,  # TODO  adjust these argument, how many steps?
            epsilon=0,  # TODO  adjust these argument
        )
        self.eval_remote_agent.seed(self.seed)
        self.eval_rollout(eval_episode_cutoff=self.eval_episode_cutoff, is_init=True)

        self.acq_remote_agent.seed(self.seed)
        # target networks
        self.policy_target = self.policy.copy()
        self.q_target_agenttarg = self.q_agent.copy()

        # specify how to update policy and value function
        self.determ_pg = coax.policy_objectives.DeterministicPG(self.policy, self.q_agent, optimizer=optax.adam(0.001))
        self.qlearning = coax.td_learning.QLearning(self.q_agent, self.policy_target, self.q_target_agent, optimizer=optax.adam(0.002))

        # specify how to trace the transitions
        self.tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)

        # action noise
        self.noise = coax.utils.OrnsteinUhlenbeckNoise(mu=0., sigma=0.2, theta=0.15)
        print("Initialized agent.")


    def epsilon_schedule_decay(self, epoch):
        epsilon_by_epoch = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(
            -1.0 * epoch / self.epsilon_exploration_decay)
        return epsilon_by_epoch

    def epsilon_static(self, **kwargs):
        return self._epsilon

    def eval_rollout(self, eval_episode_cutoff: int, is_init: bool = False):
        for a in self.eval_remote_agent.get_by_name('q_agent'):
            a.load_state_dict(_state_dict(self.q_agent, self.eval_device))
        t = 0
        if not is_init:
            self.eval_workspace.copy_n_last_steps(1)
            eval_episode_cutoff -= 1
            t = 1
        self.eval_remote_agent._asynchronous_call(
            self.eval_workspace,
            t=t,
            n_steps=eval_episode_cutoff,  # TODO how to set this? does this need to be a function arg?
            epsilon=0
        )

    def train(
            self,
            n_steps: int,
            n_episodes_eval: int,
            eval_every_n_steps: int = 1_000,
            human_log_every_n_episodes: int = 100,
            save_model_every_n_episodes: int = 100,
    ):
        self.logger.message("[DDQN] Learning")

        with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                'Remaining:',
                TimeRemainingColumn(),
                'Elapsed:',
                TimeElapsedColumn()
        ) as progress:
            # epoch_task = progress.add_task("Epochs", total=max_epoch, start=False, visible=False)
            steps_task = progress.add_task("Train Steps", total=n_steps, start=False, visible=False)

            # Setup optimizers
            # optimizer_args = get_arguments(cfg.algorithm.optimizer)
            # optimizer = get_class(cfg.algorithm.optimizer)(
            #     q_agent.parameters(), **optimizer_args
            # )
            optimizer_args = dict(lr=self.lr)
            optimizer = optim.Adam(self.q_agent.parameters(), **optimizer_args)

            self._mapping_save_components = {
                "model": self.q_agent,
                "targets": self.q_target_agent,
                "optimizer": optimizer
            }

            # TODO: calculate n_episodes, max_epochs, buffer_time_size...

            total_time_steps = 0

            # Progress speed variables to check which progressbar to actually show
            epoch_progress = 0
            n_acq_processes = self.n_acq_processes if self.n_acq_processes > 0 else 1
            step_progress = (n_acq_processes * int(self.n_envs / n_acq_processes) * self.n_train_timesteps)

            start_eval_timesteps = 0
            epoch = 0
            iteration = 0
            while not progress.finished:
                # Handles evaluation
                if not self.eval_remote_agent.is_running():  # Start new evaluation run if the eval agent is not busy
                    done, creward = self.eval_workspace["env/done", 'env/cumulated_reward']
                    creward = creward[done]
                    if not creward.size()[0] == 0:
                        self.logger.add_scalar("evaluation/reward", creward.mean().item(), total_time_steps)
                    self.eval_rollout(eval_episode_cutoff=self.eval_episode_cutoff, is_init=False)

                epsilon = self.epsilon_schedule(epoch=epoch)
                self.logger.add_scalar("monitor/epsilon", epsilon, iteration)
                # Run the acquisition agents
                self.acq_workspace.copy_n_last_steps(self.overlapping_timesteps)
                self.acq_remote_agent(
                    self.acq_workspace,
                    t=self.overlapping_timesteps,
                    n_steps=self.n_train_timesteps - self.overlapping_timesteps,
                    epsilon=epsilon,
                )
                # Update the replay buffer
                #FIXME: this is the salina replay buffer, above and below it's the coax one. Pick one and stick with it (probably this one)
                self.replay_buffer.put(self.acq_workspace, time_size=self.replay_buffer_time_size)

                total_time_steps += step_progress
                if total_time_steps == step_progress:
                    progress.start_task(steps_task)
                    if epoch_progress <= step_progress:
                        progress.update(steps_task, visible=self.render_progress)
                elif total_time_steps >= step_progress:
                    progress.advance(steps_task, step_progress)

                # Log the cumulated reward
                done, creward = self.acq_workspace["env/done", "env/cumulated_reward"]
                creward = creward[done]
                if creward.size()[0] > 0:
                    self.logger.add_scalar("monitor/reward", creward.mean().item(), iteration)
                    # print(f'\t\tReward {creward}')
                    # print(f'\t\t\t\t\t\tAVG Reward {creward.mean().item():>8.5f} | Epoch {epoch}')

                self.logger.add_scalar("monitor/replay_buffer_size", self.replay_buffer.size(), iteration)

                s = self.env.reset()
                self.noise.reset()
                self.noise.sigma *= 0.99  # slowly decrease noise scale

                for t in range(self.inner_epochs):
                    # update
                    transition_batch = self.buffer.sample(batch_size=32)
                    metrics_q = self.qlearning.update(transition_batch)
                    metrics_pi = self.determ_pg.update(transition_batch)
                    #env.record_metrics(metrics_q)
                    #env.record_metrics(metrics_pi)

                    # periodically sync target models
                    if ep % 10 == 0:
                        pi_targ.soft_update(pi, tau=1.0)
                        q_targ.soft_update(q, tau=1.0)

                    if done:
                        break

                    s = s_next

                epoch += 1
                if total_time_steps >= self.max_env_time_steps:
                    progress.finished = True  # TODO correctly emit progress finished
                # TODO add saving

    def run(
            self,
            n_steps: int,
            n_episodes_eval: int,
            eval_every_n_steps: int = 1_000,
            human_log_every_n_episodes: int = 100,
            save_model_every_n_episodes: int = 100,
    ):
        import torch.multiprocessing as mp
        context = mp.get_context("spawn")

        self.train(
            n_steps=n_steps,
            n_episodes_eval=n_episodes_eval,
            eval_every_n_steps=eval_every_n_steps,
            human_log_every_n_episodes=human_log_every_n_episodes,
            save_model_every_n_episodes=save_model_every_n_episodes
        )

    def load(self):
        """ Load checkpointed model. """
        raise NotImplementedError

    def eval(self, env, episodes):
        """
        Eval agent on an environment. (Full evaluation)
        :param env:
        :param episodes:
        :return:
        """
        raise NotImplementedError