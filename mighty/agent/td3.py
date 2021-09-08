import os
import time
import copy
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from mighty.utils.logger import Logger
from mighty.utils.rollout_worker import RolloutWorker
from mighty.utils.replay_buffer import ReplayBuffer
from mighty.utils.value_function import Actor, Critic
from mighty.utils.weight_updates import soft_update
from mighty.agent.base import AbstractAgent
from mighty.env.env_handling import DACENV

from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint


class TD3Agent(AbstractAgent):
    """
    Simple TD3 Agent
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

    def __init__(
            self,
            env: DACENV,
            env_eval: DACENV,
            logger: Logger,
            epsilon: float = 0.2,
            gamma: float = 0.99,
            batch_size: int = 64,
            max_size_replay_buffer: int = 1_000_000,
            begin_updating_weights: int = 1,
            soft_update_weight: float = 0.005,
            max_env_time_steps: int = 1_000_000,
            log_tensorboard: bool = True,
            args: argparse.Namespace = None,  # from DDQNConfigParser
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            initial_random_steps=25e3
    ):
        """
        Initialize the TD3 Agent

        args overrides all hyperparams if given.

        :param gamma: discount factor
        :param env: environment to train on
        :param logger: logging functionality of some sort #FIXME
        """
        if args:
            # overwrite defaults
            gamma = args.gamma
            epsilon = args.epsilon
            batch_size = args.batch_size
            max_size_replay_buffer = args.max_size_replay_buffer
            begin_updating_weights = args.begin_updating_weights
            soft_update_weight = args.soft_update_weight
            max_env_time_steps = args.max_env_time_steps

        super().__init__(env=env, gamma=gamma, logger=logger, env_eval=env_eval, output_dir=logger.log_dir)
        self._env_eval = env_eval  # TODO: should the abstract agent get this?

        self._replay_buffer = ReplayBuffer(max_size_replay_buffer)
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._begin_updating_weights = begin_updating_weights
        self._soft_update_weight = soft_update_weight  # type: float  # TODO add description
        self._max_env_time_steps = max_env_time_steps  # type: int
        try:
            self._n_episodes_eval = len(self.env.instance_set.keys())  # type: int
        except AttributeError:
            self._n_episodes_eval = 10
        self.output_dir = self.logger.log_dir
        self.model_dir = os.path.join(self.output_dir, 'models')

        self.writer = None
        if log_tensorboard:
            # TODO write out all the other hyperparameters
            self.writer = SummaryWriter(self.logger.log_dir)
            self.writer.add_scalar('batch_size/Hyperparameter', self._batch_size)
            self.writer.add_scalar('policy_epsilon/Hyperparameter', self._epsilon)
        
        max_action = float(env.action_space.high[0])
        self.actor = Actor(self._state_shape, self._action_dim, max_action).to(self.device)
        self.actor_target = Actor(self._state_shape, self._action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(self._state_shape, self._action_dim).to(self.device)
        self.critic_target = Critic(self._state_shape, self._action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.initial_random_steps = initial_random_steps
        self.total_updates = 0
        self._mapping_save_components = {"actor_model": self.actor,
                                           "critic_model": self.critic,
                                           "actor_optimizer": self.actor_optimizer,
                                           "critic_optimizer": self.critic_optimizer}

    def save_replay_buffer(self, path):
        self._replay_buffer.save(path)

    def load_replay_buffer(self, path):
        self._replay_buffer.load(path)

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """
        Simple helper to get action based on observation x
        """
        return self.actor(self.tt(state)).detach().numpy()

    def step(self, engine: Engine = None, iteration=None):
        """
        Used as process function for ignite. Must have as args: engine, batch.

        :param engine:
        :param iteration:
        :return:
        """
        if self.total_steps < self.initial_random_steps:
            a = self.env.action_space.sample()
        else:
            # Only for continuous action spaces. For discrete we would need normal epsilon greedy
            a = self.get_action(self.last_state) + np.random.normal(0, self.max_action * self._epsilon,
                                                                    size=self._action_dim)
            a = a.clip(-self.max_action, self.max_action)

        ns, r, d, _ = self.env.step(a)
        self.total_steps += 1
        self.logger.next_step()
        self._replay_buffer.add_transition(self.last_state, a, ns, r, d)
        self.reset_needed = d

        if self.total_steps >= self._begin_updating_weights:
            self.total_updates += 1
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                map(self.tt, self._replay_buffer.random_next_batch(self._batch_size))

            # Update the critic
            with torch.no_grad():
                noise = (torch.randn_like(batch_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_actions = (self.actor_target(batch_next_states) + noise).clamp(-self.max_action, self.max_action)

                # Compute Targets
                targetQ1, targetQ2 = self.critic_target(batch_next_states, next_actions)
                targetQ = torch.min(targetQ1, targetQ2)
                targetQ = batch_rewards.reshape(-1, 1) + (1 - batch_terminal_flags
                                                          ).reshape(-1, 1) * self.gamma * targetQ

            # Get temporal differences
            currentQ1, currentQ2 = self.critic(batch_states, batch_actions)
            critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)

            # Update weights
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor
            if self.total_updates % self.policy_freq == 0:
                actor_loss = -self.critic.Q1(batch_states, self.actor(batch_states)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                soft_update(self.critic_target, self.critic, self._soft_update_weight)
                soft_update(self.actor_target, self.actor, self._soft_update_weight)

            if self.writer is not None:
                self.writer.add_scalar('CriticLoss/train', critic_loss, self.total_steps)
                if self.total_steps % self.policy_freq == 0:
                    self.writer.add_scalar('ActorLoss/train', actor_loss, self.total_steps)
                for d in range(len(a)):
                    self.writer.add_scalar('ActionD{d}/train', a[d], self.total_steps)
                # This apparently requires a module named "past" that the docs don't mention.
                # Also this is not how arrays should be logged, I think, so it should be fixed
                # self.writer.add_embedding('State/train', self.last_state, self.total_steps)
                self.writer.add_scalar('Reward/train', r, self.total_steps)

        if d:
            if engine is not None:
                engine.terminate_epoch()
            self.end_logger_episode()

        state = ns  # stored in engine.state # TODO
        self.last_state = state
        return state

    def end_logger_episode(self):
        self.logger.next_episode()

    def check_termination(self, engine):
        if engine.state.iteration > self._max_env_time_steps:
            engine.fire_event(Events.EPOCH_COMPLETED)

    def print_epoch(self, engine):
        episode = engine.state.epoch
        n_episodes = engine.state.max_epochs
        print("%s/%s" % (episode + 1, n_episodes))

    def __repr__(self):
        return 'TD3'

    def load_checkpoint(self, path: str, replay_path: str = None):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_model'])
        self.critic.load_state_dict(checkpoint['critic_model'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if replay_path is not None:
            self._replay_buffer.load(replay_path)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
