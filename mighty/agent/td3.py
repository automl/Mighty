import os
import time
import copy
import json
import argparse
import gym

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

#FIXME: error is most likely in eval! The original eval method in in here now and currently compared to ours
#Check for global variables that may cause problems
#How parallel is the engine? Do we need an eval engine?
#Remove agent copy from rollout
#Seed envs for comparison

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
            # The eval logger should be removed as soon as the logger is reconstructed
            eval_logger: Logger,
            max_action: float,
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

        super().__init__(env=env, gamma=gamma, logger=logger)
        self._env_eval = env_eval  # TODO: should the abstract agent get this?
        self.eval_logger = eval_logger

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
                self.writer.add_scalar('Action/train', a, self.total_steps)
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

    def start_episode(self, engine):
        self.last_state = self.env.reset()
        self.logger.reset_episode()
        self.logger.set_env(self.env)

    def end_logger_episode(self):
        self.logger.next_episode()

    def check_termination(self, engine):
        if engine.state.iteration > self._max_env_time_steps:
            engine.fire_event(Events.EPOCH_COMPLETED)

    # TODO: should this maybe at least in part be in the superclass?
    # Basics should be in superclass, extensions here and everything should be extendable in runscript
    def train(
            self,
            episodes: int,
            epsilon: float,  # FIXME why are these arguments still part of the train call?
            max_env_time_steps: int,
            n_episodes_eval: int = 1,
            eval_every_n_steps: int = 1,
            max_train_time_steps: int = 1_000_000,
    ):
        # self._n_episodes_eval = n_episodes_eval

        # Init Engine
        trainer = Engine(self.step)

        # Register events
        # STARTED

        # EPOCH_STARTED
        # reset env
        trainer.add_event_handler(Events.EPOCH_STARTED, self.start_episode)

        # ITERATION_STARTED

        # ITERATION_COMPLETED
        eval_kwargs = dict(
            env=self._env_eval,
            episodes=self._n_episodes_eval,
            max_env_time_steps=self._max_env_time_steps,
        )
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=eval_every_n_steps), self.run_rollout, **eval_kwargs)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=eval_every_n_steps), self.eval_policy, **{"env_name": "Pendulum-v0", "seed": 0})
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.check_termination)

        # EPOCH_COMPLETED

        checkpoint_handler = ModelCheckpoint(self.model_dir, filename_prefix='', n_saved=None, create_dir=True)
        # TODO: add log mode saving everything (trainer, optimizer, etc.)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=100), checkpoint_handler,
                                  to_save={"actor_model": self.actor,
                                           "critic_model": self.critic,
                                           "actor_optimizer": self.actor_optimizer,
                                           "critic_optimizer": self.critic_optimizer})
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=100), self.print_epoch)

        # COMPLETED
        # order of registering matters! first in, first out
        # we need to save the model first before evaluating
        trainer.add_event_handler(Events.COMPLETED, checkpoint_handler,
                                  to_save={"actor_model": self.actor,
                                           "critic_model": self.critic,
                                           "actor_optimizer": self.actor_optimizer,
                                           "critic_optimizer": self.critic_optimizer})
        trainer.add_event_handler(Events.COMPLETED, self.run_rollout, **eval_kwargs)

        # RUN
        iterations = range(self._max_env_time_steps)
        trainer.run(iterations, max_epochs=episodes)

    def eval_policy(self, env_name, seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        #eval_env.seed(seed + 100)
        #eval_env = self._env_eval
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.get_action(np.array(state), epsilon=0)
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    # TODO: max_env_time_steps is and env option
    def run_rollout(self, env, episodes, max_env_time_steps):
        # TODO: check for existing current checkpoint
        self.checkpoint(self.output_dir)
        # TODO: for this to be nice we want to separate policy and agent
        # agent = DDQN(self.env)
        # TODO: this should be easier
        for _, m in self.eval_logger.module_logger.items():
            m.episode = self.logger.module_logger["train_performance"].episode
        #worker = RolloutWorker(self, self.output_dir, self.eval_logger)

        # TODO: Why does this use the workers evaluate method and not the agents eval method?
        env = gym.make('Pendulum-v0')
        #worker.evaluate(env, episodes)
        print("Starting evaluation")
        reward = 0
        for i in range(episodes):
            done = False
            s = env.reset()
            #self.eval_logger.reset_episode()
            #self.eval_logger.set_env(env)
            while not done:
                a = self.get_action(s, epsilon=0)
                ns, r, done, _ = env.step(a)
                reward += r
            #self.eval_logger.write()
        print(f"Eval reward:{reward / episodes}")
        # os.remove(self.output_dir / "Q")  # FIXME I don't know why this is here

    def evaluate(self, engine, env: DACENV, episodes: int = 1, max_env_time_steps: int = 1_000_000):
        eval_s, eval_r, eval_d, pols = self.eval(
            env=env, episodes=episodes, max_env_time_steps=max_env_time_steps)

        eval_stats = dict(
            elapsed_time=engine.state.times[Events.EPOCH_COMPLETED.name],
            # TODO check if this is the total time passed since start of training
            training_steps=engine.state.iteration,
            training_eps=engine.state.epoch,
            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
            avg_rew_per_eval_ep=float(np.mean(eval_r)),
            std_rew_per_eval_ep=float(np.std(eval_r)),
            eval_eps=episodes
        )
        per_inst_stats = dict(
            # eval_insts=self._train_eval_env.instances,
            reward_per_isnts=eval_r,
            steps_per_insts=eval_s,
            policies=pols
        )

        with open(os.path.join(self.output_dir, 'eval_scores.json'), 'a+') as out_fh:
            json.dump(eval_stats, out_fh)
            out_fh.write('\n')
        with open(os.path.join(self.output_dir, 'eval_scores_per_inst.json'), 'a+') as out_fh:
            json.dump(per_inst_stats, out_fh)
            out_fh.write('\n')

    def print_epoch(self, engine):
        episode = engine.state.epoch
        n_episodes = engine.state.max_epochs
        print("%s/%s" % (episode + 1, n_episodes))

    def __repr__(self):
        return 'TD3'

    def eval(self, env: DACENV, episodes: int = 1, max_env_time_steps: int = 1_000_000):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        policies = []
        with torch.no_grad():
            for e in range(episodes):
                # this_env.instance_index = this_env.instance_index % 10  # for faster debuggin on only 10 insts
                print(f'Eval Episode {e} of {episodes}')
                ed, es, er = 0, 0, 0

                s = env.reset()
                # policy = [float(this_env.current_lr.numpy()[0])]
                for _ in count():
                    a = self.get_action(s)
                    ed += 1

                    ns, r, d, _ = env.step(a)
                    er += r
                    es += 1
                    if es >= max_env_time_steps or d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)
                policies.append(None)

        # TODO: log this somehow
        return steps, rewards, decisions, policies

    def checkpoint(self, filepath: str):
        torch.save(self.critic.state_dict(), os.path.join(filepath, 'critic'))
        torch.save(self.actor.state_dict(), os.path.join(filepath, 'actor'))

        torch.save(self.actor_optimizer.state_dict(), os.path.join(filepath, 'actor_optimizer'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(filepath, 'critic_optimizer'))

    def load(self, filepath: str):
        self.actor.load_state_dict(torch.load(os.path.join(filepath, 'actor')))
        self.critic.load_state_dict(torch.load(os.path.join(filepath, 'critic')))

        self.critic_optimizer.load_state_dict(torch.load(os.path.join(filepath, 'critic_optimizer')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(filepath, 'actor_optimizer')))

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)


def run_agent(arg):
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    ##### THIS IS ONLY FOR DEBUGGING PURPOSES
    from pathlib import Path
    from collections import namedtuple
    from gym.wrappers import TransformObservation

    import gym
    from mighty.utils.logger import Logger
    from mighty.iohandling.experiment_tracking import prepare_output_dir

    class TMP(TransformObservation):

        def __init__(self, env, f):
            super().__init__(env, lambda x: x)

        def get_inst_id(self):
            return 0

    env = gym.make('Pendulum-v0')
    eenv = gym.make('Pendulum-v0')

    #env = TMP(env, None)
    #eenv = TMP(eenv, None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    args = namedtuple('Args', ['seed'])
    args.seed = 12345
    out_dir = prepare_output_dir(args)
    train_logger = Logger(
        experiment_name=f"pendulum_example_{args.seed}",
        output_path=Path(out_dir),
        step_write_frequency=None,
        episode_write_frequency=10,
    )

    eval_logger = Logger(
        experiment_name=f"pendulum_example_{args.seed}",
        output_path=Path(out_dir),
        step_write_frequency=None,
        episode_write_frequency=1,
    )

    os.makedirs(out_dir, exist_ok=True)

    agent = TD3Agent(env, eenv, logger=train_logger,
                     eval_logger=eval_logger,
                     max_action=max_action,
                     epsilon=0.1,
                     gamma=.99,
                     batch_size=256, log_tensorboard=False,
                     begin_updating_weights=1000, policy_noise=max_action * 0.2, noise_clip=max_action * 0.5,
                     max_env_time_steps=int(1e3))
    agent.train(1000, .1, 1000, eval_every_n_steps=500)

if __name__ == '__main__':
    import sys
    run_agent(sys.argv[1:])