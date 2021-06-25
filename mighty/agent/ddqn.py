import os
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from mighty.utils.logger import Logger
from mighty.utils.rollout_worker import RolloutWorker
from mighty.utils.replay_buffer import ReplayBuffer
from mighty.utils.value_function import FullyConnectedQ
from mighty.utils.weight_updates import soft_update
from mighty.agent.base import AbstractAgent
from mighty.env.env_handling import DACENV

from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint


class DDQNAgent(AbstractAgent):
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

    def __init__(
            self, 
            env: DACENV,
            env_eval: DACENV,
            logger: Logger,
            gamma: float = 0.99,
            epsilon: float = 0.2,
            batch_size: int = 64,
            learning_rate: float = 0.001,
            max_size_replay_buffer: int = 1_000_000,
            begin_updating_weights: int = 1,
            soft_update_weight: float = 0.01,
            max_env_time_steps: int = 1_000_000,
            log_tensorboard: bool = True,
            args: argparse.Namespace = None  # from AgentConfigParser
    ):
        """
        Initialize the DQN Agent

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
            learning_rate = args.learning_rate
            max_size_replay_buffer = args.max_size_replay_buffer
            begin_updating_weights = args.begin_updating_weights
            soft_update_weight = args.soft_update_weight
            max_env_time_steps = args.max_env_time_steps

        super().__init__(env=env, gamma=gamma, logger=logger)
        self._env_eval = env_eval  # TODO: should the abstract agent get this?

        self._q = FullyConnectedQ(self._state_shape, self._action_dim).to(self.device)
        self._q_target = FullyConnectedQ(self._state_shape, self._action_dim).to(self.device)

        self._loss_function = nn.MSELoss()
        self.lr = learning_rate
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=self.lr)

        self._replay_buffer = ReplayBuffer(max_size_replay_buffer)
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._begin_updating_weights = begin_updating_weights
        self._soft_update_weight = soft_update_weight  # type: float  # TODO add description
        self._max_env_time_steps = max_env_time_steps  # type: int
        self._n_episodes_eval = len(self.env.instance_set.keys())  # type: int
        self.output_dir = self.logger.log_dir
        self.model_dir = os.path.join(self.output_dir, 'models')
        
        self.writer = None
        if log_tensorboard:
            self.writer = SummaryWriter(self.logger.log_dir)
            self.writer.add_scalar('lr/Hyperparameter', self.lr)
            self.writer.add_scalar('batch_size/Hyperparameter', self._batch_size)
            self.writer.add_scalar('policy_epsilon/Hyperparameter', self._epsilon)

    def save_replay_buffer(self, path):
        self._replay_buffer.save(path)

    def load_replay_buffer(self, path):
        self._replay_buffer.load(path)

    def get_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(self.tt(state)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def step(self, engine: Engine=None, iteration=None):
        """
        Used as process function for ignite. Must have as args: engine, batch.

        :param engine:
        :param iteration:
        :return:
        """
        a = self.get_action(self.last_state, self._epsilon)
        ns, r, d, _ = self.env.step(a)
        self.total_steps += 1
        self.logger.next_step()
        self._replay_buffer.add_transition(self.last_state, a, ns, r, d)
        self.reset_needed = d

        if self.total_steps >= self._begin_updating_weights:
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                map(self.tt, self._replay_buffer.random_next_batch(self._batch_size))

            target = batch_rewards + (1 - batch_terminal_flags) * self.gamma * \
                     self._q_target(batch_next_states)[
                         torch.arange(self._batch_size).long(), torch.argmax(
                             self._q(batch_next_states), dim=1)]
            current_prediction = self._q(batch_states)[torch.arange(self._batch_size).long(), batch_actions.long()]
    
            loss = self._loss_function(current_prediction, target.detach())
            
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss, self.total_steps)
                self.writer.add_scalar('Action/train', a, self.total_steps)
                #This apparently requires a module named "past" that the docs don't mention. 
                #Also this is not how arrays should be logged, I think, so it should be fixed
                #self.writer.add_embedding('State/train', self.last_state, self.total_steps)
                self.writer.add_scalar('Reward/train', r, self.total_steps)

            self._q_optimizer.zero_grad()
            loss.backward()
            self._q_optimizer.step()
    
            soft_update(self._q_target, self._q, self._soft_update_weight)

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
        self.logger.set_train_env(self.env)

    def end_logger_episode(self):
        self.logger.next_episode()

    def check_termination(self, engine):
        if engine.state.iteration > self._max_env_time_steps:
            engine.fire_event(Events.EPOCH_COMPLETED)

    #TODO: should this maybe at least in part be in the superclass?
    # Basics should be in superclass, extensions here and everything should be extendable in runscript
    def train(
            self,
            episodes: int,
            epsilon: float,
            max_env_time_steps: int,
            n_episodes_eval: int = 1,
            eval_every_n_steps: int = 1, 
            max_train_time_steps: int = 1_000_000,
    ):
        #self._n_episodes_eval = n_episodes_eval
        
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
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.check_termination)

        # EPOCH_COMPLETED

        checkpoint_handler = ModelCheckpoint(self.model_dir, filename_prefix='', n_saved=None, create_dir=True)
        # TODO: add log mode saving everything (trainer, optimizer, etc.)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=100), checkpoint_handler, to_save={"model": self._q})
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=100), self.print_epoch)

        # COMPLETED
        # order of registering matters! first in, first out
        # we need to save the model first before evaluating
        trainer.add_event_handler(Events.COMPLETED, checkpoint_handler, to_save={"model": self._q})
        trainer.add_event_handler(Events.COMPLETED, self.run_rollout, **eval_kwargs)

        # RUN
        iterations = range(self._max_env_time_steps)
        trainer.run(iterations, max_epochs=episodes)

    #TODO: max_env_time_steps is and env option
    def run_rollout(self, env, episodes, max_env_time_steps):
        #TODO: check for existing current checkpoint
        self.checkpoint(self.output_dir)
        #TODO: for this to be nice we want to separate policy and agent
        #agent = DDQN(self.env)
        #TODO: this should be easier
        for _, m in self.logger.module_logger.items():
            m.episode = self.logger.module_logger["train_performance"].episode
        worker = RolloutWorker(self, self.output_dir, self.logger)
        worker.evaluate(env, episodes)
        os.remove(self.output_dir / "Q")

    def evaluate(self, engine, env: DACENV, episodes: int = 1, max_env_time_steps: int = 1_000_000):
        eval_s, eval_r, eval_d, pols = self.eval(
            env=env, episodes=episodes, max_env_time_steps=max_env_time_steps)

        eval_stats = dict(
            elapsed_time=engine.state.times[Events.EPOCH_COMPLETED.name],  # TODO check if this is the total time passed since start of training
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
        return 'DoubleDQN'

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
                    a = self.get_action(s, 0)
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

        #TODO: log this somehow
        return steps, rewards, decisions, policies

    def checkpoint(self, filepath: str):
        torch.save(self._q.state_dict(), os.path.join(filepath, 'Q'))

    def load(self, filepath: str):
        self._q.load_state_dict(torch.load(os.path.join(filepath, 'Q')))

