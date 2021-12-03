import os
import argparse

import numpy as np
import torch
import copy

from torch.utils.tensorboard import SummaryWriter
from mighty.env.env_handling import DACENV

from salina import get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gym import AutoResetGymAgent, GymAgent
from salina.rl.replay_buffer import ReplayBuffer
from salina_examples import weight_init


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


class SalinaTD3Agent:
    def __init__(
            self,
            env: DACENV,
            env_seed: int,
            logger,
            output_dir: str,
            log_tensorboard: bool = True,
            args: argparse.Namespace = None, #algorithm args
            n_processes: int = 8,
    ):
        """
        Initialize the TD3 Agent

        args overrides all hyperparams if given.

        :param gamma: discount factor
        :param env: environment to train on
        :param logger: logging functionality of some sort
        #FIXME
        """
        self.env = env
        self.env_seed = env_seed
        self.logger = logger
        self.output_dir = output_dir
        self.model_dir = os.path.join(self.output_dir, 'models')

        self.action_noise = args.action_noise
        self.clip_grad = args.clip_grad
        self.loss_device = args.loss_device
        self.gamma = args.gamma
        self.n_timesteps = args.n_timesteps
        self.buffer_size = args.buffer_size
        self.buffer_time_size = args.buffer_time_size
        self.overlapping_timesteps = args.overlapping_timesteps
        self.policy_delay = args.policy_delay
        self.burning_timesteps = args.burning_timesteps
        self.update_target_tau = args.update_target_tau
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.batch_size = args.batch_size

        self.writer = None
        if log_tensorboard:
            # TODO write out all the other hyperparameters
            self.writer = SummaryWriter(self.output_dir)
            self.writer.add_scalar('batch_size/Hyperparameter', self.batch_size)
            self.writer.add_scalar('gamma/Hyperparameter', self.gamma)

        self.step = 0
        self.epoch = 0
        self.n_interactions = 0
        self.iteration = 0

        self.q_agent_1 = instantiate_class(args.q_agent)
        self.q_agent_2 = instantiate_class(args.q_agent)
        self.q_agent_2.apply(weight_init)
        self.action_agent = instantiate_class(args.action_agent)

        self.action_agent.set_name("action_agent")
        self.env_agent = AutoResetGymAgent(
            get_class(env),
            get_arguments(env),
            n_envs=int(args.n_envs / n_processes),
        )
        self.q_target_agent_1 = copy.deepcopy(self.q_agent_1)
        self.q_target_agent_2 = copy.deepcopy(self.q_agent_2)
        self.action_target_agent = copy.deepcopy(self.action_agent)

        self.acq_action_agent = copy.deepcopy(self.action_agent)
        self.acq_agent = TemporalAgent(Agents(self.env_agent, self.acq_action_agent))
        self.acq_remote_agent, self.acq_workspace = NRemoteAgent.create(
            self.acq_agent,
            num_processes=n_processes,
            t=0,
            n_steps=self.n_timesteps,
            epsilon=1.0,
        )
        self.acq_remote_agent.seed(self.env_seed)

        # == Setting up the training agents
        self.train_temporal_q_agent_1 = TemporalAgent(self.q_agent_1)
        self.train_temporal_q_agent_2 = TemporalAgent(self.q_agent_2)
        self.train_temporal_action_agent = TemporalAgent(self.action_agent)
        self.train_temporal_q_target_agent_1 = TemporalAgent(self.q_target_agent_1)
        self.train_temporal_q_target_agent_2 = TemporalAgent(self.q_target_agent_2)
        self.train_temporal_action_target_agent = TemporalAgent(self.action_target_agent)

        self.train_temporal_q_agent_1.to(self.loss_device)
        self.train_temporal_q_agent_2.to(self.loss_device)
        self.train_temporal_action_agent.to(self.loss_device)
        self.train_temporal_q_target_agent_1.to(self.loss_device)
        self.train_temporal_q_target_agent_2.to(self.loss_device)
        self.train_temporal_action_target_agent.to(self.loss_device)

        self.acq_remote_agent(
            self.acq_workspace,
            t=0,
            n_steps=self.n_timesteps,
            epsilon=self.action_noise,
        )

        # == Setting up & initializing the replay buffer for DQN
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.replay_buffer.put(self.acq_workspace, time_size=self.buffer_time_size)

        optimizer_args = get_arguments(args.optimizer)
        self.optimizer_q_1 = get_class(args.optimizer)(
            self.q_agent_1.parameters(), **optimizer_args
        )
        self.optimizer_q_2 = get_class(args.optimizer)(
            self.q_agent_2.parameters(), **optimizer_args
        )
        self.optimizer_action = get_class(args.optimizer)(
            self.action_agent.parameters(), **optimizer_args
        )

    def save_replay_buffer(self, path):
        #TODO
        pass

    def load_replay_buffer(self, path):
        #TODO
        pass

    #TODO: save and load model

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """
        Simple helper to get action based on observation x
        """
        for a in self.acq_remote_agent.get_by_name("action_agent"):
            a.load_state_dict(_state_dict(self.action_agent, "cpu"))
        #TODO: what is this called?
        return self.acq_workspace["env/action"]

    def step(self):
        #TODO: this is pretty long, mabye separate into rollout and update?
        for a in self.acq_remote_agent.get_by_name("action_agent"):
            a.load_state_dict(_state_dict(self.action_agent, "cpu"))

        self.acq_workspace.copy_n_last_steps(self.overlapping_timesteps)
        self.acq_remote_agent(
            self.acq_workspace,
            t=self.overlapping_timesteps,
            n_steps=self.n_timesteps - self.overlapping_timesteps,
            epsilon=self.action_noise,
        )
        self.replay_buffer.put(self.acq_workspace, time_size=self.buffer_time_size)

        done, creward = self.acq_workspace["env/done", "env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            self.logger.add_scalar("monitor/reward", creward.mean().item(), self.epoch)
        self.logger.add_scalar("monitor/replay_buffer_size", self.replay_buffer.size(), self.epoch)

        self.n_interactions += (
            self.acq_workspace.time_size() - self.overlapping_timesteps
        ) * self.acq_workspace.batch_size()
        self.logger.add_scalar("monitor/n_interactions", self.n_interactions, self.epoch)

        batch_size = self.batch_size
        self.replay_workspace = self.replay_buffer.get(batch_size).to(
                self.loss_device)
        done, reward = self.replay_workspace["env/done", "env/reward"]

        self.train_temporal_q_agent_1(
            self.replay_workspace,
            t=0,
            n_steps=self.buffer_time_size,
            detach_action=True,
        )
        q_1 = self.replay_workspace["q"].squeeze(-1)
        self.train_temporal_q_agent_2(
                self.replay_workspace,
                t=0,
                n_steps=self.buffer_time_size,
                detach_action=True,
            )
        q_2 = self.replay_workspace["q"].squeeze(-1)

        with torch.no_grad():
            self.train_temporal_action_target_agent(
                    self.replay_workspace,
                    t=0,
                    n_steps=self.buffer_time_size,
                    epsilon=self.target_noise,
                    epsilon_clip=self.noise_clip,
                )
            self.train_temporal_q_target_agent_1(
                    self.replay_workspace,
                    t=0,
                    n_steps=self.buffer_time_size,
                )
            q_target_1 = self.replay_workspace["q"]
            self.train_temporal_q_target_agent_2(
                    self.replay_workspace,
                    t=0,
                    n_steps=self.buffer_time_size,
                )
            q_target_2 = self.replay_workspace["q"]

        q_target = torch.min(q_target_1, q_target_2).squeeze(-1)
        target = (
                reward[1:]
                + self.gamma
                * (1.0 - done[1:].float())
                * q_target[1:]
            )

        td_1 = q_1[:-1] - target
        td_2 = q_2[:-1] - target
        error_1 = td_1 ** 2
        error_2 = td_2 ** 2

        burning = torch.zeros_like(error_1)
        burning[self.burning_timesteps :] = 1.0
        error_1 = error_1 * burning
        error_2 = error_2 * burning
        error = error_1 + error_2
        loss = error.mean()
        self.logger.add_scalar("loss/td_loss_1", error_1.mean().item(), self.iteration)
        self.logger.add_scalar("loss/td_loss_2", error_2.mean().item(), self.iteration)
        self.optimizer_q_1.zero_grad()
        self.optimizer_q_2.zero_grad()
        loss.backward()

        if self.clip_grad > 0:
            n = torch.nn.utils.clip_grad_norm_(
                self.q_agent_1.parameters(), self.clip_grad
            )
            self.logger.add_scalar("monitor/grad_norm_q_1", n.item(), self.iteration)
            n = torch.nn.utils.clip_grad_norm_(
                self.q_agent_2.parameters(), self.clip_grad
            )
            self.logger.add_scalar("monitor/grad_norm_q_2", n.item(), self.iteration)

        self.optimizer_q_1.step()
        self.optimizer_q_2.step()

        if self.step % self.policy_delay:
            self.train_temporal_action_agent(
                self.replay_workspace,
                epsilon=0.0,
                t=0,
                n_steps=self.buffer_time_size,
            )
            self.train_temporal_q_agent_1(
                self.replay_workspace,
                t=0,
                n_steps=self.buffer_time_size,
            )
            q = self.replay_workspace["q"].squeeze(-1)
            burning = torch.zeros_like(q)
            burning[self.burning_timesteps :] = 1.0
            q = q * burning
            q = q * (1.0 - done.float())
            self.optimizer_action.zero_grad()
            loss = -q.mean()
            loss.backward()

            if self.clip_grad > 0:
                n = torch.nn.utils.clip_grad_norm_(
                    self.action_agent.parameters(), self.clip_grad
                )
                self.logger.add_scalar("monitor/grad_norm_action", n.item(), self.iteration)

            self.logger.add_scalar("loss/q_loss", loss.item(), self.iteration)
            self.optimizer_action.step()

            tau = self.update_target_tau
            soft_update_params(self.q_agent_1, self.q_target_agent_1, tau)
            soft_update_params(self.q_agent_2, self.q_target_agent_2, tau)
            soft_update_params(self.action_agent, self.action_target_agent, tau)

        self.iteration += 1

def run():
    import gym
    output_dir = "./test"
    logger = salina.logger.TFLogger(log_dir=output_dir, modulo=100, verbose=True)
    env={"classname": "gym.make", "env": "CartPole-v0"}
    q_agent_1 = salina_examples.rl.td3.agents.QMLPAgent(env=env, hidden_size=256, n_layers=2)
    action_agent = salina_examples.rl.td3.agents.ActionMLPAgent(env=env, hidden_size=256, n_layers=2)
    args = {"q_agent": q_agent_1,
            "action_agent": action_agent,
            "gamma": 0.99,
            "target_noise": 0.2,
            "action_noise": 0.1,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "burning_timesteps": 0,
            "clip_grad": 2,
            "optimizer": torch.optim.Adam(0.0001),
            "n_envs": 1,
            "overlapping_timesteps": 1,
            "buffer_time_size": 2,
            "buffer_size": 1000000
            }
    agent = SalinaTD3Agent(gym.make("CartPole-v0"), env_seed=0, logger, output_dir=output_dir,
                           log_tensorboard=False, args)
    agent.step()
