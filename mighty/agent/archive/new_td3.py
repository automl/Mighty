import os
import argparse

import numpy as np
import torch
import copy

from torch.utils.tensorboard import SummaryWriter
from mighty.env.env_handling import DACENV

from salina import get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.rl.replay_buffer import ReplayBuffer

import gym
import torch
import torch.nn as nn
import torch.nn.init as init
from gym.wrappers import TimeLimit

from salina import TAgent, instantiate_class
from salina.agents import Agents
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch


def make_atari_env(**env_args):
    e = make_atari(env_args["env_name"])
    e = wrap_deepmind(e)
    e = wrap_pytorch(e)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def make_gym_env(**env_args):
    e = gym.make(env_args["env_name"])
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class ActionMLPAgent(TAgent):
    def __init__(self, env, n_layers, hidden_size):
        super().__init__()
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size] + list(hidden_sizes) + [output_size],
            activation=nn.ReLU,
        )

    def forward(self, t, epsilon, epsilon_clip=100000, **kwargs):
        input = self.get(("env/env_obs", t))
        action = self.fc(input)
        action = torch.tanh(action)
        s = action.size()
        noise = torch.randn(*s, device=action.device) * epsilon
        noise = torch.clip(noise, min=-epsilon_clip, max=epsilon_clip)
        action = action + noise
        action = torch.clip(action, min=-1.0, max=1.0)
        self.set(("action", t), action)


class QMLPAgent(TAgent):
    def __init__(self, env, n_layers, hidden_size):
        super().__init__()
        env = instantiate_class(env)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        hidden_sizes = [hidden_size for _ in range(n_layers)]
        self.fc = mlp(
            [input_size + output_size] + list(hidden_sizes) + [1],
            activation=nn.ReLU,
        )

    def forward(self, t, detach_action=False, **kwargs):
        input = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_action:
            action = action.detach()
        x = torch.cat([input, action], dim=1)
        q = self.fc(x)
        self.set(("q", t), q)


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


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
        args: argparse.Namespace = None,  # algorithm args
        n_processes: int = 1,
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
        self.model_dir = os.path.join(self.output_dir, "models")

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
            self.writer.add_scalar("batch_size/Hyperparameter", self.batch_size)
            self.writer.add_scalar("gamma/Hyperparameter", self.gamma)

        self.c_step = 0
        self.epoch = 0
        self.n_interactions = 0
        self.iteration = 0

        q_class = args.q_agent["classname"]
        action_class = args.action_agent["classname"]
        del args.q_agent["classname"]
        del args.action_agent["classname"]

        self.q_agent_1 = q_class(**args.q_agent)
        self.q_agent_2 = q_class(**args.q_agent)
        self.q_agent_2.apply(weight_init)
        self.action_agent = action_class(**args.action_agent)

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
            n_steps=20,
            epsilon=1.0,
        )
        self.acq_remote_agent.seed(self.env_seed)

        # == Setting up the training agents
        self.train_temporal_q_agent_1 = TemporalAgent(self.q_agent_1)
        self.train_temporal_q_agent_2 = TemporalAgent(self.q_agent_2)
        self.train_temporal_action_agent = TemporalAgent(self.action_agent)
        self.train_temporal_q_target_agent_1 = TemporalAgent(self.q_target_agent_1)
        self.train_temporal_q_target_agent_2 = TemporalAgent(self.q_target_agent_2)
        self.train_temporal_action_target_agent = TemporalAgent(
            self.action_target_agent
        )

        self.train_temporal_q_agent_1.to(self.loss_device)
        self.train_temporal_q_agent_2.to(self.loss_device)
        self.train_temporal_action_agent.to(self.loss_device)
        self.train_temporal_q_target_agent_1.to(self.loss_device)
        self.train_temporal_q_target_agent_2.to(self.loss_device)
        self.train_temporal_action_target_agent.to(self.loss_device)

        print("targets set up")
        self.acq_remote_agent(
            self.acq_workspace,
            t=0,
            n_steps=1,  # self.n_timesteps,
            epsilon=self.action_noise,
        )
        print("workspace set up")
        # == Setting up & initializing the replay buffer for DQN
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.replay_buffer.put(self.acq_workspace, time_size=self.buffer_time_size)
        print("replay buffer set up")

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
        print("optimizer initialized")

    def save_replay_buffer(self, path):
        # TODO
        pass

    def load_replay_buffer(self, path):
        # TODO
        pass

    # TODO: save and load model

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """
        Simple helper to get action based on observation x
        """
        for a in self.acq_remote_agent.get_by_name("action_agent"):
            a.load_state_dict(_state_dict(self.action_agent, "cpu"))
        # TODO: what is this called?
        return self.acq_workspace["env/action"]

    def step(self):
        print("starting step")
        # TODO: this is pretty long, mabye separate into rollout and update?
        for a in self.acq_remote_agent.get_by_name("action_agent"):
            a.load_state_dict(_state_dict(self.action_agent, "cpu"))

        self.acq_workspace.copy_n_last_steps(self.overlapping_timesteps)
        print("Before remote agent")
        self.acq_remote_agent(
            self.acq_workspace,
            t=self.overlapping_timesteps,
            n_steps=self.n_timesteps - self.overlapping_timesteps,
            epsilon=self.action_noise,
        )
        print("after remote agent")
        self.replay_buffer.put(self.acq_workspace, time_size=self.buffer_time_size)

        print("overlap steps copied")
        done, creward = self.acq_workspace["env/done", "env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            self.logger.add_scalar("monitor/reward", creward.mean().item(), self.epoch)
        self.logger.add_scalar(
            "monitor/replay_buffer_size", self.replay_buffer.size(), self.epoch
        )

        self.n_interactions += (
            self.acq_workspace.time_size() - self.overlapping_timesteps
        ) * self.acq_workspace.batch_size()
        self.logger.add_scalar(
            "monitor/n_interactions", self.n_interactions, self.epoch
        )

        batch_size = self.batch_size
        self.replay_workspace = self.replay_buffer.get(batch_size).to(self.loss_device)
        done, reward = self.replay_workspace["env/done", "env/reward"]

        print("experience collected")
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
        target = reward[1:] + self.gamma * (1.0 - done[1:].float()) * q_target[1:]

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

        if self.c_step % self.policy_delay:
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
                self.logger.add_scalar(
                    "monitor/grad_norm_action", n.item(), self.iteration
                )

            self.logger.add_scalar("loss/q_loss", loss.item(), self.iteration)
            self.optimizer_action.step()

            tau = self.update_target_tau
            soft_update_params(self.q_agent_1, self.q_target_agent_1, tau)
            soft_update_params(self.q_agent_2, self.q_target_agent_2, tau)
            soft_update_params(self.action_agent, self.action_target_agent, tau)

        self.iteration += 1


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def run():
    import gym
    from salina import logger

    output_dir = "./test"
    logger = logger.TFLogger(log_dir=output_dir, modulo=100, verbose=True)
    env = {"classname": "gym.make", "id": "Pendulum-v1"}
    q_agent_1 = {"classname": QMLPAgent, "hidden_size": 256, "n_layers": 2, "env": env}
    action_agent = {
        "classname": ActionMLPAgent,
        "hidden_size": 256,
        "n_layers": 2,
        "env": env,
    }
    args = AttrDict(
        {
            "q_agent": q_agent_1,
            "action_agent": action_agent,
            "gamma": 0.99,
            "target_noise": 0.2,
            "action_noise": 0.1,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "burning_timesteps": 0,
            "clip_grad": 2,
            "optimizer": {"classname": "torch.optim.Adam", "lr": 0.0001},
            "n_envs": 2,
            "overlapping_timesteps": 1,
            "buffer_time_size": 1,
            "buffer_size": 1000000,
            "loss_device": "cpu",
            "n_timesteps": 50,
            "update_target_tau": 0.005,
            "batch_size": 8,
        }
    )
    agent = SalinaTD3Agent(
        env,
        env_seed=0,
        logger=logger,
        output_dir=output_dir,
        log_tensorboard=False,
        args=args,
    )
    agent.step()
