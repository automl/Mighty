import pickle

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
from collections import namedtuple
import time

from ddqn_utils import *
from dacbench.benchmarks import SGDBenchmark
from dacbench.wrappers import ObservationWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    See soft_update
    """
    soft_update(target, source, 1.0)


class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)

    def save(self, path):
        with open(os.path.join(path, 'rpb.pkl'), 'wb') as fh:
            pickle.dump(list(self._data), fh)

    def load(self, path):
        with open(os.path.join(path, 'rpb.pkl'), 'rb') as fh:
            data = pickle.load(fh)
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data.states = data[0]
        self._data.actions = data[1]
        self._data.next_states = data[2]
        self._data.rewards = data[3]
        self._data.terminal_flags = data[4]
        self._size = len(data[0])


class DQN:
    """
    Simple double DQN Agent
    """

    def __init__(self, state_dim: int, action_dim: int, gamma: float,
                 env: gym.Env, eval_env: gym.Env, train_eval_env: gym.Env = None, vision: bool = False,
                 discrete_actions: list = None):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param eval_env: environment to evaluate on with training data
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        if not vision:  # For featurized states
            self._q = Q(state_dim, action_dim).to(device)
            self._q_target = Q(state_dim, action_dim).to(device)
        else:  # For image states, i.e. Atari
            raise NotImplementedError

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._env = env
        self._eval_env = eval_env
        self._train_eval_env = train_eval_env
        self.discrete_actions = discrete_actions

    def save_rpb(self, path):
        self._replay_buffer.save(path)

    def load_rpb(self, path):
        self._replay_buffer.load(path)

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        :return:
        """
        total_steps = 0
        start_time = time.time()
        print(f'Start training at {start_time}')
        for e in range(episodes):
            # print('\033c')
            # print('\x1bc')
            if e % 100 == 0:
                print("%s/%s" % (e + 1, episodes))
            s = self._env.reset()
            for t in range(max_env_time_steps):
                a = self.get_action(s, epsilon)
                current_lr = self._env.current_lr.numpy()[0]
                env_a = np.clip(self.discrete_actions[a] * current_lr, 10**-7, 100)
                ns, r, d, _ = self._env.step(env_a)
                total_steps += 1

                ########### Begin Evaluation
                if (total_steps % eval_every_n_steps) == 0:
                    print('Begin Evaluation')
                    eval_s, eval_r, eval_d, pols = self.eval(eval_eps, max_env_time_steps)
                    eval_stats = dict(
                        elapsed_time=time.time() - start_time,
                        training_steps=total_steps,
                        training_eps=e,
                        avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                        avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                        avg_rew_per_eval_ep=float(np.mean(eval_r)),
                        std_rew_per_eval_ep=float(np.std(eval_r)),
                        eval_eps=eval_eps
                    )
                    per_inst_stats = dict(
                            # eval_insts=self._train_eval_env.instances,
                            reward_per_isnts=eval_r,
                            steps_per_insts=eval_s,
                            policies=pols
                        )

                    with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
                        json.dump(eval_stats, out_fh)
                        out_fh.write('\n')
                    with open(os.path.join(out_dir, 'eval_scores_per_inst.json'), 'a+') as out_fh:
                        json.dump(per_inst_stats, out_fh)
                        out_fh.write('\n')

                    if self._train_eval_env is not None:
                        eval_s, eval_r, eval_d, pols = self.eval(eval_eps, max_env_time_steps, train_set=True)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                            avg_rew_per_eval_ep=float(np.mean(eval_r)),
                            std_rew_per_eval_ep=float(np.std(eval_r)),
                            eval_eps=eval_eps
                        )
                        per_inst_stats = dict(
                            # eval_insts=self._train_eval_env.instances,
                            reward_per_isnts=eval_r,
                            steps_per_insts=eval_s,
                            policies=pols
                        )

                        with open(os.path.join(out_dir, 'train_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                        with open(os.path.join(out_dir, 'train_scores_per_inst.json'), 'a+') as out_fh:
                            json.dump(per_inst_stats, out_fh)
                            out_fh.write('\n')
                    print('End Evaluation')
                ########### End Evaluation

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    self._replay_buffer.random_next_batch(64)

                ########### Begin double Q-learning update
                target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                         self._q_target(batch_next_states)[torch.arange(64).long(), torch.argmax(
                             self._q(batch_next_states), dim=1)]
                current_prediction = self._q(batch_states)[torch.arange(64).long(), batch_actions.long()]

                loss = self._loss_function(current_prediction, target.detach())

                self._q_optimizer.zero_grad()
                loss.backward()
                self._q_optimizer.step()

                soft_update(self._q_target, self._q, 0.01)
                ########### End double Q-learning update

                if d:
                    break
                s = ns
                if total_steps >= max_train_time_steps:
                    break
            if total_steps >= max_train_time_steps:
                break

        # Final evaluation
        eval_s, eval_r, eval_d, pols = self.eval(eval_eps, max_env_time_steps)
        eval_stats = dict(
            elapsed_time=time.time() - start_time,
            training_steps=total_steps,
            training_eps=e,
            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
            avg_rew_per_eval_ep=float(np.mean(eval_r)),
            std_rew_per_eval_ep=float(np.std(eval_r)),
            eval_eps=eval_eps
        )
        per_inst_stats = dict(
                # eval_insts=self._train_eval_env.instances,
                reward_per_isnts=eval_r,
                steps_per_insts=eval_s,
                policies=pols
            )

        with open(os.path.join(out_dir, 'eval_scores.json'), 'a+') as out_fh:
            json.dump(eval_stats, out_fh)
            out_fh.write('\n')
        with open(os.path.join(out_dir, 'eval_scores_per_inst.json'), 'a+') as out_fh:
            json.dump(per_inst_stats, out_fh)
            out_fh.write('\n')

        if self._train_eval_env is not None:
            eval_s, eval_r, eval_d, pols = self.eval(eval_eps, max_env_time_steps, train_set=True)
            eval_stats = dict(
                elapsed_time=time.time() - start_time,
                training_steps=total_steps,
                training_eps=e,
                avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                avg_rew_per_eval_ep=float(np.mean(eval_r)),
                std_rew_per_eval_ep=float(np.std(eval_r)),
                eval_eps=eval_eps
            )
            per_inst_stats = dict(
                # eval_insts=self._train_eval_env.instances,
                reward_per_isnts=eval_r,
                steps_per_insts=eval_s,
                policies=pols
            )

            with open(os.path.join(out_dir, 'train_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')
            with open(os.path.join(out_dir, 'train_scores_per_inst.json'), 'a+') as out_fh:
                json.dump(per_inst_stats, out_fh)
                out_fh.write('\n')

    def __repr__(self):
        return 'DDQN'

    def eval(self, episodes: int, max_env_time_steps: int, train_set: bool = False):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        policies = []
        this_env = self._eval_env if not train_set else self._train_eval_env
        for e in range(episodes):
            # this_env.instance_index = this_env.instance_index % 10  # for faster debuggin on only 10 insts
            print(f'Eval Episode {e} of {episodes}')
            ed, es, er = 0, 0, 0

            s = this_env.reset()
            policy = [float(this_env.current_lr.numpy()[0])]
            for _ in count():
                with torch.no_grad():
                    a = self.get_action(s, 0)
                current_lr = this_env.current_lr.numpy()[0]
                env_a = np.clip(self.discrete_actions[a] * current_lr, 10**-7, 100)
                policy.append(float(env_a))
                ed += 1

                ns, r, d, _ = this_env.step(env_a)
                er += r
                es += 1
                if es >= max_env_time_steps or d:
                    break
                s = ns
            steps.append(es)
            rewards.append(er)
            decisions.append(ed)
            policies.append(policy)

        return steps, rewards, decisions, policies

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q'))

    def load(self, path):
        self._q.load_state_dict(torch.load(os.path.join(path, 'Q')))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('Online DQN training')
    parser.add_argument('--episodes', '-e',
                        default=100,
                        type=int,
                        help='Number of training episodes.')
    parser.add_argument('--training-steps', '-t',
                        default=1_000_000,
                        type=int,
                        help='Number of training episodes.')

    parser.add_argument('--out-dir',
                        default=None,
                        type=str,
                        help='Directory to save results. Defaults to tmp dir.')
    parser.add_argument('--out-dir-suffix',
                        default='seed',
                        type=str,
                        choices=['seed', 'time'],
                        help='Created suffix of directory to save results.')
    parser.add_argument('--seed', '-s',
                        default=12345,
                        type=int,
                        help='Seed')
    parser.add_argument('--eval-after-n-steps',
                        default=10 ** 3,
                        type=int,
                        help='After how many steps to evaluate')
    parser.add_argument('--env-max-steps',
                        default=200,
                        type=int,
                        help='Maximal steps in environment before termination.')
    parser.add_argument('--load-model', default=None)
    parser.add_argument('--agent-epsilon', default=0.2, type=float, help='Fixed epsilon to use during training',
                        dest='epsilon')

    # setup output dir
    args = parser.parse_args()

    if not args.load_model:
        out_dir = prepare_output_dir(args, user_specified_dir=args.out_dir,
                                     subfolder_naming_scheme=args.out_dir_suffix)

    # create the benchmark
    benchmark = SGDBenchmark()
    val_bench = SGDBenchmark(None, instance_set_path='../instance_sets/sgd/sgd_val.csv')

    env = benchmark.get_benchmark(seed=args.seed)
    eval_env = val_bench.get_benchmark(seed=args.seed)
    env = ObservationWrapper(env)
    eval_env = ObservationWrapper(eval_env)

    s = env.reset()

    # Setup agent
    state_dim = env.observation_space.shape[0]
    discrete_actions = [.1, .2, .5, 1, 2, 5, 10]

    agent = DQN(state_dim, len(discrete_actions), gamma=0.99, env=env, eval_env=eval_env,
                discrete_actions=discrete_actions)

    episodes = args.episodes
    max_env_time_steps = args.env_max_steps
    epsilon = args.epsilon

    if args.load_model is None:
        print('#'*80)
        print(f'Using agent type "{agent}" to learn')
        print('#'*80)
        num_eval_episodes = 100  # 10  # use 10 for faster debugging but also set it in the eval method above
        agent.train(episodes, max_env_time_steps, epsilon, num_eval_episodes, args.eval_after_n_steps,
                    max_train_time_steps=args.training_steps)
        os.mkdir(os.path.join(out_dir, 'final'))
        agent.save_model(os.path.join(out_dir, 'final'))
        agent.save_rpb(os.path.join(out_dir, 'final'))
    else:
        print('#'*80)
        print(f'Loading {agent} from {args.load_model}')
        print('#'*80)
        agent.load(args.load_model)
        steps, rewards, decisions = agent.eval(1, 100000)
        np.save(os.path.join(out_dir, 'eval_results.npy'), [steps, rewards, decisions])
