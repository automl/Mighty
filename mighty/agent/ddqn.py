import os
import time
import json

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
from torch.autograd import Variable

from mighty.utils.replay_buffer import ReplayBuffer
from mighty.utils.value_function import FullyConnectedQ
from mighty.utils.weight_updates import soft_update


class DDQN:
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

    def __init__(self, state_dim: int, action_dim: int, gamma: float,
                 env: gym.Env, eval_env: gym.Env, train_eval_env: gym.Env = None, vision: bool = False,
                 out_dir: str = None):
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
            self._q = FullyConnectedQ(state_dim, action_dim).to(self.device)
            self._q_target = FullyConnectedQ(state_dim, action_dim).to(self.device)
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
        self.out_dir = out_dir

    def save_rpb(self, path):
        self._replay_buffer.save(path)

    def load_rpb(self, path):
        self._replay_buffer.load(path)

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(self.tt(x)).detach().numpy())
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
        :param eval_eps: number of episodes to run for evaluation
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
                ns, r, d, _ = self._env.step(a)
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

                    with open(os.path.join(self.out_dir, 'eval_scores.json'), 'a+') as out_fh:
                        json.dump(eval_stats, out_fh)
                        out_fh.write('\n')
                    with open(os.path.join(self.out_dir, 'eval_scores_per_inst.json'), 'a+') as out_fh:
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

                        with open(os.path.join(self.out_dir, 'train_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                        with open(os.path.join(self.out_dir, 'train_scores_per_inst.json'), 'a+') as out_fh:
                            json.dump(per_inst_stats, out_fh)
                            out_fh.write('\n')
                    print('End Evaluation')
                ########### End Evaluation

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    self._replay_buffer.random_next_batch(64)
                batch_states = self.tt(batch_states)
                batch_actions = self.tt(batch_actions)
                batch_next_states = self.tt(batch_next_states)
                batch_rewards = self.tt(batch_rewards)
                batch_terminal_flags = self.tt(batch_terminal_flags)

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

        with open(os.path.join(self.out_dir, 'eval_scores.json'), 'a+') as out_fh:
            json.dump(eval_stats, out_fh)
            out_fh.write('\n')
        with open(os.path.join(self.out_dir, 'eval_scores_per_inst.json'), 'a+') as out_fh:
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

            with open(os.path.join(self.out_dir, 'train_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')
            with open(os.path.join(self.out_dir, 'train_scores_per_inst.json'), 'a+') as out_fh:
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
        with torch.no_grad():
            for e in range(episodes):
                # this_env.instance_index = this_env.instance_index % 10  # for faster debuggin on only 10 insts
                print(f'Eval Episode {e} of {episodes}')
                ed, es, er = 0, 0, 0

                s = this_env.reset()
                # policy = [float(this_env.current_lr.numpy()[0])]
                for _ in count():
                    a = self.get_action(s, 0)
                    ed += 1

                    ns, r, d, _ = this_env.step(a)
                    er += r
                    es += 1
                    if es >= max_env_time_steps or d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)
                policies.append(None)

        return steps, rewards, decisions, policies

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q'))

    def load(self, path):
        self._q.load_state_dict(torch.load(os.path.join(path, 'Q')))


if __name__ == "__main__":
    import argparse

    from dacbench.benchmarks import SigmoidBenchmark
    from dacbench.wrappers import ObservationWrapper

    from mighty.iohandling.experiment_tracking import prepare_output_dir
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
    benchmark = SigmoidBenchmark()
    val_bench = SigmoidBenchmark()

    env = benchmark.get_benchmark(seed=args.seed)
    eval_env = val_bench.get_benchmark(seed=args.seed)

    # Setup agent
    state_dim = env.observation_space.shape[0]

    agent = DDQN(state_dim, env.action_space.n, gamma=0.99, env=env, eval_env=eval_env, out_dir=out_dir)

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
