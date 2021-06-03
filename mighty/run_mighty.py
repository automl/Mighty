import argparse
import os
from pathlib import Path
import numpy as np

from dacbench.benchmarks import SigmoidBenchmark
from dacbench.logger import Logger
from dacbench.wrappers import PerformanceTrackingWrapper

from mighty.agent.ddqn import DDQNAgent
from mighty.iohandling.experiment_tracking import prepare_output_dir

def parse_arguments():
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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    if not args.load_model:
        out_dir = prepare_output_dir(args, user_specified_dir=args.out_dir,
                                     subfolder_naming_scheme=args.out_dir_suffix)
    logger = Logger(
        experiment_name="sigmoid_example",
        output_path=Path(out_dir),
        step_write_frequency=None,
        episode_write_frequency=None,
    )
    performance_logger = logger.add_module(PerformanceTrackingWrapper)

    # create the benchmark
    benchmark = SigmoidBenchmark()
    # benchmark.config['instance_set_path'] = '../instance_sets/sigmoid/sigmoid_1D3M_train.csv'
    # benchmark.set_action_values((2, ))
    val_bench = SigmoidBenchmark()
    # val_bench.config['instance_set_path'] = '../instance_sets/sigmoid/sigmoid_1D3M_train.csv'
    # val_bench.set_action_values((2, ))

    env = benchmark.get_benchmark(seed=args.seed)
    env = PerformanceTrackingWrapper(env, logger=performance_logger)
    logger.set_env(env)
    logger.set_additional_info(seed=args.seed)

    eval_env = val_bench.get_benchmark(seed=args.seed)

    # Setup agent
    #state_dim = env.observation_space.shape[0]
    agent = DDQNAgent(gamma=0.99, env=env, env_eval=eval_env, epsilon=args.epsilon, logger=logger, batch_size=64)
    #TODO: parse args additional hooks into agent

    episodes = args.episodes
    max_env_time_steps = args.env_max_steps
    epsilon = args.epsilon

    if args.load_model is None:
        print('#'*80)
        print(f'Using agent type "{agent}" to learn')
        print('#'*80)
        num_eval_episodes = 100  # 10  # use 10 for faster debugging but also set it in the eval method above
        agent.train(episodes, epsilon, max_env_time_steps, num_eval_episodes, args.eval_after_n_steps,
                    max_train_time_steps=args.training_steps)
        os.mkdir(os.path.join(out_dir, 'final'))
        agent.checkpoint(os.path.join(out_dir, 'final'))
        agent.save_replay_buffer(os.path.join(out_dir, 'final'))
    else:
        print('#'*80)
        print(f'Loading {agent} from {args.load_model}')
        print('#'*80)
        agent.load(args.load_model)
        steps, rewards, decisions = agent.eval(1, 100000)
        np.save(os.path.join(out_dir, 'eval_results.npy'), [steps, rewards, decisions])
