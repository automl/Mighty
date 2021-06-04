import os
from pathlib import Path
import numpy as np

from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import PerformanceTrackingWrapper

from mighty.agent.ddqn import DDQNAgent
from mighty.iohandling.experiment_tracking import prepare_output_dir
from mighty.utils.logger import Logger

from utils.scenario_config_parser import ScenarioConfigParser

if __name__ == "__main__":
    parser = ScenarioConfigParser()
    args = parser.parse()

    if not args.load_model:
        out_dir = prepare_output_dir(args, user_specified_dir=args.out_dir)
        
    train_logger = Logger(
        experiment_name=f"sigmoid_example_s{args.seed}",
        output_path=Path(out_dir),
        step_write_frequency=None,
        episode_write_frequency=10,
    )
    performance_logger = train_logger.add_module(PerformanceTrackingWrapper, "train_performance")
    
    #TODO: this should not be separate! Extend the logger to support multiple envs
    eval_logger = Logger(
            experiment_name=f"sigmoid_example_s{args.seed}",
            output_path=Path(out_dir),
            step_write_frequency=None,
            episode_write_frequency=1,
    )
    eval_module = eval_logger.add_module(PerformanceTrackingWrapper, "eval_performance")


    if not args.load_model:
        out_dir = prepare_output_dir(args, user_specified_dir=args.out_dir,
                                     subfolder_naming_scheme=args.out_dir_suffix)

    # create the benchmark
    benchmark = SigmoidBenchmark()
    # benchmark.config['instance_set_path'] = '../instance_sets/sigmoid/sigmoid_1D3M_train.csv'
    # benchmark.set_action_values((2, ))
    val_bench = SigmoidBenchmark()
    # val_bench.config['instance_set_path'] = '../instance_sets/sigmoid/sigmoid_1D3M_train.csv'
    # val_bench.set_action_values((2, ))

    env = benchmark.get_benchmark(seed=args.seed)
    env = PerformanceTrackingWrapper(env, logger=performance_logger)
    train_logger.set_env(env)
    train_logger.set_additional_info(seed=args.seed)

    eval_env = val_bench.get_benchmark(seed=args.seed)
    eval_env = PerformanceTrackingWrapper(eval_env, logger=eval_module)
    eval_logger.set_env(env)
    eval_logger.set_additional_info(seed=args.seed)
    # Setup agent
    #state_dim = env.observation_space.shape[0]
    agent = DDQNAgent(gamma=0.99, env=env, env_eval=eval_env, eval_logger=eval_logger, epsilon=args.epsilon, logger=train_logger, batch_size=64)
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
                    max_train_time_steps=args.max_train_steps)
        os.mkdir(os.path.join(train_logger.log_dir, 'final'))
        agent.checkpoint(os.path.join(train_logger.log_dir, 'final'))
        agent.save_replay_buffer(os.path.join(train_logger.log_dir, 'final'))
    else:
        print('#'*80)
        print(f'Loading {agent} from {args.load_model}')
        print('#'*80)
        agent.load(args.load_model)
        steps, rewards, decisions = agent.eval(1, 100000)
        np.save(os.path.join(out_dir, 'eval_results.npy'), [steps, rewards, decisions])
    #TODO: this should go in a general cleanup function
    agent.writer.close()
    train_logger.close()
    eval_logger.close()

