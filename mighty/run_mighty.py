import os
from pathlib import Path
import numpy as np
from rich import print

from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import PerformanceTrackingWrapper

from mighty.agent.factory import get_agent_class
from mighty.utils.logger import Logger

import importlib
import mighty.utils.main_parser
importlib.reload(mighty.utils.main_parser)

from omegaconf import DictConfig
import hydra


@hydra.main("./configs", "base")
def main(cfg: DictConfig):
    print(cfg)

    out_dir = os.getcwd()  # working directory changes to hydra.run.dir
    seed = cfg.seed

    logger = Logger(
        experiment_name=f"sigmoid_example_s{seed}",
        output_path=Path(out_dir),
        step_write_frequency=None,
        episode_write_frequency=10,
    )

    # if not args.load_model:
    #     out_dir = prepare_output_dir(args, user_specified_dir=args.out_dir,
    #                                  subfolder_naming_scheme=args.out_dir_suffix)

    # create the benchmark
    benchmark = SigmoidBenchmark()
    # benchmark.config['instance_set_path'] = '../instance_sets/sigmoid/sigmoid_1D3M_train.csv'
    # benchmark.set_action_values((2, ))
    val_bench = SigmoidBenchmark()
    # val_bench.config['instance_set_path'] = '../instance_sets/sigmoid/sigmoid_1D3M_train.csv'
    # val_bench.set_action_values((2, ))

    env = benchmark.get_benchmark(seed=seed)
    eval_env = val_bench.get_benchmark(seed=seed)

    performance_logger = logger.add_module(PerformanceTrackingWrapper, env, "train_performance")
    eval_logger = logger.add_module(PerformanceTrackingWrapper, eval_env, "eval_performance")
    env = PerformanceTrackingWrapper(env, logger=performance_logger)
    eval_env = PerformanceTrackingWrapper(eval_env, logger=eval_logger)

    logger.set_train_env(env)
    logger.set_eval_env(env)

    # Setup agent
    agent_class = get_agent_class(cfg.algorithm)
    args_agent = dict(cfg.algorithm_kwargs)  # {"lr": 0.001, "epsilon": 0.1}
    agent = agent_class(
        env=env,
        eval_env=eval_env,
        logger=logger,
        **args_agent,  # by using args we can build a general interface
    )

    #max_env_time_steps = args_agent.max_env_time_steps
    epsilon = args_agent["epsilon"]
    n_episodes_eval = len(eval_env.instance_set.keys())
    eval_every_n_steps = cfg.eval_every_n_steps
    #save_model_every_n_episodes = args.save_model_every_n_episodes

    if cfg.checkpoint is None:
        print('#' * 80)
        print(f'Using agent type "{agent}" to learn')
        print('#' * 80)
        num_eval_episodes = 100  # 10  # use 10 for faster debugging but also set it in the eval method above
        agent.train(
            n_steps=cfg.num_steps,
            n_episodes_eval=n_episodes_eval,
            eval_every_n_steps=eval_every_n_steps,
            #human_log_every_n_episodes=100,
            #save_model_every_n_episodes=save_model_every_n_episodes,
        )
        #TODO: integrate this into trainer
        #os.mkdir(os.path.join(logger.log_dir, 'final'))
        #agent.checkpoint(os.path.join(logger.log_dir, 'final'))
    else:
        raise NotImplementedError
        # print('#' * 80)
        # print(f'Loading {agent} from {args.load_model}')
        # print('#' * 80)
        # agent.load(args.load_model)
        # steps, rewards, decisions = agent.eval(1, 100000)
        # np.save(os.path.join(out_dir, 'eval_results.npy'), [steps, rewards, decisions])
    # TODO: this should go in a general cleanup function
    # TODO: should only happen if there is a writer to close
    #agent.writer.close()
    logger.close()


if __name__ == "__main__":
    main()
