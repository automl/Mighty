# TODO: Can we sweep over envs here? Do we want to?
#DQN on classic control
python mighty/run_mighty.py -m env=CartPole-v1 +search_space=dqn_gym_classic algorithm=dqn 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=MountainCar-v0 +search_space=dqn_gym_classic algorithm=dqn 'num_steps=25e4' +cluster=local --config-name smac

#SAC on classic control
python mighty/run_mighty.py -m env=LunarLander-v2 +search_space=sac_gym_classic algorithm=sac env_kwargs={'continuous': true} 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=MountainCarContinuous-v0 +search_space=sac_gym_classic algorithm=sac 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=Pendulum-v1 +search_space=sac_gym_classic algorithm=sac 'num_steps=25e4' +cluster=local --config-name smac

#PPO on classic control
python mighty/run_mighty.py -m env=LunarLander-v2 +search_space=ppo_gym_classic algorithm=ppo env_kwargs={'continuous': true} 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=MountainCarContinuous-v0 +search_space=ppo_gym_classic algorithm=ppo 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=Pendulum-v1 +search_space=ppo_gym_classic algorithm=ppo 'num_steps=25e4' +cluster=local --config-name smac

#DQN on DACBench
python mighty/run_mighty.py -m env=SigmoidBenchmark env_wrappers=[dacbench.wrappers.MultiDiscreteWrapper] +search_space=dqn_dacbench_toy algorithm=dqn env_kwargs={"benchmark": True, "dimension": 1} 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=SigmoidBenchmark env_wrappers=[dacbench.wrappers.MultiDiscreteWrapper] +search_space=dqn_dacbench_toy algorithm=dqn env_kwargs={"benchmark": True, "dimension": 5} 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=FastDownwardBenchmark +search_space=dqn_dacbench_fd algorithm=dqn 'num_steps=25e4' +cluster=local --config-name smac