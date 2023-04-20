# TODO: Can we sweep over envs here? Do we want to?
#DQN on classic control
python run_mighty.sh --config smac -m env=CartPole-v1 search_space=dqn_gym_classic algorithm=dqn
python run_mighty.sh --config smac -m env=MountainCar-v0 search_space=dqn_gym_classic algorithm=dqn

#SAC on classic control
python run_mighty.sh --config smac -m env=LunarLander-v2 search_space=sac_gym_classic algorithm=sac env_kwargs={'continuous': true}
python run_mighty.sh --config smac -m env=MountainCarContinuous-v0 search_space=sac_gym_classic algorithm=sac
python run_mighty.sh --config smac -m env=Pendulum-v1 search_space=sac_gym_classic algorithm=sac

#PPO on classic control
python run_mighty.sh --config smac -m env=LunarLander-v2 search_space=ppo_gym_classic algorithm=ppo env_kwargs={'continuous': true}
python run_mighty.sh --config smac -m env=MountainCarContinuous-v0 search_space=ppo_gym_classic algorithm=ppo
python run_mighty.sh --config smac -m env=Pendulum-v1 search_space=ppo_gym_classic algorithm=ppo

#DQN on DACBench
#TODO: setup env_kwarg parsing for DACBench
python run_mighty.sh --config smac -m env=SigmoidBenchmark env_wrappers=[dacbench.wrappers.MultiDiscreteWrapper] search_space=dqn_dacbench_toy algorithm=dqn 
python run_mighty.sh --config smac -m env=SigmoidBenchmark env_wrappers=[dacbench.wrappers.MultiDiscreteWrapper] search_space=dqn_dacbench_toy algorithm=dqn 
python run_mighty.sh --config smac -m env=FastDownwardBenchmark search_space=dqn_dacbench_fd algorithm=dqn