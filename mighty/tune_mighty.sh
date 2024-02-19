# TODO: Can we sweep over envs here? Do we want to?

# Allocate a job first in a tmux session on the login node
# Start tmux session or attach to existing
tmux new -s mighty
tmux a -t mighty
# Allocate job
salloc -t 72:00:00 -c 11
ssh x
cd /scratch/hpc-prf-intexml/cbenjamins/repos/Mighty-DACS
conda activate /scratch/hpc-prf-intexml/cbenjamins/envs/mighty

#DQN on classic control
python mighty/run_mighty.py -m +environment=cartpole +search_space=dqn_gym_classic algorithm=dqn +cluster=local --config-name smac
python mighty/run_mighty.py -m +environment=mountaincar +search_space=dqn_gym_classic algorithm=dqn +cluster=local --config-name smac

#SAC on classic control
python mighty/run_mighty.py -m env=LunarLander-v2 +search_space=sac_gym_classic algorithm=sac env_kwargs={'continuous': true} 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=MountainCarContinuous-v0 +search_space=sac_gym_classic algorithm=sac 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=Pendulum-v1 +search_space=sac_gym_classic algorithm=sac 'num_steps=25e4' +cluster=local --config-name smac

#PPO on classic control
python mighty/run_mighty.py -m env=LunarLander-v2 +search_space=ppo_gym_classic algorithm=ppo env_kwargs={'continuous': true} 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=MountainCarContinuous-v0 +search_space=ppo_gym_classic algorithm=ppo 'num_steps=25e4' +cluster=local --config-name smac
python mighty/run_mighty.py -m env=Pendulum-v1 +search_space=ppo_gym_classic algorithm=ppo 'num_steps=25e4' +cluster=local --config-name smac

#DQN on DACBench
python mighty/run_mighty.py -m +environment=sigmoid +search_space=dqn_dacbench_toy algorithm=dqn +cluster=local --config-name smac
python mighty/run_mighty.py -m env=FastDownwardBenchmark +search_space=dqn_dacbench_fd algorithm=dqn 'num_steps=25e4' +cluster=local --config-name smac