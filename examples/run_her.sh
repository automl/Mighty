python ../mighty/run_mighty.py algorithm=ppo env=Ant-v4 env_wrappers=[mighty.mighty_replay.HERGoalWrapper] +algorithm_kwargs.learning_starts=500 "+wrapper_kwargs={goal: [1000, 0], check_achieved: examples.run_her_utils.check_goal_reached}" +algorithm_kwargs.replay_buffer_kwargs.alternate_goal_function=examples.run_her_utils.get_goal +algorithm_kwargs.replay_buffer_kwargs.gamma=0.9 algorithm_kwargs.replay_buffer_class=mighty.mighty_replay.HER