try:
    import dacbench
    DACENV = dacbench.AbstractEnv
except:
    import gym
    DACENV = gym.Env
    print("Warning: you do not have DACBench installed, will default to gym.Env. These environment may not behave the same!")
