import importlib

MIGHTYENV = None

dacbench = importlib.util.find_spec("dacbench")
dacbench_found = dacbench is not None
if dacbench_found:
    import dacbench
    MIGHTYENV = dacbench.AbstractEnv
    DACENV = dacbench.AbstractEnv
else:
    #FIXME: so this is strange. It can't be None, because I need to typecheck this. Int works, but it's strange. Can we improve it?
    DACENV = int

carl = importlib.util.find_spec("carl")
carl_found = carl is not None
if carl_found:
    import carl
    if MIGHTYENV is None:
        MIGHTYENV = carl.envs.CARLEnv
    CARLENV = int#carl.envs.CARLEnv
else:
    CARLENV = int

if not carl_found and not dacbench_found:
    import gym
    MIGHTYENV = gym.Env
    print("Warning: you do not have DACBench or CARL installed, will default to gym.Env. These environment may not behave the same!")
