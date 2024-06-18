"""Env typing utilities for Mighty."""
from __future__ import annotations

import importlib

MIGHTYENV = None

dacbench = importlib.util.find_spec("dacbench")
dacbench_found = dacbench is not None
if dacbench_found:
    import dacbench

    MIGHTYENV = dacbench.AbstractEnv
    DACENV = dacbench.AbstractEnv
else:
    DACENV = int

carl = importlib.util.find_spec("carl")
carl_found = carl is not None
if carl_found:
    from carl.envs.carl_env import CARLEnv

    if MIGHTYENV is None:
        MIGHTYENV = CARLEnv
    CARLENV = CARLEnv
else:
    CARLENV = int

if not carl_found and not dacbench_found:
    import gymnasium as gym

    MIGHTYENV = gym.Env
