from __future__ import annotations

import pytest
from pathlib import Path
import gymnasium as gym
from utils import DummyEnv, clean
from mighty.mighty_agents.dqn import MightyAgent, MightyDQNAgent
from mighty.mighty_utils.logger import Logger


class TestMightyAgent:
    def test_init(self):
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_base_agent", "test_base_agent")
        with pytest.raises(NotImplementedError):
            MightyAgent(
                env,
                logger,
                meta_kwargs=None,
                wandb_kwargs=None,
                meta_methods=None,
                log_tensorboard=True,
            )
        clean(logger)

    def test_make_checkpoint_dir(self):
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_dbase_agent", "test_base_agent")
        agent = MightyDQNAgent(env, logger)
        agent.make_checkpoint_dir(1)
        assert Path(agent.checkpoint_dir).exists()
        clean(logger)

    def test_apply_config(self):
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_base_agent", "test_base_agent")
        agent = MightyDQNAgent(env, logger)
        config = {
            "learning_rate": -1,
        }
        agent.apply_config(config)
        assert agent.learning_rate == -1
        clean(logger)
