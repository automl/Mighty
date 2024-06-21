from __future__ import annotations

from mighty.mighty_agents.dqn import MightyDQNAgent
import gymnasium as gym
from utils import DummyEnv, clean
from mighty.utils.logger import Logger


class TestCosineLR:
    def test_decay(self) -> None:
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_dqn_agent", "test_dqn_agent")
        dqn = MightyDQNAgent(
            env,
            logger,
            meta_methods=["mighty.mighty_meta.CosineLRSchedule"],
            meta_kwargs=[{"initial_lr": 0.2, "num_decay_steps": 100, "restart_every": 0}],
        )
        lr = 1.5
        dqn.learning_rate = lr
        for i in range(4):
            metrics = dqn.run(n_steps=10*(i+1), n_episodes_eval=0)
            assert metrics["hp/lr"] == dqn.learning_rate, f"Learning rate should be set to schedule value {metrics['hp/lr']} instead of {dqn.learning_rate}."
            assert dqn.learning_rate < lr, f"Learning rate should decrease: {dqn.learning_rate} is not less than {lr}."
            lr = dqn.learning_rate.copy()
        clean(logger)

    def test_restart(self) -> None:
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_dqn_agent", "test_dqn_agent")
        dqn = MightyDQNAgent(
            env,
            logger,
            meta_methods=["mighty.mighty_meta.CosineLRSchedule"],
            meta_kwargs=[{"initial_lr": 0.2, "num_decay_steps": 100, "restart_every": 5}],
        )
        dqn.run(6, 0)
        assert dqn.meta_modules["CosineLRSchedule"].n_restarts == 1, "Restart counter should increase."
        assert dqn.learning_rate >= dqn.meta_modules["CosineLRSchedule"].eta_max, f"Restart should increase learning rate: {dqn.learning_rate} is not {dqn.meta_modules['CosineLRSchedule'].eta_max}."
        clean(logger)
