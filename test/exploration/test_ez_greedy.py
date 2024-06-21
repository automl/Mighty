import numpy as np
import gymnasium as gym
from utils import DummyEnv, clean
from mighty.utils.logger import Logger
from mighty.mighty_agents.dqn import MightyDQNAgent
from mighty.mighty_exploration.ez_greedy import EZGreedy


class TestEZGreedy:
    def test_init(self) -> None:
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_dqn_agent", "test_dqn_agent")
        dqn = MightyDQNAgent(
            env,
            logger,
            use_target=False,
            policy_class="mighty.mighty_exploration.EZGreedy",
        )
        assert isinstance(
            dqn.policy, EZGreedy
        ), "Policy should be an instance of EZGreedy when creating with string."
        assert dqn.policy.epsilon == 0.1, "Default epsilon should be 0.1."
        assert dqn.policy.zipf_param == 2, "Default zipf_param should be 2."
        assert dqn.policy.skipped is None, "Skip should be initialized at None."
        assert (
            dqn.policy.frozen_actions is None
        ), "Frozen actions should be initialized at None."

        dqn = MightyDQNAgent(
            env,
            logger,
            use_target=False,
            policy_class=EZGreedy,
            policy_kwargs={"epsilon": [0.5, 0.3], "zipf_param": 3},
        )
        assert isinstance(
            dqn.policy, EZGreedy
        ), "Policy should be an instance of EZGreedy when creating with class."
        assert np.all(dqn.policy.epsilon == [0.5, 0.3]), "Epsilon should be [0.5, 0.3]."
        assert dqn.policy.zipf_param == 3, "zipf_param should be 3."
        clean(logger)

    def test_skip_single(self) -> None:
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(1)])
        logger = Logger("test_ezgreedy", "test_ezgreedy")
        dqn = MightyDQNAgent(
            env,
            logger,
            use_target=False,
            policy_class="mighty.mighty_exploration.EZGreedy",
            policy_kwargs={"epsilon": 0.0, "zipf_param": 3},
        )

        state, _ = env.reset()
        action = dqn.policy([state])
        assert np.all(
            action < env.single_action_space.n
        ), "Action should be within the action space."
        assert len(action) == len(state), "Action should be predicted per state."

        dqn.policy.skipped = np.array([1])
        next_action = dqn.policy([state])
        assert np.all(
            action == next_action
        ), "Action should be the same as the previous action when skip is active."
        assert dqn.policy.skipped[0] == 0, "Skip should be decayed by one."
        clean(logger)

    def test_skip_batch(self) -> None:
        env = gym.vector.SyncVectorEnv([DummyEnv for _ in range(2)])
        logger = Logger("test_ezgreedy", "test_ezgreedy")
        dqn = MightyDQNAgent(
            env,
            logger,
            use_target=False,
            policy_class=EZGreedy,
            policy_kwargs={"epsilon": [0.5, 1.0], "zipf_param": 3},
        )

        state, _ = env.reset()
        action = dqn.policy(state)
        assert all(
            [a < env.single_action_space.n for a in action]
        ), "Actions should be within the action space."
        assert len(action) == len(state), "Action should be predicted per state."

        dqn.policy.skipped = np.array([3, 0])
        next_action = dqn.policy(state + 2)
        assert (
            action[0] == next_action[0]
        ), f"First action should be the same as the previous action when skip is active: {action[0]} != {next_action[0]}"
        assert dqn.policy.skipped[0] == 2, "Skip should be decayed by one."
        assert dqn.policy.skipped[1] >= 0, "Skip should not be decayed below one."
        clean(logger)
