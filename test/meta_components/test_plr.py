from __future__ import annotations

import pytest
import numpy as np
import gymnasium as gym
from mighty.mighty_meta import PrioritizedLevelReplay as PLR
from utils import DummyEnv, clean
from mighty.mighty_utils.logger import Logger
from mighty.mighty_agents.dqn import MightyDQNAgent
from mighty.mighty_utils.wrappers import ContextualVecEnv


class EnvSim:
    def __init__(self, inst_id):
        self.inst_ids = [inst_id]
        self.instance_id_list = np.arange(10)
        self.action_space = gym.spaces.Discrete(2)


DUMMY_METRICS = [
    {
        "env": EnvSim(1),
        "rollout_values": np.array([[0.3, 0.7, 0.45, 0.6, 0.7]]),
        "episode_reward": np.array([[2]]),
    },
    {
        "env": EnvSim(3),
        "rollout_values": np.array([[0.1, 0.01, 0.15, 0.03, 0.07]]),
        "episode_reward": np.array([[0.5]]),
    },
    {
        "env": EnvSim(4),
        "rollout_values": np.array([[0.8, 0.7, 0.57, 0.9, 0.9]]),
        "episode_reward": np.array([[20]]),
        "rollout_logits": np.array(
            [
                [
                    [0.3, 0.7],
                    [0.7, 0.3],
                    [0.45, 0.55],
                    [0.6, 0.4],
                    [0.7, 0.3],
                ]
            ]
        ),
    },
]


class TestPLR:
    def test_init(self) -> None:
        plr = PLR(
            alpha=0.1,
            rho=0.1,
            staleness_coeff=0.3,
            sample_strategy="gae",
            score_transform="max",
            temperature=0.8,
            staleness_transform="max",
            staleness_temperature=0.8,
            eps=1e-5,
        )
        assert plr.rng is not None, "Random number generator should be initialized."
        assert plr.alpha == 0.1, "Alpha should be set to 0.1."
        assert plr.rho == 0.1, "Rho should be set to 0.1."
        assert plr.staleness_coef == 0.3, "Staleness coefficient should be set to 0.3."
        assert plr.sample_strategy == "gae", "Sample strategy should be set to GAE."
        assert plr.score_transform == "max", "Score transform should be set to max."
        assert plr.temperature == 0.8, "Temperature should be set to 0.8."
        assert plr.eps == 1e-5, "Epsilon should be set to 1e-5."
        assert (
            plr.staleness_transform == "max"
        ), "Staleness transform should be set to max."
        assert (
            plr.staleness_temperature == 0.8
        ), "Staleness temperature should be set to 0.8."

        assert (
            plr.instance_scores == {}
        ), "Instance scores should be an empty dictionary."
        assert plr.staleness == {}, "Staleness should be an empty dictionary."
        assert plr.all_instances is None, "All instances should be None."
        assert plr.index == 0, "Index should be 0."
        assert plr.num_actions is None, "Number of actions should be None."

    def test_get_instance(self) -> None:
        plr = PLR()
        for m in DUMMY_METRICS:
            plr.add_rollout(m)
        metrics = {"env": EnvSim(1)}
        plr.get_instance(metrics=metrics)
        assert metrics["env"].inst_ids is not None, "Instance should not be None."
        assert (
            metrics["env"].inst_ids[0] in plr.instance_scores.keys()
        ), "Instance should be in instance scores."
        assert plr.all_instances is not None, "All instances should be initialized."
        assert all(
            [i in plr.all_instances for i in plr.instance_scores.keys()]
        ), "All instances should be in instance scores."

        plr.sample_strategy = "sequential"
        index = plr.index
        metrics = {"env": EnvSim(10)}
        original_instance = 10
        plr.get_instance(metrics=metrics)
        assert (
            original_instance != metrics["env"].inst_ids[0]
        ), "Instance should be changed."
        assert plr.index == index + 1, "Index should be incremented by 1."

    def test_sample_weights(self) -> None:
        plr = PLR()
        for m in DUMMY_METRICS:
            plr.add_rollout(m)
        weights = plr.sample_weights()
        assert len(weights) == len(
            plr.instance_scores.keys()
        ), "Length of weights should be equal to the number of instances."
        assert np.isclose(sum(weights), 1), "Sum of weights should be 1."

    def test_score_transforms(self) -> None:
        plr = PLR()
        for m in DUMMY_METRICS:
            plr.add_rollout(m)
        for score_transform in [
            "constant",
            "max",
            "eps_greedy",
            "rank",
            "power",
            "softmax",
        ]:
            plr.score_transform = score_transform
            weights = plr.sample_weights()
            assert len(weights) == len(
                plr.instance_scores.keys()
            ), "Length of weights should be equal to the number of instances."
            assert np.isclose(sum(weights), 1), "Sum of weights should be 1."

    @pytest.mark.parametrize("metrics", DUMMY_METRICS)
    def test_score_function(self, metrics) -> None:
        for score_func in [
            "one_step_td_error",
            "value_l1",
            "min_margin",
            "gae",
            "least_confidence",
            "policy_entropy",
            "random",
        ]:
            plr = PLR(sample_strategy=score_func)
            if (
                score_func in ["least_confidence", "policy_entropy", "min_margin"]
                and "rollout_logits" not in metrics
            ):
                with pytest.raises(ValueError):
                    plr.add_rollout(metrics)
            else:
                plr.add_rollout(metrics)
                assert (
                    plr.instance_scores[metrics["env"].inst_ids[0]] is not None
                ), "Instance score should not be None."

            if score_func == "random":
                assert (
                    plr.instance_scores[metrics["env"].inst_ids[0]] == 1
                ), "Random score should be 1."

    @pytest.mark.parametrize("metrics", DUMMY_METRICS)
    def test_add_rollout(self, metrics) -> None:
        plr = PLR()
        plr.add_rollout(metrics)
        assert (
            metrics["env"].inst_ids[0] in plr.instance_scores
        ), "Instance should be added to instance scores."

    def test_in_loop(self) -> None:
        env = ContextualVecEnv([DummyEnv for _ in range(2)])
        logger = Logger("test_plr", "test_plr")
        dqn = MightyDQNAgent(
            env,
            logger,
            use_target=False,
            meta_methods=["mighty.mighty_meta.PrioritizedLevelReplay"],
        )
        assert (
            dqn.meta_modules["PrioritizedLevelReplay"] is not None
        ), "PLR should be initialized."
        dqn.run(100, 0)
        assert (
            dqn.meta_modules["PrioritizedLevelReplay"].all_instances is not None
        ), "All instances should be initialized."
        assert (
            env.inst_ids[0] in dqn.meta_modules["PrioritizedLevelReplay"].all_instances
        ), "Instance should be in all instances."
        clean(logger)
