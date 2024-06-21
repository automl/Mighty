from __future__ import annotations

from mighty.mighty_meta import SPaCE
from utils import DummyEnv, clean, DummyModel
from mighty.utils.logger import Logger
from mighty.mighty_agents.dqn import MightyDQNAgent
from mighty.utils.wrappers import ContextualVecEnv

class TestSPaCE:
    def test_init(self) -> None:
        space = SPaCE(criterion="improvement", threshold=0.5, k=2)
        assert space.criterion == "improvement"
        assert space.threshold == 0.5
        assert space.increase_by_k_instances == 2
        assert space.current_instance_set_size == 2
        assert space.last_evals is None

    def test_get_instances(self) -> None:
        space = SPaCE()
        metrics = {"env": DummyEnv(), "vf": DummyModel(), "rollout_values": [[0.0, 0.6, 0.7]]}
        space.get_instances(metrics)
        assert len(space.all_instances) == 1, f"Expected 1, got {len(space.all_instances)}"
        assert len(space.instance_set) == 1, f"Expected 1, got {len(space.instance_set)}"
        assert space.last_evals is not None, f"Evals should not be None."

    def test_get_evals(self) -> None:
        vf = DummyModel()
        env = DummyEnv()
        space = SPaCE()
        space.all_instances = env.instance_id_list
        values = space.get_evals(env, vf)
        assert len(values) == 1, f"Expected 1 value, got {len(values)}"

    def test_in_loop(self) -> None:
        env = ContextualVecEnv([DummyEnv for _ in range(2)])
        logger = Logger("test_plr", "test_plr")
        dqn = MightyDQNAgent(
            env,
            logger,
            use_target=False,
            meta_methods=["mighty.mighty_meta.SPaCE"],
        )
        assert (
            dqn.meta_modules["SPaCE"] is not None
        ), "SPaCE should be initialized."
        dqn.run(100, 0)
        assert (
            dqn.meta_modules["SPaCE"].all_instances is not None
        ), "All instances should be initialized."
        assert (
            env.inst_ids[0] in dqn.meta_modules["SPaCE"].all_instances
        ), "Instance should be in all instances."
        clean(logger)
        
