import unittest
import numpy as np

from mighty.agent.factory_deprecated import get_agent_class
from mighty.agent.dqn_deprecated import MightyDQNAgent
from mighty.agent.sac_deprecated import MightySACAgent
from mighty.agent.ppo_deprecated import MightyPPOAgent


class FactoryTestCase(unittest.TestCase):
    def test_get_agent_classes(self):
        agent_types = ["DQN", "DDQN", "SAC", "PPO"]
        agent_classes = [MightyDQNAgent, MightyDQNAgent, MightySACAgent, MightyPPOAgent]

        factory_agent_classes = [get_agent_class(a) for a in agent_types]
        self.assertTrue(np.all(np.equal(agent_classes, factory_agent_classes)))

    def test_wrong_inputs(self):
        wrong_inputs = ["sdfsef", 3456, None, ["DDQN", "DDQN", "DDQN"]]
        for wrong_input in wrong_inputs:
            self.assertRaises(ValueError, get_agent_class, wrong_input)


if __name__ == "__main__":
    unittest.main()
