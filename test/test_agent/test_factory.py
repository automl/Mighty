import unittest
import numpy as np

from mighty.agent.factory import get_agent_class
from mighty.agent.dqn import DQNAgent
from mighty.agent.sac import SACAgent
from mighty.agent.ppo import PPOAgent


class FactoryTestCase(unittest.TestCase):
    def test_get_agent_classes(self):
        agent_types = ["DQN", "DDQN", "SAC", "PPO"]
        agent_classes = [DQNAgent, DQNAgent, SACAgent, PPOAgent]

        factory_agent_classes = [get_agent_class(a) for a in agent_types]
        self.assertTrue(np.all(np.equal(agent_classes, factory_agent_classes)))

    def test_wrong_inputs(self):
        wrong_inputs = ["sdfsef", 3456, None, ["DDQN", "DDQN", "DDQN"]]
        for wrong_input in wrong_inputs:
            self.assertRaises(ValueError, get_agent_class, wrong_input)


if __name__ == "__main__":
    unittest.main()
