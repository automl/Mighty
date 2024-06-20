from __future__ import annotations

import pytest
from mighty.mighty_agents.factory import get_agent_class
from mighty.mighty_agents.dqn import MightyDQNAgent

VALID_AGENT_TYPES = ["DQN", "DDQN"]
AGENT_CLASSES = {"DQN": MightyDQNAgent, "DDQN": MightyDQNAgent}


class TestFactory:
    def test_create_agent(self):
        for agent_type in VALID_AGENT_TYPES:
            agent_class = get_agent_class(agent_type)
            assert agent_class == AGENT_CLASSES[agent_type]

    def test_create_agent_with_invalid_type(self):
        with pytest.raises(ValueError):
            get_agent_class("INVALID")
