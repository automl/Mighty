"""Factory for creating agents based on config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mighty.mighty_agents.dqn import MightyDQNAgent

if TYPE_CHECKING:
    from mighty.mighty_agents.base_agent import MightyAgent


def get_agent_class(agent_type: str) -> type(MightyAgent):
    """Transforms config keyword for agents to class."""
    agent_class = None
    if agent_type in ("DDQN", "DQN"):
        agent_class = MightyDQNAgent
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class
