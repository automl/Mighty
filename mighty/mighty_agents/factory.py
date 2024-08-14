"""Factory for creating agents based on config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mighty.mighty_agents import VALID_AGENT_TYPES, AGENT_CLASSES


if TYPE_CHECKING:
    from mighty.mighty_agents.base_agent import MightyAgent


def get_agent_class(agent_type: str) -> MightyAgent:
    """Transforms config keyword for agents to class."""
    agent_class = None
    if agent_type in VALID_AGENT_TYPES:
        agent_class = AGENT_CLASSES[agent_type]
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class  # type: ignore
