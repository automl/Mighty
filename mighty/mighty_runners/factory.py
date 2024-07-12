"""Factory for creating runners based on config."""

from __future__ import annotations

from typing import TYPE_CHECKING
from mighty.mighty_runners import VALID_RUNNER_TYPES, RUNNER_CLASSES

if TYPE_CHECKING:
    from mighty.mighty_runners.mighty_runner import MightyRunner


def get_runner_class(agent_type: str) -> type(MightyRunner):
    """Transforms config keyword for agents to class."""
    agent_class = None
    if agent_type in VALID_RUNNER_TYPES:
        agent_class = RUNNER_CLASSES[agent_type]
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class
