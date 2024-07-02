"""Factory for creating runners based on config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mighty.mighty_runners.mighty_online_runner import MightyOnlineRunner
from mighty.mighty_runners.mighty_maml_runner import (
    MightyMAMLRunner,
    MightyTRPOMAMLRunner,
)

if TYPE_CHECKING:
    from mighty.mighty_agents.mighty_agent import MightyAgent


def get_agent_class(agent_type: str) -> type(MightyAgent):
    """Transforms config keyword for agents to class."""
    agent_class = None
    if agent_type in ("default", "standard", "online"):
        agent_class = MightyOnlineRunner
    elif agent_type in ("maml", "MAML"):
        agent_class = MightyMAMLRunner
    elif agent_type in ("trpo_maml", "TRPO_MAML"):
        agent_class = MightyTRPOMAMLRunner
    else:
        raise ValueError(f"Unknown agent_type {agent_type}.")

    return agent_class
