from __future__ import annotations

import pytest
from mighty.mighty_runners.factory import get_runner_class
from mighty.mighty_runners.mighty_online_runner import MightyOnlineRunner

VALID_RUNNER_TYPES = ["standard", "default", "online"]
RUNNER_CLASSES = {
    "standard": MightyOnlineRunner,
    "default": MightyOnlineRunner,
    "online": MightyOnlineRunner,
}


class TestFactory:
    def test_create_agent(self):
        for runner_type in VALID_RUNNER_TYPES:
            runner_class = get_runner_class(runner_type)
            assert runner_class == RUNNER_CLASSES[runner_type], f"Runner class should be {RUNNER_CLASSES[runner_type]}"

    def test_create_agent_with_invalid_type(self):
        with pytest.raises(ValueError):
            get_runner_class("INVALID")
