"""Type helpers for the mighty package."""
from __future__ import annotations

from typing import Any, NewType

from omegaconf import DictConfig

TypeKwargs = NewType("TypeKwargs", dict[str, Any] | DictConfig)
