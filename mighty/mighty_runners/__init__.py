from typing import Dict

from .mighty_runner import MightyRunner
from .mighty_online_runner import MightyOnlineRunner
from .mighty_maml_runner import MightyMAMLRunner, MightyTRPOMAMLRunner


VALID_RUNNER_TYPES = ["standard", "default", "online"]
RUNNER_CLASSES: Dict[str, type[MightyRunner]] = {
    "standard": MightyOnlineRunner,
    "default": MightyOnlineRunner,
    "online": MightyOnlineRunner,
}

import importlib.util as iutil  # noqa: E402

spec = iutil.find_spec("evosax")
found = spec is not None
if found:
    from .mighty_es_runner import MightyESRunner

    VALID_RUNNER_TYPES.append("es")
    RUNNER_CLASSES["es"] = MightyESRunner


from .factory import get_runner_class  # noqa: E402

__all__ = [
    "MightyRunner",
    "MightyOnlineRunner",
    "MightyMAMLRunner",
    "MightyTRPOMAMLRunner",
    "MightyESRunner",
    "get_runner_class",
]
