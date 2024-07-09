from .mighty_runner import MightyRunner
from .mighty_online_runner import MightyOnlineRunner
from .mighty_maml_runner import MightyMAMLRunner, MightyTRPOMAMLRunner
from .mighty_es_runner import MightyESRunner

VALID_RUNNER_TYPES = ["standard", "default", "online", "es"]
RUNNER_CLASSES = {
    "standard": MightyOnlineRunner,
    "default": MightyOnlineRunner,
    "online": MightyOnlineRunner,
    "es": MightyESRunner,
}
from .factory import get_runner_class  # noqa: E402

__all__ = [
    "MightyRunner",
    "MightyOnlineRunner",
    "MightyMAMLRunner",
    "MightyTRPOMAMLRunner",
    "MightyESRunner",
    "get_runner_class",
]
