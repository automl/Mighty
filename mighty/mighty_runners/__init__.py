from .mighty_runner import MightyRunner
from .mighty_online_runner import MightyOnlineRunner
from .mighty_maml_runner import MightyMAMLRunner, MightyTRPOMAMLRunner

VALID_RUNNER_TYPES = ["standard", "default", "online"]
RUNNER_CLASSES = {
    "standard": MightyOnlineRunner,
    "default": MightyOnlineRunner,
    "online": MightyOnlineRunner,
}
from .factory import get_runner_class  # noqa: E402

__all__ = [
    "MightyRunner",
    "MightyOnlineRunner",
    "MightyMAMLRunner",
    "MightyTRPOMAMLRunner",
    "get_runner_class",
]
