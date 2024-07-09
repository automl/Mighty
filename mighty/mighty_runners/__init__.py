from .mighty_runner import MightyRunner
from .mighty_online_runner import MightyOnlineRunner
from .mighty_maml_runner import MightyMAMLRunner, MightyTRPOMAMLRunner
from .mighty_nes_runner import MightyNESRunner

VALID_RUNNER_TYPES = ["standard", "default", "online", "nes", "xnes", "snes"]
RUNNER_CLASSES = {
    "standard": MightyOnlineRunner,
    "default": MightyOnlineRunner,
    "online": MightyOnlineRunner,
    "nes": MightyNESRunner,
    "xnes": MightyNESRunner,
    "snes": MightyNESRunner,
}
from .factory import get_runner_class  # noqa: E402

__all__ = [
    "MightyRunner",
    "MightyOnlineRunner",
    "MightyMAMLRunner",
    "MightyTRPOMAMLRunner",
    "MightyNESRunner",
    "get_runner_class",
]
