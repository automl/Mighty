from omegaconf import DictConfig
import hydra

def retrieve_class(cls: str | DictConfig | type, default_cls: type) -> type:
    """Get coax or mighty class."""
    if cls is None:
        cls = default_cls
    elif isinstance(cls, DictConfig):
        cls = hydra.utils.get_class(cls._target_)
    elif isinstance(cls, str):
        cls = hydra.utils.get_class(cls)
    return cls
