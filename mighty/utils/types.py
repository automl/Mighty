from typing import Any, Dict, NewType, Union

from omegaconf import DictConfig

TypeKwargs = NewType("TypeKwargs", Union[Dict[str, Any], DictConfig])
