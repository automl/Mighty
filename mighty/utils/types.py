from typing import Dict, Any, Union, NewType
from omegaconf import DictConfig

TypeKwargs = NewType("TypeKwargs", Union[Dict[str, Any], DictConfig])
