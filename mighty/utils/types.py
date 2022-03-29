from typing import Optional, Dict, Any, Union, Tuple, Type, NewType
from omegaconf import DictConfig

TypeKwargs = NewType("TypeKwargs", Union[Dict[str, Any], DictConfig])
