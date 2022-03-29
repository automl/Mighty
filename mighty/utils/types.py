from typing import Optional, Dict, Any, Union, Tuple, Type, NewType
from omegaconf import DictConfig

TKwargs = NewType("TKwargs", Union[Dict[str, Any], DictConfig])
