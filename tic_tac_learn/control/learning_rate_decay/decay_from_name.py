
from .decay_rate_types import DecayType
from typing import Optional
from tic_tac_learn.control.config_base_class import ConfigBaseClass

def decay_from_name(name: str, step: int, conf: Optional[ConfigBaseClass] = None) -> float:
    if name is None:
        raise ValueError(f"Unknown decay function: {name}")
    
    func  = DecayType(name).value 
    return func(step, conf)