
from .decay_rate_types import DecayType
import logging
from typing import Optional
from tic_tac_learn.control.config_base_class import ConfigBaseClass

def decay_from_name(name: str, step: int, conf: Optional[ConfigBaseClass] = None) -> float:
    if name is None:
        logging.warning(f"Learning rate decay name is {name}")
        raise ValueError(f"Unknown decay function: {name}")
    else : 
        logging.info(f"Decay rate name : {name}")
        func  = DecayType[name].value 
        #logging.info(f"Decay rate function name : {func}")
        return func(step, conf)