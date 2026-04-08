
from .decay_rate_types import apply_decay
import logging
from typing import Optional, Any

def decay_from_name(name: str, step: int, conf: Optional[Any] = None) -> float:
    if name is None:
        logging.warning(f"Learning rate decay name is {name}")
        raise ValueError(f"Unknown decay function: {name}")
    else: 
        logging.debug(f"Decay rate name : {name}")
        return apply_decay(name, step, conf)