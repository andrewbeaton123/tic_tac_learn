from enum import Enum

from .decay_wrappers import (_constant_decay_from_config,
                               _e_decay_from_config,
                               _linear_decay_from_config)

class DecayType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"



def apply_decay(decay_type: str, step: int):
    DECAY_FUNCTIONS = {
    "constant": _constant_decay_from_config,
    "linear": _linear_decay_from_config,
    "exponential": _e_decay_from_config,
    }

    DECAY_FUNCTIONS[decay_type](step)