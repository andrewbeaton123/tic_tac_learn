from enum import Enum

from .decay_wrappers import (_constant_decay_from_config,
                               _e_decay_from_config,
                               _linear_decay_from_config)

class DecayType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"



def apply_decay(decay_type: str, step: int, conf=None):
    DECAY_FUNCTIONS = {
        "constant": _constant_decay_from_config,
        "linear": _linear_decay_from_config,
        "exponential": _e_decay_from_config,
    }

    # Normalize decay_type (e.g., "LINEAR" -> "linear")
    normalized_type = decay_type.lower()
    if normalized_type not in DECAY_FUNCTIONS:
        raise ValueError(f"Unknown decay type: {decay_type}")

    return DECAY_FUNCTIONS[normalized_type](step, conf)