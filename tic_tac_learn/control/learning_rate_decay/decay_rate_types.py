from enum import Enum



class DecayType(Enum):
    CONSTANT = '_constant_decay_from_config'
    LINEAR = '_linear_decay_from_config'
    EXPONENTIAL = '_e_decay_from_config'