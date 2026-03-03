from enum import Enum



class DecayType(Enum):
    CONSTANT = 'constant'
    LINEAR = '_linear_decay_from_config'
    EXPONENTIAL = '_e_decay_from_config'