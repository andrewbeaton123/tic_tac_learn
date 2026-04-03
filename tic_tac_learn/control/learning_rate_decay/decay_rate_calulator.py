

import math as maths 
import logging

def e_decay(step : int,
            starting_learning_rate: float,
            decay_rate: float) ->float: 
    """
    Docstring for e_decay
    
    :param step: Description
    :type step: int
    :param starting_learning_rate: Description
    :type starting_learning_rate: float
    :param decay_rate: Description
    :type decay_rate: float
    """
    
    return float(starting_learning_rate * maths.e**(step*decay_rate))

def constant_learning_rate( 
                            starting_learning_rate : float ) -> float : 
    
    return starting_learning_rate


def linear_decay(decay_step: int, 
                learning_rate_scaling: float,
                learning_rate_start: float,
                learning_rate_min: float,
                total_decay_steps: int) -> float: 
    
    if total_decay_steps <= 0:
        logging.warning("Config Warning: Decay steps are zero or negative. Learning rate will not decay.")
        return learning_rate_start
    else:
        decay_per_step = (learning_rate_start - learning_rate_min) / total_decay_steps
        decay_per_step *= learning_rate_scaling
        new_lr = learning_rate_start - (decay_step * decay_per_step)
        return round(max(new_lr, learning_rate_min), 4)