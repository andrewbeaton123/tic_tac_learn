

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


def linear_decay(step: int, 
                learning_rate_scaling : float,
                learning_rate_start : float,
                learning_rate_min : float,
                decay_steps : int
                 ) -> float: 
    
    if decay_steps <= 0: # Prevent division by zero or negative steps for decay
        logging.warning("Config Warning: Decay steps are zero or negative. Learning rate will not decay.")

        return  0.0
    else:
        return  step* round(learning_rate_scaling *
                        (learning_rate_start - learning_rate_min
                        ) / decay_steps, 4)