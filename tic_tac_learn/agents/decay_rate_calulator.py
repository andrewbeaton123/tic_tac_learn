

import math as maths 


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