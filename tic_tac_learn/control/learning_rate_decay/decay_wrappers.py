
from tic_tac_learn.control.config_class_v2_MC import Config_2_MC
from tic_tac_learn.control.learning_rate_decay.decay_rate_calulator import e_decay,linear_decay
from typing import Optional

def _e_decay_from_config(step: int, 
                         conf : Optional[Config_2_MC] = None) -> float :
    
    conf = conf or Config_2_MC()
    params  = (conf.learning_rate_dict or {}).get("params", {}) or {}
    decay_rate = params.get("decay_rate",params.get("rate",0.0))
    return e_decay(step,
                   params.learning_rate_inital,
                   decay_rate)


def _linear_decay_from_config( step : int , 
                              conf: Optional[Config_2_MC] = None ) -> float: 

    conf = conf or Config_2_MC() 
    params = (conf.learning_rate_dict or {}).get("params", {}) or {}
    learning_rate_inital = params.get("learning_rate_inital",1)
    scaling_rate_scaling  = params.get("learning_rate_scaling", 1)
    learning_rate_min  = params.get("learning_rate_min", 0)
    scaling_frozen_steps = params.get("scaling_frozen_steps", 0)
    decay_steps = params.steps - scaling_frozen_steps

    return linear_decay(step,
                        scaling_rate_scaling,
                        learning_rate_inital,
                        learning_rate_min,
                        decay_steps)