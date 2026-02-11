
from tic_tac_learn.control.config_class_v2_MC import Config_2_MC
from tic_tac_learn.control.learning_rate_decay.decay_rate_calulator import e_decay
from typing import Optional

def _e_decay_from_config(step: int, 
                         conf : Optional[Config_2_MC] = None
                         ) -> float :
    
    conf = conf or Config_2_MC()
    params  = (conf.learning_rate_method or {}).get("params", {}) or {}
    decay_rate = params.get("decay_rate",params.get("rate",0.0))
    return e_decay(step,conf.learning_rate_start,decay_rate)
