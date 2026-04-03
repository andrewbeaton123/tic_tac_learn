

from tic_tac_learn.control.learning_rate_decay.decay_rate_calulator import e_decay,linear_decay,constant_learning_rate
from typing import Optional

#TODO  Shoudl improve from the lazy importing

def _constant_decay_from_config(step: int, conf=None) -> float:
    params = (conf.learning_rate_dict or {}).get("params", {}) or {}
    return constant_learning_rate(params.get("learning_rate_inital", 1.0))

def _e_decay_from_config(step: int, conf=None) -> float:
    params = (conf.learning_rate_dict or {}).get("params", {}) or {}
    decay_rate = params.get("decay_rate", params.get("rate", 0.0))
    learning_rate_inital = params.get("learning_rate_inital", 1.0)
    
    # Match the config YAML key for frozen steps
    frozen_steps = params.get("learning_rate_frozen_steps", params.get("scaling_frozen_steps", 0))
    
    # Calculate relative step
    decay_step = step - frozen_steps + 1
    if decay_step < 0:
        decay_step = 0
        
    return e_decay(decay_step, learning_rate_inital, decay_rate)

def _linear_decay_from_config(step: int, conf=None) -> float:
    params = (conf.learning_rate_dict or {}).get("params", {}) or {}
    learning_rate_inital = params.get("learning_rate_inital", 1.0)
    scaling_rate_scaling = params.get("learning_rate_scaling", 1.0)
    learning_rate_min = params.get("learning_rate_min", 0.0)
    
    # Match the config YAML key for frozen steps
    frozen_steps = params.get("learning_rate_frozen_steps", params.get("scaling_frozen_steps", 0))
    
    total_steps = getattr(conf, 'steps', 10)
    total_decay_steps = total_steps - frozen_steps

    # Calculate how far along we are in the decay phase (1-indexed for the formula to drop rate each step)
    decay_step = step - frozen_steps + 1
    if decay_step < 0:
        decay_step = 0

    return linear_decay(decay_step,
                        scaling_rate_scaling,
                        learning_rate_inital,
                        learning_rate_min,
                        total_decay_steps)