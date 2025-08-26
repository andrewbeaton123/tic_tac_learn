import mlflow

import logging
from tic_tac_learn.src.config_management import ConfigManager



def learning_rate_scaling(rate: float, current_games_played : int, step : int):
    conf = ConfigManager().config
    if current_games_played >= conf.training.learning_rate_flat_games : 
                logging.debug("Starting learning rate change")
                rate -= conf.training.learning_rate_decay_rate# 0.1 #learning_rate_change*25
                if rate < conf.training.learning_rate_min: 
                    rate = conf.training.learning_rate_min
                mlflow.log_metric("Learning Rate", rate,step=step)
    return rate