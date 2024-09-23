import mlflow


from src.control import Config_2_MC



def learning_rate_scaling(rate: float, current_games_played : int):
    conf = Config_2_MC
    if current_games_played >= conf.learning_rate_flat_games : 
                
                rate -= conf.learning_rate_decay_rate# 0.1 #learning_rate_change*25
                if rate < conf.learning_rate_min: 
                    rate = conf.learning_rate_min
                mlflow.log_metric("Learning Rate", rate)
    return rate