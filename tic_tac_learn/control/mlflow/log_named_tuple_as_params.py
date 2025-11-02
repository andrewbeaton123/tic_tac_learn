import mlflow
from tic_tac_learn.control import Config_2_MC

def log_named_tuple_as_params(named_tuple: Config_2_MC):
    for field, value in named_tuple._asdict().items():
        mlflow.log_param(field, value)