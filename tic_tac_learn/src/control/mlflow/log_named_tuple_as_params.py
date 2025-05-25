import mlflow
from tic_tac_learn.src.control import Config_2_MC

def log_named_tuple_as_params(config: Config_2_MC):
    """
    Log the fields of a named tuple as MLflow parameters.

    Parameters:
    named_tuple (NamedTuple): The named tuple to log.
    """
    for attr_name, attr_value in config.__dict__.items():
        mlflow.log_param(attr_name, attr_value)