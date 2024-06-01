import mlflow

def log_named_tuple_as_params(named_tuple):
    """
    Log the fields of a named tuple as MLflow parameters.

    Parameters:
    named_tuple (NamedTuple): The named tuple to log.
    """
    for field, value in named_tuple._asdict().items():
        mlflow.log_param(field, value)