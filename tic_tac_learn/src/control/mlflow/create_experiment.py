import mlflow

def create_mlflow_experiment(experiment_name):
    """
    Create an MLflow experiment if one with the given name doesn't already exist.

    Parameters:
    experiment_name (str): The name of the experiment.

    Returns:
    str: The ID of the experiment.
    """
    # Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        # Experiment does not exist, so create it
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    else:
        # Experiment already exists
        experiment_id = experiment.experiment_id
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment_id}")
    
    return experiment_id

# Example usage
if __name__ == "__main__":
    experiment_id = create_mlflow_experiment("my_experiment")
    print(f"Using experiment ID: {experiment_id}")
