import mlflow
import logging
import pickle 
import os
import tempfile
from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
from mlflow.exceptions import MlflowException
from src.control.config_class_v2_MC import Config_2_MC

#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent


def log_in_progress_mc_model(agent: "MonteCarloAgent", episodes: int) -> None:
    """Logs and saves a Monte Carlo agent model during training.

    This function saves a snapshot of the Monte Carlo agent model to MLflow
    using a filename that includes the current episode number. This allows
    tracking of model progress during training. The episode number is formatted
    in scientific notation for better readability with very large episode counts.

    Args:
        agent: The Monte Carlo agent model to be saved.
        episodes: The current episode number.

    Raises:
      mlflow.exceptions.MlflowException: If there is an error during model saving.

    Example:
        >>> from your_module import MonteCarloAgent
        >>> agent = MonteCarloAgent(...) # Initialize your agent
        >>> log_in_progress_mc_model(agent, 1000)
        >>> log_in_progress_mc_model(agent, 1000000) # Example with large episode number

    Note:
        This function assumes that MLflow is properly configured and an active
        run is already started. The model is saved to the "mc_agent" directory
        within the MLflow artifacts.

    """

    logging.debug("Mc model in progress saving starting")
    config = Config_2_MC()
    custom_model_name = config.custom_model_name

    sci_format_episodes = f"{episodes:e}"
    artifact_path = f"mc_agent/in-progress-{sci_format_episodes}"
    #q_values_path = os.path.join(f"{artifact_path}_q_values", "saved_q_values.pkl")

    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_artifact_path = os.path.join(temp_dir, artifact_path)
        os.makedirs(temp_artifact_path, exist_ok=True)

        # Save the q_values to a file
        temp_q_values_path = os.path.join(temp_artifact_path, "saved_q_values.pkl")
        with open(temp_q_values_path, 'wb') as f:
            pickle.dump(agent.q_values, f)

        try:
            mlflow.pyfunc.save_model(
                path=f"{temp_artifact_path}_model",
                python_model=agent,
                artifacts={"q_values": temp_q_values_path}
            )

            # Log the model artifact
            mlflow.log_artifact(temp_artifact_path)

            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
            mlflow.register_model(model_uri=model_uri, name=custom_model_name)

        except MlflowException as e:
            logging.error(f"Error saving model: {e}")
            raise # Re-raise to alert the calling function