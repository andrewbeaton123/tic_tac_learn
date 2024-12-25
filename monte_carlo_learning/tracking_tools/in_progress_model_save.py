import mlflow
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent


def log_in_progress_mc_model(agent: MonteCarloAgent, episodes: int ) -> None:
    logging.debug("Mc model in progress saving starting")
    sci_format_episodes= f"{episodes:e}"
    mlflow.save_model(agent, f"mc_agent/in-progress-{episodes}")


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
    sci_format_episodes = f"{episodes:e}"
    try:
        mlflow.save_model(agent, f"mc_agent/in-progress-{sci_format_episodes}")
    except mlflow.exceptions.MlflowException as e:
        logging.error(f"Error saving model: {e}")
        raise # Re-raise to alert the calling function