from dataclasses import dataclass
import logging
from ..config_management.config_model import ExperimentConfig

@dataclass
class TrainingParameters:
    frozen_learning_rate_steps: int
    games_per_step: int
    learning_rate_decay_rate: float

class TrainingCalculator:
    @staticmethod
    def calculate_training_parameters(config: ExperimentConfig) -> TrainingParameters:
        # Calculate frozen learning rate steps
        frozen_steps = max(1, int(config.training.learning_rate_flat_games /
                          (config.total_games / config.steps)))
        
        # Calculate games per step
        games_per_step = int(config.total_games / config.steps)
        
        # Calculate learning rate decay
        decay_rate = round(
            config.training.learning_rate_scaling *
            (config.training.learning_rate_start - config.training.learning_rate_min) /
            (config.steps - frozen_steps),
            4
        )
        
        params = TrainingParameters(
            frozen_learning_rate_steps=frozen_steps,
            games_per_step=games_per_step,
            learning_rate_decay_rate=decay_rate
        )
        
        logging.info("Monte Carlo Pre run calculations finished.")
        logging.debug(f"frozen_learning_rate_steps = {params.frozen_learning_rate_steps}")
        logging.debug(f"games_per_step = {params.games_per_step}")
        logging.debug(f"learning_rate_decay_rate = {params.learning_rate_decay_rate}")
        
        return params