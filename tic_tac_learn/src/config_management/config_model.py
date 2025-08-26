
import logging
import tomli

from pydantic import BaseModel, Field, computed_field
from typing import Optional

from pathlib import Path

def get_version_from_pyproject() -> str:
    try:
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomli.load(f)
        return pyproject_data["project"]["version"]
    except Exception as e:
        logging.warning(f"Could not read version from pyproject.toml: {e}")
        return "0.0.0"
    

class TrainingConfig(BaseModel):
    cores: int = Field(default=3, ge=1)
    learning_rate_start: float = Field(default=0.8, gt=0, le=1)
    learning_rate_min: float = Field(default=0.001, gt=0, le=1)
    learning_rate_scaling: float = Field(default=1, gt=0)
    test_games_per_step: int = Field(default=30000, gt=0)
    learning_rate_flat_games: Optional[float] = None
    _total_games: int = 0  # Private field for calculations
    _steps: int = 0        # Private field for calculations

    def update_training_params(self, total_games: int, steps: int) -> None:
        """Update the training parameters needed for calculations"""
        self._total_games = total_games
        self._steps = steps

    @computed_field
    @property
    def frozen_learning_rate_steps(self) -> int:
        return max(1, int(self.learning_rate_flat_games / (self._total_games / self._steps)))

    @computed_field
    @property
    def games_per_step(self) -> int:
        return int(self._total_games / self._steps)

    @computed_field
    @property
    def learning_rate_decay_rate(self) -> float:
        return round(
            self.learning_rate_scaling *
            (self.learning_rate_start - self.learning_rate_min) /
            (self._steps - self.frozen_learning_rate_steps),
            4
        )



class ExperimentConfig(BaseModel):
    name: str = f"Tic Tac Learn {get_version_from_pyproject()}"
    level: str = "UNCONFIGURED"
    total_games: int = Field(default=1_000, gt=0)
    steps: int = Field(default=4, gt=0)
    mlflow_name: str = Field(default= "No mlflow_name set")
    agent_reload: str|None = Field(default = None)
    training: TrainingConfig = TrainingConfig()
    

    def pre_run_calculations(self) -> None:
        """
        Performs and logs pre-run calculations.
        Maintains compatibility with existing code.
        """
        if self.training.learning_rate_flat_games is None:
            self.training.learning_rate_flat_games = self.total_games * 0.2
            
        # Update training config with required parameters
        self.training.update_training_params(self.total_games, self.steps)

        logging.info("Monte Carlo Pre run calculations finished.")
        logging.debug(f"frozen_learning_rate_steps = {self.training.frozen_learning_rate_steps}")
        logging.debug(f"games_per_step = {self.training.games_per_step}")
        logging.debug(f"learning_rate_decay_rate = {self.training.learning_rate_decay_rate}")

    
    
    @property
    def run_name(self) -> str:
        return f"One Billion Games {self.steps} Steps - {self.level} - {self.total_games}"

    @property
    def custom_model_name(self) -> str:
        return f"{self.run_name}_2mc"
    