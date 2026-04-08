from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class DecayType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    NONE = "none"

@dataclass(frozen=True)
class EnvConfig:
    game_id: str = "tic_tac_toe"
    allowed_players: List[int] = field(default_factory=lambda: [1, 2])
    training_player: int = 1

@dataclass(frozen=True)
class AgentConfig:
    learning_rate_inital: float = 0.7
    learning_rate_min: float = 0.01
    decay_type: str = "LINEAR"
    decay_params: Dict[str, Any] = field(default_factory=dict)
    discount_factor: float = 0.9
    exploration_rate: float = 0.1
    
@dataclass(frozen=True)
class RunnerConfig:
    run_name: str = "Default Run"
    experiment_name: str = "Tic Tac Learn"
    total_games: int = 10000
    steps: int = 10
    cores: int = 1
    test_games_per_step: int = 1000
    mlflow_tracking_uri: Optional[str] = None
    log_mlflow: bool = True

@dataclass(frozen=True)
class MonteCarloConfig:
    env: EnvConfig
    agent: AgentConfig
    runner: RunnerConfig
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_games(self) -> int: return self.runner.total_games
    @property
    def learning_rate_type(self) -> str: return self.agent.decay_type

    @property
    def steps(self) -> int: return self.runner.steps
    @property
    def cores(self) -> int: return self.runner.cores
    @property
    def learning_rate_start(self) -> float: return self.agent.learning_rate_inital
    @property
    def learning_rate_min(self) -> float: return self.agent.learning_rate_min
    @property
    def discount_factor(self) -> float: return self.agent.discount_factor
    @property
    def exploration_rate(self) -> float: return self.agent.exploration_rate
    @property
    def training_player(self) -> int: return self.env.training_player
    @property
    def test_games_per_step(self) -> int: return self.runner.test_games_per_step
    @property
    def frozen_learning_rate_steps(self) -> int: 
        return self.agent.decay_params.get("learning_rate_frozen_steps", 0)
    @property
    def learning_rate_dict(self) -> Dict[str, Any]:
        return {
            "type": self.agent.decay_type,
            "params": self.agent.decay_params
        }
