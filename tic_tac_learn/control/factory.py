import yaml
import logging
from typing import Dict, Any
from pathlib import Path
from .schemas import MonteCarloConfig, EnvConfig, AgentConfig, RunnerConfig



def _validate_and_create_directories(config: Dict) -> None:
    """Create required directories from config upfront.
    
    Extracts directory paths from app.paths config and creates them.
    Fails fast during initialization rather than during training.
    """
    app_settings = config.get("app",{}).get("paths", {})
    required_dirs = [
        app_settings.get("q_tables_dir", "q_tables"),
        app_settings.get("logs_dir", "logs"),
        app_settings.get("models_dir", "models")
    ]
    
    for dir_path in required_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logging.info(f"Created/Validated directory: {dir_path}")
        except Exception as e:
            logging.error (f"Failed to create directory {dir_path}:{e}")
            raise


def load_config(config_path: str = "tic_tac_learn/config.yml") -> MonteCarloConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    app_settings = raw.get("app", {})
    game_id = app_settings.get("current_game", "tic_tac_toe")
    
    # Get game specific settings
    game_meta = raw.get("games", {}).get(game_id, {})
    allowed_players = game_meta.get("allowed_players", [1, 2])
    
    # Monte Carlo specific settings
    mc_settings = raw.get("monte_carlo_settings", {})
    
    _validate_and_create_directories(raw)
    
    # Handle Decay Merging (Logic moved from Singleton to Factory)
    decay_config = mc_settings.get("decay", {})
    decay_type = decay_config.get("type", "LINEAR")
    decay_params = decay_config.get("params", {})
    
    env_conf = EnvConfig(
        game_id=game_id,
        allowed_players=allowed_players,
        training_player=mc_settings.get("training_player", 1)
    )
    
    agent_conf = AgentConfig(
        learning_rate_inital=decay_params.get("learning_rate_inital", 0.7),
        learning_rate_min=decay_params.get("learning_rate_min", 0.01),
        decay_type=decay_type,
        decay_params=decay_params,
        discount_factor=mc_settings.get("discount_factor", 0.9),
        exploration_rate=mc_settings.get("exploration_rate", 0.1)
    )
    
    runner_conf = RunnerConfig(
        run_name=mc_settings.get("run_name", "Default Run"),
        experiment_name=mc_settings.get("experiment_name", "Tic Tac Learn"),
        total_games=mc_settings.get("total_games", 10000),
        steps=mc_settings.get("steps", 10),
        cores=mc_settings.get("cores", 1),
        test_games_per_step=mc_settings.get("test_games_per_step", 1000)
    )
    
    return MonteCarloConfig(
        env=env_conf,
        agent=agent_conf,
        runner=runner_conf,
        raw_config=raw
    )
