from pathlib import Path
import yaml
import os
from .config_model import ExperimentConfig

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.config = self.load_config()
    
    def load_config(self) -> ExperimentConfig:
        # Get environment from env var or default to development
        env = os.getenv('TICLEARN_ENV', 'development').lower()
        
        # Construct path to config file
        config_dir = Path(Path.cwd(), "config", env)
        config_path = config_dir / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found for environment '{env}' at {config_path}"
            )

        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
            return ExperimentConfig(**yaml_config)