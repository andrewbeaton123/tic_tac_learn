# holds the config manager that will ingest and provide attributes from the config yaml


import yaml
import logging

from typing import Dict
from pathlib import Path

class ConfigManager:
    """Manages the game configs"""

    _instance = None 
    _config = None 

    def __new__ (cls, config_path : str = "config.yml"):
        """Singleton pattern to enforce only one current config """

        if cls._instance is None : 
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instnace

    def __init__(self, config_path: str = "config.yml"):
        if self._initialized: 
            return 

        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._iniialized = True 

    def _load_config(self) -> Dict:
        "load the config from the yml"
        try: 
            if not self.config_path.exists():
                logging.warning(f"Config file {self.config_path} not found. Using default config")
                return self._get_defualt_config()
            

            with open(self.config_path, "r") as f : 
                config = yaml.safe_load(f)
                logging.info(f"Loaded config file from {self.config_p}")
                
                return config or self._get_default_config()
        
        except Exception as e:
            logging.error (f"Error during config load {e} - Using default config")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'app': {
                'current_game': 'tic_tac_toe',
                'environment': 'development'
            },
            'games': {
                'tic_tac_toe': {
                    'allowed_players': [1, 2],
                    'min_players': 2,
                    'max_players': 2,
                    'display_name': 'Tic Tac Toe'
                }
            },
            'default': {
                'allowed_players': [1, 2],
                'min_players': 2,
                'max_players': 2,
                'display_name': 'Generic Game'
            }
        }