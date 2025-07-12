# holds the config manager that will ingest and provide attributes from the config yaml


import yaml
import logging

from typing import Dict
from pathlib import Path
from .utils import MissingConfigError

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
    
    @property
    def current_game(self) -> str : 
        """ Gets the current game returning default values where game does not exist"""
        return self._config.get("app",{}).get("current_game","default")

    @current_game.setter
    def current_game(self, game_name: str) :
        """
        Sets the current game in the configuration.
        Args:
            game_name (str): The name of the game to set as current.
        Logs:
            - Warning if the 'app' key is missing in the configuration.
            - Error if the specified game is not configured.
            - Info when the current game is successfully set.
        """

        if "app" not in self._config:
            logging.warning("Potential config configuration issue app was set set before game name")
        
        if game_name not in self._config["app"]:
            logging.error(f"{game_name} is not a configured game please select from: {list(self._config["app"]["games"].keys())}")


        self._config["app"]["current_game"] = game_name
        
        logging.info(f"Current game set to f{game_name}")
    
    @property
    def app_config(self) ->  Dict | None :
        """
        Retrieves the application-specific configuration.
        Returns:
            dict | None: The configuration dictionary for the "app" section if present,
            otherwise an empty dictionary.
        """
        
        
        return self._config.get("app",{})
    
    def get_game_config(self, game_name :str = None) -> Dict:
        """
        Retrieves the configuration dictionary for a specified game.
        Args:
            game_name (str, optional): The name of the game to retrieve the configuration for.
                If None, uses the currently selected game (`self.current_game`).
        Returns:
            Dict: The configuration dictionary for the specified game if found.
            None: If the configuration for the specified game does not exist.
        Logs:
            A warning if the configuration for the specified game is not found.
        """

        if game_name is None : 
            game_name = self.current_game
        
        game_config = self._config.get("app", {}).get("games", {}).get(game_name.lower())

        if not game_config: 
            
            raise MissingConfigError(f"No config for game: {game_name} in current  config -available games are {list(self._config["app"]["games"].keys())}")
        
        return game_config