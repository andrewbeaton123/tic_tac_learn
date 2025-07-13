# holds the config manager that will ingest and provide attributes from the config yaml


import yaml
import logging

from typing import Dict, List
from pathlib import Path
from .utils import MissingConfigError, c

class ConfigManager:
    """Manages the game configs"""

    _instance = None 
    _config = None 

    def __new__ (cls):
        """Singleton pattern to enforce only one current config """

        if cls._instance is None : 
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "config.yml"):
        if self._initialized: 
            return 

        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._available_games = list(self._config.get("app", {}).get("games", {}).keys())
        self._initialized = True 

    def _load_config(self) -> Dict:
        "load the config from the yml"
        try: 
            if not self.config_path.exists():
                logging.warning(f"Config file {self.config_path} not found. Using default config")
                return self._get_default_config()
            

            with open(self.config_path, "r") as f : 
                config = yaml.safe_load(f)
                logging.info(f"Loaded config file from {self.config_path}")
                
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
            logging.warning("Potential configuration issue: 'app' key missing before setting game name")
        
        if game_name not in self._config["app"]["games"]:
            logging.error(f"{game_name} is not a configured game please select from: {list(self._config["app"]["games"].keys())}")


        self._config["app"]["current_game"] = game_name
        
        logging.info(f"Current game set to {game_name}")
    
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
            
            raise MissingConfigError(f"No config for game: {game_name} in current config - available games are {self._available_games}")
        
        return game_config
    
    def get_allowed_players(self, game_name: str  = None) -> List[int] | None:
        """
        game_config = self.get_game_config(game_name)
        Args:
            game_name (str, optional): The name of the game to retrieve the allowed players for.
                If None, uses the currently selected game (`self.current_game`).
        Returns:
            List[int] | None: A list of allowed players for the specified game if found,
            otherwise None.
        """
        game_config = self.get_game_config(game_name)
        return game_config.get("allowed_players", None)
    

    def get_player_range (self, 
                          game_name : str  = None 
                          ) -> tuple[int, int] :  
        
        game_config = self.get_game_config(game_name)

        min_players = game_config.get("min_players", None)
        max_players = game_config.get("max_players", None )
        
        if not min_players:
            raise MissingConfigError(f"min_players not set in config")
        
        if not max_players: 
            raise MissingConfigError(f"max_players not set in config")
        
        return min_players, max_players
    

    def get_all_games(self) -> List: 
        """Get the keys of the config games"""
        return list(self._config.get("games",{}).keys())


    def reload_config(self) -> None :
        """Trigger standard load_config"""
        self._config = self._load_config()
        logging.info("Config Reloaded ")

    @property
    def config(self) -> Dict : 
        """Return the current config dict"""
        return self._config

