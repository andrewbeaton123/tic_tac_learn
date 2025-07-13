
import logging 

from datetime import datetime
from src.config_management  import ConfigManager
cm = ConfigManager()

class GameError(Exception):
    """ Base exception for all game errors"""
   
    def __init__(self, message,code=None):
        self.message = message
        self.code = code
        self.timestamp = datetime.now()
        super().__init__(self.message)
        self.log_error()

    def log_error(self):
        logging.error(f"[{self.timestamp}] - ConfigError ({self.code}) : {self.message}")


class InvalidPlayerError(GameError):
    """Raised when player is set to something other than 1 or 2"""
    logging


def validate_player_numbers(player_number):

    """Ensures that the players is allowed"""
    #TODO work with claude example to write the player nuimber ingestion from yaml

    if player_number not in cm.get_allowed_players():
        raise InvalidPlayerError(f"Invalid player {player_number} is not in {cm.get_allowed_players()}")

    
