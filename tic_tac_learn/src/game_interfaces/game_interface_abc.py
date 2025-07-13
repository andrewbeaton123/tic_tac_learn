# creates the abc class that will be used as a universal game interface


from abc import ABC, abstractmethod
import logging 
from datetime import datetime

class GameInterface(ABC):
    
    @abstractmethod
    def __init__(self, current_player: int ,
                config_manager,
                game_state = None):
        pass
    
    
    @abstractmethod
    def make_move(self, position: int) -> bool:
        """ make a move and return true if successful"""
        pass

    @abstractmethod
    def get_state(self) -> any:
        """Get the current game state"""
        pass
    
    @abstractmethod
    def is_game_over(self) -> bool: 
        """ Check if the game is finished """
        pass

    @abstractmethod
    def check_player_is_valid(self,  player_number : int) -> bool :
        pass


    
