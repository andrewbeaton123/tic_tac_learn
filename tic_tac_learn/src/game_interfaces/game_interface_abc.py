# creates the abc class that will be used as a universal game interface


from abc import ABC, abstractmethod


class GameInterface(ABC):

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