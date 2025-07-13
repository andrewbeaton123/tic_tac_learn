# game interface  tic tac toe that follows the game interface ABC

import logging 
from .game_interface_abc import GameInterface
from .utils  import InvalidPlayerError
from typing import Any


class TicTacToeGameInterface(GameInterface):

    def __init__ (self, current_player: int ,
                  config_manager,
                   game_state = None):
        
        self.config_namager =config_manager
        self.current_player = current_player
        self.game_sate = game_state
        
        if not self.check_player_is_valid(self.current_player):
            raise InvalidPlayerError(f"Invalid player {self.current_player} is not in {self.config_namager.get_allowed_players()}")
        #TODO finish the game interface once the player id error handling is built
    
    def check_player_is_valid(self, player_number: int ) -> bool:
        return  player_number in self.config_namager.get_allowed_players()
    

    def make_move(self, position: int) -> bool:
        """Make a move at the specified position."""
        logging.debug(f"Player {self.player_id} attempting move at position {position}")
        
        try:
            result = self.game.make_move(position)
            if result:
                logging.info(f"Player {self.player_id} successfully moved to position {position}")
            else:
                logging.warning(f"Player {self.player_id} failed to move to position {position} (invalid move)")
            return result
        except Exception as e:
            logging.error(f"Error making move for player {self.player_id} at position {position}: {e}")
            return False
    
    def get_state(self) -> Any:
        """Get the current board state."""
        try:
            state = self.game.get_board()
            logging.debug(f"Retrieved game state for player {self.player_id}")
            return state
        except Exception as e:
            logging.error(f"Error getting game state for player {self.player_id}: {e}")
            return None
    
    def is_game_over(self) -> bool:
        """Check if the game is finished."""
        try:
            game_over = self.game.is_game_over()
            logging.debug(f"Game over status for player {self.player_id}: {game_over}")
            return game_over
        except Exception as e:
            logging.error(f"Error checking game over status for player {self.player_id}: {e}")
            return True
