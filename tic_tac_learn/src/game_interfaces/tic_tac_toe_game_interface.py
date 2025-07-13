# game interface  tic tac toe that follows the game interface ABC

import logging 
from .game_interface_abc import GameInterface
from .utils  import InvalidPlayerError
from typing import Any
from tic_tac_toe_game.game import TicTacToe


class TicTacToeGameInterface(GameInterface):

    def __init__ (self, current_player: int ,
                  config_manager,
                   game_state = None):
        
        self.config_manager = config_manager
        self.current_player = current_player
        
        if game_state is None:
            self.game = TicTacToe(current_player)
        else:
            # Convert tuple game_state to a list of lists for TicTacToe
            board_list = [list(game_state[i:i+3]) for i in range(0, 9, 3)]
            self.game = TicTacToe(current_player, board=board_list)
        
        if not self.check_player_is_valid(self.current_player):
            raise InvalidPlayerError(f"Invalid player {self.current_player} is not in {self.config_manager.get_allowed_players()}")
    
    def check_player_is_valid(self, player_number: int ) -> bool:
        # Assuming config_manager has a method to get allowed players
        return  player_number in self.config_manager.get_allowed_players()
    

    def make_move(self, position: int) -> bool:
        """Make a move at the specified position."""
        logging.debug(f"Player {self.current_player} attempting move at position {position}")
        
        row = position // 3
        col = position % 3
        
        try:
            self.game.make_move(row, col)
            logging.info(f"Player {self.current_player} successfully moved to position {position}")
            return True
        except ValueError as e: # Catch specific ValueError for invalid moves
            logging.warning(f"Player {self.current_player} failed to move to position {position} (invalid move): {e}")
            return False
        except Exception as e:
            logging.error(f"Error making move for player {self.current_player} at position {position}: {e}")
            return False
    
    def get_state(self) -> Any:
        """Get the current board state as a tuple of integers."""
        try:
            state = tuple(int(cell) for row in self.game.board for cell in row)
            logging.debug(f"Retrieved game state for player {self.current_player}: {state}")
            return state
        except Exception as e:
            logging.error(f"Error getting game state for player {self.current_player}: {e}")
            return None
    
    def is_game_over(self) -> bool:
        """Check if the game is finished."""
        try:
            game_over = self.game.is_game_over()
            logging.debug(f"Game over status for player {self.current_player}: {game_over}")
            return game_over
        except Exception as e:
            logging.error(f"Error checking game over status for player {self.current_player}: {e}")
            return True

    def get_valid_moves(self) -> list[int]:
        """Get a list of valid moves as integer positions."""
        valid_moves_rc = self.game.get_valid_moves()
        return [int(move[0] * 3 + move[1]) for move in valid_moves_rc]

    def get_winner(self) -> int:
        """Get the winner of the game (0 for draw, 1 or 2 for player)."""
        return self.game.winner
