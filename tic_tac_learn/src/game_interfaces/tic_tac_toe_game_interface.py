from .game_interface_abc import GameInterface
from .utils  import InvalidPlayerError
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
    