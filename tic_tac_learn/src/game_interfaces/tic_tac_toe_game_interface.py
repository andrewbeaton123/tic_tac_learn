from .game_interface_abc import GameInterface


class TicTacToeGameInterface(GameInterface):

    def __init__ (self, current_player: int , board = None):
        
        #TODO finish the game interface once the player id error handling is built