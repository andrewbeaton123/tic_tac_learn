

import logging

from src.config_management import ConfigManager
class GameInterfaceFactory:
    """Factory for creating game interfaces with proper validation."""
    
    @staticmethod
    def create_game_interface(game_type: str = None, player_id: int = 1, *args, **kwargs):
        """Create a game interface based on game type."""
        # If no game_type provided, use current game from config
        if game_type is None:
            game_type = config_manager.get_current_game()
            logger.info(f"Using current game from config: {game_type}")
        
        game_type = game_type.lower()
        
        if game_type == "tic_tac_toe":
            #TODO Change this approach so that it can handle game additons 
            return TicTacToeGameInterface(player_id, game_name=game_type, *args, **kwargs)
        elif game_type == "chess":
            # Future implementation
            # return ChessGameInterface(player_id, game_name=game_type, *args, **kwargs)
            raise NotImplementedError(f"Game type '{game_type}' not implemented yet")
        elif game_type == "poker":
            # Future implementation
            # return PokerGameInterface(player_id, game_name=game_type, *args, **kwargs)
            raise NotImplementedError(f"Game type '{game_type}' not implemented yet")
        else:
            raise ValueError(f"Unknown game type: {game_type}")
    
    @staticmethod
    def create_current_game_interface(player_id: int, 
                                      config_manager :ConfigManager,
                                    *args, **kwargs):
        """Create a game interface for the current game in config."""
        current_game = config_manager.get_current_game()
        return GameInterfaceFactory.create_game_interface(current_game, player_id, *args, **kwargs)