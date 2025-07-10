class GameError(Exception):
    """ Base exception for all game errors"""
    pass

class InvalidPlayerError(GameError):
    """Raised when player is set to something other than 1 or 2"""
    pass


def validate_player_numbers(player_number):

    """Ensures that the players is allowed"""
    #TODO work with claude example to write the player nuimber ingestion from yaml

    if player_number not in []

