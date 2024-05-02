
from typing import NamedTuple

class ConfigClass(NamedTuple):
    """class the creats a named tuple which 
    hold ths settings for a monte carlo tic tac toe learnign
    model to be trianed in. 

    Args:
        NamedTuple (_type_): settings for monte carlo training run
        for tic tac toe
    """
    cores: int
    steps: int
    total_training_games: int
    test_games: int
    learning_rate: list
    run_name : str