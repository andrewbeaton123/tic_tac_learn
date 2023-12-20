""" holds a function that generates all possible
game states for tic tac toe"""

import logging
import pickle as pkl
import datetime
from itertools import product
import numpy as np
from game.game_2 import TicTacToe


def generate_all_states() -> list:
    """generates all possible states in the game of tic tac toe

    Returns:
        list: holds the instances of tic tac toe that have a single possible 
        game board state
    """
    states = []
    try: 
        with open("all_possible_states_classess.pkl", "rb") as fl2:
            all_possible_states = pkl.load(fl2)

    except Exception:
        logging.info(f"main - Failed to load all states generating now -{datetime.now()}")
        all_possible_states = generate_all_states()

    with open("all_possible_states_classess.pkl","wb") as fl :
        pkl.dump(all_possible_states,fl)
        logging.info(f"main - length of all states is {len(all_possible_states)}")
    logging.debug("Geenerate_all_states  is starting ")
    for player_marks in product([0, 1, 2], repeat=9):  # 0 represents an empty cell
        board = np.array(player_marks).reshape((3, 3))
        tic_tac_toe_instance = TicTacToe(1,board)
        states.append(tic_tac_toe_instance)
    logging.debug("Geenerate_all_states  is finishing ")
    return states
        

        