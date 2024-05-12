""" holds a function that generates all possible
game states for tic tac toe"""

import logging

def generate_all_states():
            all_states = []
            logging.info("Starting to generate all possible game states")
            for i in range(3**9):  # There are 3^9 possible states in a 3x3 Tic Tac Toe game
                state = []
                for j in range(9):  # Each state is a list of 9 numbers
                    state.append(i % 3)
                    i //= 3
                all_states.append(tuple(state))
            return all_states
