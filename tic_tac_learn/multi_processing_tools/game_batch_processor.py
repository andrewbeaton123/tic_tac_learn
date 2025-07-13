import logging
from tic_tac_learn.monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
from tic_tac_learn.src.game_interfaces.tic_tac_toe_game_interface import TicTacToeGameInterface
from tic_tac_learn.src.control import Config_2_MC

def process_game_batch(args):
    """
    Processes a batch of games for a Monte Carlo agent, updating shared Q-values.

    Args:
        args: A tuple containing the learning rate, shared Q-values, shared returns,
              config manager, number of games to simulate, and all possible states.
              The format is (learning_rate, shared_q_values, shared_returns, config_manager, num_games_to_simulate, all_possible_states).
    """
    learning_rate, shared_q_values, shared_returns, config_manager, num_games_to_simulate, all_possible_states= args

    # Create a temporary agent for this process to run a single game
    agent = MonteCarloAgent(learning_rate, all_possible_states, config_manager) 
    agent.q_values = shared_q_values
    agent.returns = shared_returns

    

    agent.learn(TicTacToeGameInterface(1, config_manager), num_games_to_simulate) # Run one episode

    logging.debug(f"Process finished simulating {num_games_to_simulate} games.")