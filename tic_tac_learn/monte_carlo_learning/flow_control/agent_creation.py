from tic_tac_learn.monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
from typing import Dict
from tic_tac_learn.src.control import Config_2_MC
from tic_tac_learn.src.game_interfaces.tic_tac_toe_game_interface import TicTacToeGameInterface

def mc_create_run_instance(args):
    """Trains a Monte Carlo agent for Tic-Tac-Toe.

    Args:
        args: A tuple containing the learning rate, shared Q-values, shared returns, and config manager.
              The format is (learning_rate, shared_q_values, shared_returns, config_manager).

    Returns:
        None (updates shared Q-values directly).
    """
    learning_rate, shared_q_values, shared_returns, config_manager = args

    # Create a temporary agent for this process to run a single game
    agent = MonteCarloAgent(learning_rate, [], config_manager) # all_possible_states is not needed here
    agent.q_values = shared_q_values
    agent.returns = shared_returns

    agent.learn(TicTacToeGameInterface(1, config_manager), 1) # Run one episode

    # No return needed as shared_q_values and shared_returns are updated directly

  



