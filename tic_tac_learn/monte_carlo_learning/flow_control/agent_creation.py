from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
from typing import Dict

def mc_create_run_instance(args) -> tuple[int,Dict]:
    """Trains a Monte Carlo agent for Tic-Tac-Toe.

    Args:
        args: A tuple containing the Monte Carlo agent and the number of episodes to train for. 
              The format is (agent, episodes_in).

    Returns:
        The trained Monte Carlo agent.
    """
    agent, episodes_in = args # Unpack the agent and number of episodes from the input tuple
    agent.check_q_value_space_exists() # Ensure that the space for Q-values exists for the agent
    agent.train(episodes_in) # Train the agent for the specified number of episodes
    return agent # Return the trained agent

def setup_mc_class(args) -> MonteCarloAgent:
    """Initializes and configures a Monte Carlo Agent for Tic-Tac-Toe.

    Args:
        args (tuple): A tuple containing two elements:
            - all_states (list): A list representing all possible states in the game.
            - lr (float): The learning rate used by the agent.

    Returns:
        MonteCarloAgent: A trained Monte Carlo Agent ready for use.
    """
    all_states,lr = args  # Unpack the learning rate and all possible states 
    agent= MonteCarloAgent(lr,all_states) # Initialize a new Monte Carlo Agent with the given parameters.
    agent.check_q_value_space_exists() # Ensure space for q-values exists before training
    #agent.initialize_q_values() # Uncomment this line if you want to initialize q-values manually
    return agent  



