from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
from typing import Dict

def mc_create_run_instance(args) -> tuple[int,Dict]:
    agent, episodes_in = args
    agent.train(episodes_in)
    return agent

def setup_mc_class(args) -> MonteCarloAgent:
    all_states,lr = args
    agent= MonteCarloAgent(lr,all_states)
    #agent.initialize_q_values()
    return agent
