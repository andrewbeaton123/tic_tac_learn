
from monte_carlo_learning.monte_carlo_tic_tac_2 import MonteCarloAgent
from typing import Dict

class SuperCarloAgent(MonteCarloAgent):
    def __init__(self,
                 q_values:Dict,
                 epsilon: float):
        self.q_values = q_values
        self.returns = {}
        self.epsilon = epsilon
        self.standard_q_length = 9
    def to_serializable(self):
        return {
            'q_values': {str(k): v for k, v in self.q_values.items()},
            'epsilon': self.epsilon
        }
    