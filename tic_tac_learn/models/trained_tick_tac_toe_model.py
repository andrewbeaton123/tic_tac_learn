
from .trained_model_abc import TrainedModelABC
from tic_tac_toe_game import TicTacToe
import numpy as np

class TicTacToeModel(TrainedModelABC):

    def predict(self, 
                game_state, 
                current_player : int) -> int :
        current_game = TicTacToe(current_player,
                            np.reshape(game_state, (3, 3)))
        
        

        return self._get_action(current_game)
    

    def _get_state(self,
                   env: TicTacToe) -> tuple:
            """
            Gets the current state of the environment as a hashable tuple.
            """
            return tuple(int(x) for x in env.board.reshape(-1))
    

    def _get_action(self, game_state: TicTacToe) -> int:

        state_key = self._get_state(game_state)

        if state_key in self.q_values:
            actions_q_values = self.q_values[state_key]
            # Find the action with the maximum Q-value
            best_action = max(actions_q_values, key=actions_q_values.get)
            return best_action
        else:
            raise ValueError(f"Untrained game state encountered : {state_key}")