
import numpy as np
import os 
import logging
import ast
    
from typing import Dict
from .trained_model_abc import TrainedModelABC
from tic_tac_toe_game import TicTacToe
from pathlib import Path
from safetensors.numpy  import save_file, load_file

class TicTacToeModel(TrainedModelABC):


    def __init__(self,
                 q_values: Dict, 
                 metadata: Dict, 
                 hyperparameters: Dict, 
                 training_config: Dict,
                 meta_data: Dict):
         
        super().__init__(
             model_data= q_values, 
             meta_data=metadata, 
             game_interface_type="tic_tac_toe", 
             hyperparameters=hyperparameters, 
             training_config=training_config, 
             meta_data = meta_data
        )
    # meta data example placeholder 
    # "training_date": "2025-04-11",
    # "games_trained": 100000,
    # "win_rate_vs_random": 0.98,
    # "author": "andrew",
    # "model_version": "2.0",
    # "opponent_type": "random",
    # "tags": ["production", "high-win-rate"],
    # "experiment_id": "exp_123"

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
    

    def save(self,
            save_folder_path : Path|None ):
        
        tensors = {str(k): np.array(v) for k ,v in self.q_values.items()}
        file_size = sum(t.nbytes for t in tensors.values())

        if save_folder_path:
            save_file(tensors , os.path.join(save_folder_path, "Q_values.safetensors"))
        else:
            save_file(tensors, "Q_values.safetensors")
        
        logging.info(
        f"Saving Q-values model",
        extra={
            "filepath": save_folder_path,
            "num_states": len(tensors),
            "file_size_mb": file_size / (1024**2),
            "model_version": self.get_metadata().get("model_version","Model Versiion Not Specified")
        }
    )
        
    
    def load(self, load_folder_path: Path | None):
        """Load Q-values from safetensors file.
        
        Args:
            load_folder_path: Path to folder containing Q_values.safetensors, or None for current directory.
        """

        filepath = os.path.join(load_folder_path or ".", "Q_values.safetensors")
        
        try:
            tensors = load_file(filepath)
            
            # Convert string keys back to tuples
            self.q_values = {ast.literal_eval(k): v for k, v in tensors.items()}
            
            num_states = len(self.q_values)
            
            logging.info(
                f"Loaded Q-values model",
                extra={
                    "filepath": filepath,
                    "num_states": num_states,
                    "model_version": self.get_metadata().get("model_version", "Model Version Not Specified")
                }
            )
        except FileNotFoundError:
            logging.error(f"Q-values file not found at {filepath}")
            raise
        except Exception as e:
            logging.error(f"Error loading Q-values from {filepath}: {e}")
            raise