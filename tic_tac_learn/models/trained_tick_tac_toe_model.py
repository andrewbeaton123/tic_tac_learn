
import numpy as np
import os 
import logging
import ast
    
from typing import Dict, List,Union
from .trained_model_abc import TrainedModelABC
from tic_tac_toe_game import TicTacToe
from pathlib import Path
from safetensors.numpy  import save_file, load_file
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, ParamSchema

class TicTacToeModelMonteCarlo(TrainedModelABC):


    def __init__(self,
                 q_values: Dict, 
                 hyperparameters: Dict, 
                 training_config: Dict,
                 meta_data: Dict):
         
        super().__init__(
             model_data= q_values,
             game_interface_type="tic_tac_toe", 
             hyperparameters=hyperparameters, 
             training_config=training_config, 
             meta_data = meta_data
        )
        self.q_values = q_values
        self.artifact_dir = None
    # meta data example placeholder 
    # "training_date": "2025-04-11",
    # "games_trained": 100000,
    # "win_rate_vs_random": 0.98,
    # "author": "andrew",
    # "model_version": "2.0",
    # "opponent_type": "random",
    # "tags": ["production", "high-win-rate"],
    # "experiment_id": "exp_123"

    def predict(self, context, 
                model_input: List[Dict[str,Union[int,List[int]]]]) -> Dict :
        
        game_state = model_input[0].get("game_state",[0,0,0,0,0,0,0,0,0])
        current_player = model_input[0].get("current_player", 1)
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
            return {"action": best_action,
                    "q_value" : float(actions_q_values[best_action])}
        else:
            raise ValueError(f"Untrained game state encountered : {state_key}")
    

    def save(self,
            save_folder_path : Path|None ):
        
        # Safetensors requires numeric arrays. Convert dictionary of actions to a 2D float array [[action, q_value], ...]
        tensors = {str(k): np.array(list(v.items()), dtype=np.float32) for k, v in self.q_values.items()}
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
            "model_version": self.meta_data.get("model_version", "Model Version Not Specified")
        }
    )
        self.artifact_dir = save_folder_path or Path(".")
        
    
    def load(self, load_folder_path: Path | None):
        """Load Q-values from safetensors file.
        
        Args:
            load_folder_path: Path to folder containing Q_values.safetensors, or None for current directory.
        """

        filepath = os.path.join(load_folder_path or ".", "Q_values.safetensors")
        
        try:
            tensors = load_file(filepath)
            
            # Convert string keys back to tuples and recreate the {action: q_value} dictionaries
            self.q_values = {
                ast.literal_eval(k): {int(action): float(q) for action, q in arr}
                for k, arr in tensors.items()
            }
            
            num_states = len(self.q_values)
            
            logging.info(
                f"Loaded Q-values model",
                extra={
                    "filepath": filepath,
                    "num_states": num_states,
                    "model_version": self.meta_data.get("model_version", "Model Version Not Specified")
                }
            )
            self.artifact_dir = load_folder_path or Path(".")
        except FileNotFoundError:
            logging.error(f"Q-values file not found at {filepath}")
            raise
        except Exception as e:
            logging.error(f"Error loading Q-values from {filepath}: {e}")
            raise

    def get_artifact_path(self) -> Dict[str, str]:
        """
        Returns a dictionary mapping artifact names to their file paths.
        
        Used for MLflow compatibility to locate model artifacts for logging.
        
        Returns:
            Dict[str, str]: Dictionary with artifact names as keys and absolute file paths as values.
            
        Raises:
            ValueError: If the model has not been saved or loaded yet.
        """
        if self.artifact_dir is None:
            raise ValueError("Model artifacts not available. Call save() or load() first.")
        
        return {
            "q_values": os.path.join(str(self.artifact_dir), "Q_values.safetensors")
        }
    

    def get_model_signature(self) -> ModelSignature:

        input_schema = Schema([
            ColSpec("integer", "current_player"),
            ColSpec("integer", "game_state")
        ])

        output_schema = Schema(
            [
                ColSpec("integer", "action"),
                ColSpec("float", "q_value")
            ]
        )

        return ModelSignature(inputs=input_schema, 
                              outputs=output_schema)
    

    def get_input_example(self) -> List[Dict]:
        return [{
            "current_player" : 1, 
            "game_state": [0, 0, 0, 0, 1, 0, 0, 0, 0]
        }]
    

    def get_model_uri(self) -> str:
        """
        Return the MLflow Model Registry address for this model.
        
        Format: models:/<model_name>/<version>
        Example: models:/tictactoe-agent/1
        
        This allows:
        - Version control (v1, v2, v3 of the same model)
        - Production promotion (dev → staging → prod)
        - Rollback if new version is bad
        """
        model_name = self.meta_data.get("model_name", "tictactoe-agent")
        model_version = self.meta_data.get("model_version", "1.0")
        
        return f"models:/{model_name}/{model_version}"