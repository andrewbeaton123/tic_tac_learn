import pytest
import numpy as np
from pathlib import Path
from tic_tac_toe_model import TicTacToeModel
from tic_tac_toe_game import TicTacToe

@pytest.fixture
def sample_model():
    q_values = {
        (0, 0, 0, 0, 0, 0, 0, 0, 0): {0: 0.1, 1: 0.5, 2: 0.2},
        (1, 0, 0, 0, -1, 0, 0, 0, 0): {1: -0.5, 2: 0.8, 3: 0.1}
    }
    hyperparameters = {"alpha": 0.1, "gamma": 0.9}
    training_config = {"epochs": 100}
    meta_data = {"model_version": "1.0", "model_name": "test_model"}
    return TicTacToeModel(
        q_values=q_values,
        hyperparameters=hyperparameters,
        training_config=training_config,
        meta_data=meta_data
    )

def test_predict(sample_model):
    model_input = [{
        "current_player": 1,
        "game_state": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }]
    # Context is None for testing
    result = sample_model.predict(context=None, model_input=model_input)
    assert result["action"] == 1
    assert result["q_value"] == 0.5

def test_predict_missing_keys(sample_model):
    model_input = [{}]
    # Should default to player 1 and empty board
    result = sample_model.predict(context=None, model_input=model_input)
    assert result["action"] == 1
    assert result["q_value"] == 0.5

def test_get_action_untrained_state(sample_model):
    # State not in q_values
    untrained_state = [1, 1, 1, -1, -1, -1, 0, 0, 0]
    game = TicTacToe(1, np.reshape(untrained_state, (3, 3)))
    with pytest.raises(ValueError, match="Untrained game state encountered"):
        sample_model._get_action(game)

def test_get_model_uri(sample_model):
    uri = sample_model.get_model_uri()
    assert uri == "models:/test_model/1.0"

def test_get_model_signature(sample_model):
    sig = sample_model.get_model_signature()
    assert sig is not None
    assert "game_state" in sig.inputs.input_names()
    assert "action" in sig.outputs.input_names()

def test_get_input_example(sample_model):
    example = sample_model.get_input_example()
    assert isinstance(example, list)
    assert len(example) > 0
    assert "game_state" in example[0]

def test_save_and_load(sample_model, tmp_path):
    sample_model.save(tmp_path)

    new_model = TicTacToeModel(
        q_values={},
        hyperparameters={},
        training_config={},
        meta_data={}
    )
    new_model.load(tmp_path)
        
    assert len(new_model.q_values) == len(sample_model.q_values)
    assert (0, 0, 0, 0, 0, 0, 0, 0, 0) in new_model.q_values
    assert new_model.q_values[(0, 0, 0, 0, 0, 0, 0, 0, 0)][1] == pytest.approx(0.5)
    assert new_model.q_values[(1, 0, 0, 0, -1, 0, 0, 0, 0)][2] == pytest.approx(0.8)

def test_get_artifact_path(sample_model, tmp_path):
    with pytest.raises(ValueError, match="Model artifacts not available"):
        sample_model.get_artifact_path()

    sample_model.save(tmp_path)
    artifact_path = sample_model.get_artifact_path()
    assert "q_values_safetensors" in artifact_path
    assert artifact_path["q_values_safetensors"].endswith("Q_values.safetensors")

def test_load_file_not_found():
    model = TicTacToeModel(q_values={}, hyperparameters={}, training_config={}, meta_data={})
    with pytest.raises(FileNotFoundError):
        model.load(Path("/non/existent/path"))
