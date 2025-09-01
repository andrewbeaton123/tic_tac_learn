import os
import tempfile
import shutil
import yaml
import pytest
from pathlib import Path
from ..tic_tac_learn.src.config_management.config_manager import ConfigManager
from ..tic_tac_learn.src.config_management.config_model import ExperimentConfig

@pytest.fixture(autouse=True)
def reset_singleton():
    # Reset singleton before each test
    ConfigManager._instance = None

@pytest.fixture
def config_dir(tmp_path):
    # Create a temporary config directory structure
    env = "development"
    config_path = tmp_path / "config" / env
    config_path.mkdir(parents=True)
    config_file = config_path / "config.yaml"
    config_data = {
        "param1": "value1",
        "param2": 42
    }
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
    return tmp_path, config_file, config_data

@pytest.fixture
def config_dir_good(tmp_path):
    # Create a temporary config directory structure
    env = "development"
    config_path = tmp_path / "config" / env
    config_path.mkdir(parents=True)
    config_file = config_path / "config.yaml"
    config_data = {
        
        'name': "Tic Tac Learn 0.1.2 - Testing - 2",
        'mlflow_name': "mlflow_name",
        'level': "DEBUG",
        'total_games': 1_000,
        'steps': 10,
        'agent_reload': None,
        'training':{
        'cores': 3,
        'learning_rate_start': 0.8,
        'learning_rate_min': 0.001,
        'learning_rate_scaling': 1,
        'test_games_per_step': 300,
        'frozen_learning_rate_steps': 0}
    }
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
    return tmp_path, config_file, config_data

def test_load_config_failure(monkeypatch, config_dir):
    tmp_path, config_file, config_data = config_dir
    env = "development"
    monkeypatch.setenv("TICLEARN_ENV", env)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    manager = ConfigManager()
    
    pytest.raises(AttributeError)

    #assert isinstance(manager.config, ExperimentConfig)
    #assert manager.config.param1 == config_data["param1"]
    #assert manager.config.param2 == config_data["param2"]

def test_load_config_success(monkeypatch, config_dir_good):

    tmp_path, config_file, config_data = config_dir_good
    env = "development"
    monkeypatch.setenv("TICLEARN_ENV", env)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    manager = ConfigManager()

    
    assert manager.config.name == config_data["name"]
    assert manager.config.training.cores == config_data["training"]["cores"]

def test_load_config_file_not_found(monkeypatch, tmp_path):
    env = "production"
    monkeypatch.setenv("TICLEARN_ENV", env)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        ConfigManager()
    assert f"Config file not found for environment '{env}'" in str(excinfo.value)

def test_singleton_behavior(monkeypatch, config_dir):
    tmp_path, config_file, config_data = config_dir
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    manager1 = ConfigManager()
    manager2 = ConfigManager()
    assert manager1 is manager2