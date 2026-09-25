"""Config parsing for the tracking block, and a full (tiny) training run with no MLflow server."""
import pytest
import yaml

from tic_tac_learn.control.factory import load_config
from tic_tac_learn.monte_carlo_learning.flow_control.run_monte_carlo import run_parallel_training
from tic_tac_learn.tracking import NullTracker

BASE_CONFIG = {
    "app": {"paths": {"q_tables_dir": "q_tables", "logs_dir": "logs", "models_dir": "models"}},
    "games": {"tic_tac_toe": {"allowed_players": [1, 2]}},
    "monte_carlo_settings": {
        "run_name": "offline test",
        "experiment_name": "Tic Tac Test",
        "total_games": 40,
        "steps": 2,
        "cores": 1,
        "test_games_per_step": 10,
        "decay": {"type": "LINEAR",
                  "params": {"learning_rate_frozen_steps": 0,
                             "learning_rate_scaling": 1,
                             "learning_rate_inital": 0.5,
                             "learning_rate_min": 0.01}},
        "discount_factor": 0.9,
        "exploration_rate": 0.3,
        "training_player": 1,
    },
}


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _write(extra=None):
        config = {**BASE_CONFIG, **(extra or {})}
        path = tmp_path / "config.yml"
        path.write_text(yaml.safe_dump(config))
        return str(path)

    return _write


class TestTrackingConfig:

    def test_defaults_when_tracking_block_missing(self, write_config):
        runner = load_config(write_config()).runner
        assert runner.log_mlflow is True
        assert runner.mlflow_tracking_uri is None
        assert runner.mlflow_on_unavailable == "warn"
        assert runner.mlflow_health_check_timeout == 3.0

    def test_tracking_block_is_parsed(self, write_config):
        runner = load_config(write_config({"tracking": {
            "log_mlflow": False,
            "mlflow_tracking_uri": "http://example.invalid",
            "on_unavailable": "FAIL",
            "health_check_timeout_seconds": 1,
        }})).runner
        assert runner.log_mlflow is False
        assert runner.mlflow_tracking_uri == "http://example.invalid"
        assert runner.mlflow_on_unavailable == "fail"
        assert runner.mlflow_health_check_timeout == 1.0

    def test_invalid_on_unavailable_raises(self, write_config):
        with pytest.raises(ValueError, match="on_unavailable"):
            load_config(write_config({"tracking": {"on_unavailable": "ignore"}}))

    def test_tracking_settings_not_in_logged_params(self, write_config):
        conf = load_config(write_config({"tracking": {"log_mlflow": False}}))
        assert "tracking" not in conf.raw_config["monte_carlo_settings"]


def test_training_runs_without_mlflow_and_saves_model_locally(write_config, tmp_path):
    conf = load_config(write_config({"tracking": {"log_mlflow": False}}))

    run_parallel_training(conf, NullTracker())

    saved_models = list((tmp_path / "models").glob("*/Q_values.safetensors"))
    assert len(saved_models) == 1
    assert saved_models[0].parent.name.startswith("Tic_Tac_Test_offline_test_")
    assert len(list((tmp_path / "q_tables").glob("q_table_step_*.pkl"))) == 2
