from unittest import mock

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from tic_tac_learn.control.schemas import RunnerConfig
from tic_tac_learn.errors.MlflowUnavailableError import MlflowUnavailableError
from tic_tac_learn.tracking import MlflowTracker, NullTracker, create_tracker, mlflow_server_reachable

UNREACHABLE_URI = "http://127.0.0.1:1"  # nothing listens on port 1 -> connection refused immediately


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """A throwaway local sqlite tracking store; artifacts land under tmp_path."""
    monkeypatch.chdir(tmp_path)
    original_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    yield uri
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(original_uri)


def _only_run(uri, experiment_name):
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    return runs[0]


class TestCreateTracker:

    def test_disabled_in_config_returns_null_tracker(self):
        conf = RunnerConfig(log_mlflow=False, mlflow_tracking_uri=UNREACHABLE_URI)
        assert isinstance(create_tracker(conf), NullTracker)

    def test_unreachable_server_warn_returns_null_tracker(self):
        conf = RunnerConfig(mlflow_tracking_uri=UNREACHABLE_URI, mlflow_on_unavailable="warn")
        assert isinstance(create_tracker(conf), NullTracker)

    def test_unreachable_server_fail_raises(self):
        conf = RunnerConfig(mlflow_tracking_uri=UNREACHABLE_URI, mlflow_on_unavailable="fail")
        with pytest.raises(MlflowUnavailableError):
            create_tracker(conf)

    def test_local_store_returns_mlflow_tracker(self, local_mlflow):
        conf = RunnerConfig(mlflow_tracking_uri=local_mlflow)
        assert isinstance(create_tracker(conf), MlflowTracker)

    def test_non_http_uris_skip_health_check(self):
        assert mlflow_server_reachable("sqlite:///some.db")
        assert mlflow_server_reachable("file:./mlruns")


class TestNullTracker:

    def test_all_calls_are_noops(self):
        tracker = NullTracker()
        with tracker.start_run():
            tracker.log_params({"a": 1})
            tracker.log_metrics({"m": 1.0}, step=0)
            tracker.log_artifact("does/not/exist.pkl", "q_tables")
            tracker.log_model(object(), "model")


class TestMlflowTracker:

    def test_logs_params_and_metrics_and_finishes(self, local_mlflow):
        tracker = MlflowTracker(local_mlflow, "tracker-test", "run-1")
        with tracker.start_run():
            tracker.log_params({"steps": 2})
            tracker.log_metrics({"test_win_percentage": 50.0}, step=0)

        run = _only_run(local_mlflow, "tracker-test")
        assert run.info.status == "FINISHED"
        assert run.data.params["steps"] == "2"
        assert run.data.metrics["test_win_percentage"] == 50.0

    def test_training_exception_marks_run_failed_and_propagates(self, local_mlflow):
        tracker = MlflowTracker(local_mlflow, "tracker-fail-test", "run-1")
        with pytest.raises(RuntimeError, match="training broke"):
            with tracker.start_run():
                raise RuntimeError("training broke")

        assert _only_run(local_mlflow, "tracker-fail-test").info.status == "FAILED"

    def test_mid_run_failure_disables_tracking(self, local_mlflow):
        tracker = MlflowTracker(local_mlflow, "tracker-drop-test", "run-1")
        with tracker.start_run():
            with mock.patch("mlflow.log_metrics", side_effect=ConnectionError("server gone")):
                tracker.log_metrics({"m": 1.0}, step=0)  # must not raise

            with mock.patch("mlflow.log_params") as log_params:
                tracker.log_params({"a": 1})
                log_params.assert_not_called()

    def test_log_model_error_does_not_escape(self, local_mlflow):
        broken_model = mock.Mock()
        broken_model.get_model_signature.side_effect = TypeError("bad signature")
        tracker = MlflowTracker(local_mlflow, "tracker-model-test", "run-1")
        with tracker.start_run():
            tracker.log_model(broken_model, "model")  # must not raise

        assert _only_run(local_mlflow, "tracker-model-test").info.status == "FINISHED"

    def test_start_failure_warn_continues_untracked(self):
        tracker = MlflowTracker(None, "exp", "run", on_unavailable="warn")
        with mock.patch("mlflow.set_experiment", side_effect=ConnectionError("down")):
            with tracker.start_run():
                with mock.patch("mlflow.log_metrics") as log_metrics:
                    tracker.log_metrics({"m": 1.0})
                    log_metrics.assert_not_called()

    def test_start_failure_fail_raises(self):
        tracker = MlflowTracker(None, "exp", "run", on_unavailable="fail")
        with mock.patch("mlflow.set_experiment", side_effect=ConnectionError("down")):
            with pytest.raises(MlflowUnavailableError):
                with tracker.start_run():
                    pass
