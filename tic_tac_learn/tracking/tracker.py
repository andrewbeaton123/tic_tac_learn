"""
Experiment tracking abstraction.

Training code talks to an ExperimentTracker instead of calling mlflow directly, so a run can
proceed when no MLflow server is reachable (NullTracker), and a server that drops out mid-run
disables tracking instead of stalling or crashing training (MlflowTracker).
"""
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from tic_tac_learn.errors.MlflowUnavailableError import MlflowUnavailableError

logger = logging.getLogger(__name__)

ON_UNAVAILABLE_OPTIONS = ("warn", "fail")


class ExperimentTracker(ABC):

    @abstractmethod
    def start_run(self):
        """Context manager wrapping the whole training run."""

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def log_model(self, model: Any, name: str) -> None:
        """Log a model that has already been saved to disk (TicTacToeModel after save())."""


class NullTracker(ExperimentTracker):
    """Tracker that records nothing. Used when MLflow is disabled or unreachable."""

    @contextmanager
    def start_run(self) -> Iterator[None]:
        logger.info("Experiment tracking is off: metrics are only written to the log.")
        yield

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass

    def log_model(self, model: Any, name: str) -> None:
        pass


class MlflowTracker(ExperimentTracker):
    """
    MLflow-backed tracker. The first failed tracking call is logged and disables tracking for
    the rest of the run, so a lost server never kills (or repeatedly stalls on retries) training.
    """

    def __init__(self,
                 tracking_uri: Optional[str],
                 experiment_name: str,
                 run_name: str,
                 on_unavailable: str = "warn"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.on_unavailable = on_unavailable
        self._active = False

    @contextmanager
    def start_run(self) -> Iterator[None]:
        import mlflow

        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name=self.experiment_name)
            run = mlflow.start_run(run_name=self.run_name)
        except Exception as e:
            if self.on_unavailable == "fail":
                raise MlflowUnavailableError(f"Could not start MLflow run: {e}") from e
            logger.warning(f"Could not start MLflow run, continuing without tracking: {e}")
            yield
            return

        logger.info(f"MLflow run started (ID: {run.info.run_id})")
        self._active = True
        status = "FINISHED"
        try:
            yield
        except BaseException:
            status = "FAILED"
            raise
        finally:
            self._active = False
            try:
                mlflow.end_run(status=status)
            except Exception as e:
                logger.warning(f"Failed to end MLflow run cleanly: {e}")

    def _call(self, description: str, fn: Callable, *args, **kwargs) -> None:
        if not self._active:
            return
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"MLflow {description} failed, disabling tracking for the rest of this run: {e}")
            self._active = False

    def log_params(self, params: Dict[str, Any]) -> None:
        import mlflow
        self._call("log_params", mlflow.log_params, params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        import mlflow
        self._call("log_metrics", mlflow.log_metrics, metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        import mlflow
        self._call("log_artifact", mlflow.log_artifact, local_path, artifact_path=artifact_path)

    def log_model(self, model: Any, name: str) -> None:
        import mlflow

        # Build signature/artifacts inside _call so a model-side error can't escape and kill the run
        def _log():
            mlflow.pyfunc.log_model(
                artifact_path=name,
                python_model=model,
                artifacts=model.get_artifact_path(),
                signature=model.get_model_signature(),
                input_example=model.get_input_example(),
            )

        self._call("log_model", _log)


def mlflow_server_reachable(tracking_uri: str, timeout: float = 3.0) -> bool:
    """
    Quick health check for http(s) tracking servers. Non-http URIs (file:, sqlite:, plain paths)
    are local stores and are treated as reachable.
    """
    if not tracking_uri.startswith(("http://", "https://")):
        return True
    health_url = tracking_uri.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError) as e:
        logger.debug(f"MLflow health check against {health_url} failed: {e}")
        return False


def create_tracker(runner_conf) -> ExperimentTracker:
    """Build the tracker for a run from RunnerConfig's tracking fields."""
    if not runner_conf.log_mlflow:
        logger.info("MLflow logging disabled in config (tracking.log_mlflow: false).")
        return NullTracker()

    uri = runner_conf.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri and not mlflow_server_reachable(uri, timeout=runner_conf.mlflow_health_check_timeout):
        message = f"MLflow server at {uri} is not reachable."
        if runner_conf.mlflow_on_unavailable == "fail":
            raise MlflowUnavailableError(
                f"{message} Set tracking.on_unavailable: warn or tracking.log_mlflow: false to run without it."
            )
        logger.warning(
            f"{message} Continuing WITHOUT experiment tracking; the trained model is still saved locally."
        )
        return NullTracker()

    return MlflowTracker(uri,
                         runner_conf.experiment_name,
                         runner_conf.run_name,
                         on_unavailable=runner_conf.mlflow_on_unavailable)
