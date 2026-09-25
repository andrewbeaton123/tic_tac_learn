class MlflowUnavailableError(Exception):
    """Raised when MLflow tracking is required (tracking.on_unavailable: fail) but the server is unreachable."""
    pass
