"""MLflow tracking utilities for experiment management."""

import os
from contextlib import contextmanager
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(
    experiment_name: str = "trading-signals",
    tracking_uri: str | None = None,
) -> str:
    """
    Configure MLflow tracking.

    URI resolution order:
      1. explicit ``tracking_uri`` argument
      2. ``MLFLOW_TRACKING_URI`` environment variable (set by docker-compose)
      3. local file store at ``./mlruns``

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI

    Returns:
        Experiment ID
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "file:./mlruns"
    mlflow.set_tracking_uri(uri)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


@contextmanager
def training_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    nested: bool = False,
):
    """
    Context manager for MLflow training runs.

    Args:
        run_name: Name for this training run
        tags: Optional tags to attach to the run
        nested: Whether this is a nested run (for walk-forward windows)

    Yields:
        Active MLflow run
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_hyperparameters(params: dict[str, Any]) -> None:
    """
    Log training hyperparameters.

    Args:
        params: Dictionary of hyperparameter names and values
    """
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """
    Log training metrics.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (epoch) for the metrics
    """
    mlflow.log_metrics(metrics, step=step)


def log_model_artifact(path: str, artifact_path: str | None = None) -> None:
    """
    Log model weights or config as an artifact.

    Args:
        path: Local path to the file to log
        artifact_path: Optional subdirectory in the artifact store
    """
    if os.path.exists(path):
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_training_history(history: dict[str, list[float]]) -> None:
    """
    Log training history metrics per epoch.

    Args:
        history: Keras history.history dictionary
    """
    num_epochs = len(history.get("loss", []))
    for epoch in range(num_epochs):
        epoch_metrics = {}
        for metric_name, values in history.items():
            if epoch < len(values):
                epoch_metrics[metric_name] = values[epoch]
        if epoch_metrics:
            log_metrics(epoch_metrics, step=epoch)


def get_recent_runs(
    experiment_name: str = "trading-signals",
    run_type: str | None = None,
    ticker: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """
    Fetch recent runs from an experiment, optionally filtered by type and ticker.

    Args:
        experiment_name: MLflow experiment to query
        run_type: Filter by ``run_type`` tag (e.g. ``"backtest"``, ``"standard"``)
        ticker: Filter by ``ticker`` tag (exact match)
        max_results: Maximum number of runs to return (most recent first)

    Returns:
        List of run dicts with keys: run_id, run_name, start_time, metrics, params, tags
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    filter_parts = []
    if run_type:
        filter_parts.append(f"tags.run_type = '{run_type}'")
    if ticker:
        filter_parts.append(f"tags.ticker = '{ticker}'")
    filter_string = " and ".join(filter_parts)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=max_results,
    )

    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "start_time": r.info.start_time,
            "metrics": r.data.metrics,
            "params": r.data.params,
            "tags": r.data.tags,
        }
        for r in runs
    ]


def get_best_run(
    experiment_name: str,
    metric: str = "test_signal_accuracy",
    ascending: bool = False,
) -> dict | None:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        ascending: If True, lower is better

    Returns:
        Dictionary with run info or None if no runs found
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        return None

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "artifact_uri": run.info.artifact_uri,
    }
