"""MLflow tracking utilities for experiment management."""

import os
from contextlib import contextmanager
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(
    experiment_name: str = "trading-signals",
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Configure MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI (default: local ./mlruns)

    Returns:
        Experiment ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Default to local file store
        mlflow.set_tracking_uri("file:./mlruns")

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
    tags: Optional[dict[str, str]] = None,
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


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """
    Log training metrics.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (epoch) for the metrics
    """
    mlflow.log_metrics(metrics, step=step)


def log_model_artifact(path: str, artifact_path: Optional[str] = None) -> None:
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


def get_best_run(
    experiment_name: str,
    metric: str = "test_signal_accuracy",
    ascending: bool = False,
) -> Optional[dict]:
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
