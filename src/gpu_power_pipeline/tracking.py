from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def log_run_to_mlflow(
    out_dir: str | Path,
    metrics: pd.DataFrame,
    params: dict[str, Any],
    tracking_uri: str = "mlruns",
    experiment_name: str = "gpu-power-modeling",
) -> None:
    """Log run metadata, metrics, and artifacts to local MLflow when installed."""
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MLflow is not installed. Install optional tracking dependencies first.") from exc

    output_dir = Path(out_dir)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=output_dir.name):
        for key, value in params.items():
            if value is None:
                continue
            mlflow.log_param(key, value)
        if not metrics.empty:
            for _, row in metrics.iterrows():
                model = str(row.get("model", "unknown"))
                for metric_name in ["mae", "rmse", "r2"]:
                    if metric_name in row:
                        mlflow.log_metric(f"{model}_{metric_name}", float(row[metric_name]))
        mlflow.log_artifacts(str(output_dir))


def log_run_to_wandb(
    out_dir: str | Path,
    metrics: pd.DataFrame,
    params: dict[str, Any],
    project: str = "gpu-power-modeling",
    mode: str = "offline",
) -> None:
    """Log run metadata, metrics, and artifacts to W&B when installed."""
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Weights & Biases is not installed. Install optional tracking dependencies first.") from exc

    output_dir = Path(out_dir)
    run = wandb.init(project=project, mode=mode, config={k: v for k, v in params.items() if v is not None}, name=output_dir.name)
    try:
        metric_payload: dict[str, float] = {}
        if not metrics.empty:
            for _, row in metrics.iterrows():
                model = str(row.get("model", "unknown"))
                for metric_name in ["mae", "rmse", "r2"]:
                    if metric_name in row:
                        metric_payload[f"{model}_{metric_name}"] = float(row[metric_name])
        if metric_payload:
            wandb.log(metric_payload)
        artifact = wandb.Artifact(f"{output_dir.name}-artifacts", type="run-output")
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
    finally:
        run.finish()
