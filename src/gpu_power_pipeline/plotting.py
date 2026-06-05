from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .train import ModelArtifacts


def plot_predicted_vs_actual(artifacts: ModelArtifacts, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.scatter(artifacts.y_test, artifacts.y_pred, s=12, alpha=0.45)
    line_min = float(min(np.min(artifacts.y_test), np.min(artifacts.y_pred)))
    line_max = float(max(np.max(artifacts.y_test), np.max(artifacts.y_pred)))
    ax.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=1.2)
    ax.set_title(f"{artifacts.model_name}: Predicted vs Actual")
    ax.set_xlabel("Actual Power (W)")
    ax.set_ylabel("Predicted Power (W)")
    ax.grid(alpha=0.25)
    path = out_dir / f"{artifacts.model_name}_pred_vs_actual.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residuals(artifacts: ModelArtifacts, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    residuals = artifacts.y_test - artifacts.y_pred
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.scatter(artifacts.y_pred, residuals, s=12, alpha=0.45)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.1)
    ax.set_title(f"{artifacts.model_name}: Residuals")
    ax.set_xlabel("Predicted Power (W)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.grid(alpha=0.25)
    path = out_dir / f"{artifacts.model_name}_residuals.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residual_distribution(artifacts: ModelArtifacts, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    residuals = artifacts.y_test - artifacts.y_pred
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.hist(residuals, bins=40, alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.1)
    ax.set_title(f"{artifacts.model_name}: Residual Distribution")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    path = out_dir / f"{artifacts.model_name}_residual_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(artifacts: ModelArtifacts, out_dir: Path, top_k: int = 15) -> Path | None:
    if artifacts.feature_importance is None or artifacts.feature_importance.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    top = artifacts.feature_importance.head(top_k).copy()
    top = top.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"{artifacts.model_name}: Feature Importance (Top {top_k})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    path = out_dir / f"{artifacts.model_name}_feature_importance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_feature_importance_table(artifacts: ModelArtifacts, out_dir: Path) -> Path | None:
    if artifacts.feature_importance is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifacts.model_name}_feature_importance.csv"
    artifacts.feature_importance.to_csv(path, index=False)
    return path


def save_permutation_importance_table(artifacts: ModelArtifacts, out_dir: Path) -> Path | None:
    if artifacts.permutation_importance is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifacts.model_name}_permutation_importance.csv"
    artifacts.permutation_importance.to_csv(path, index=False)
    return path


def save_dataset_preview(df: pd.DataFrame, out_dir: Path, n: int = 10) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dataset_preview.csv"
    df.head(n).to_csv(path, index=False)
    return path
