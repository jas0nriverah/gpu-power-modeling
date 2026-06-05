from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .train import ModelArtifacts


def save_target_quality_summary(df: pd.DataFrame, out_dir: Path, target_col: str = "power_watts") -> pd.DataFrame:
    """Save summary stats and quality flags for target distribution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.dropna()
    if valid.empty:
        raise ValueError("Target column has no valid numeric values.")
    stats = {
        "count": int(valid.shape[0]),
        "missing_targets": int(y.isna().sum()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "p95": float(valid.quantile(0.95)),
        "p99": float(valid.quantile(0.99)),
        "outlier_ratio_p99_over_p50": float(valid.quantile(0.99) / max(valid.median(), 1e-9)),
    }
    stats["invalid_grouped_metric_risk"] = bool(
        stats["max"] > 2000 or stats["outlier_ratio_p99_over_p50"] > 25
    )
    summary = pd.DataFrame([stats])
    summary.to_csv(out_dir / "target_summary.csv", index=False)

    worst_abs = (
        df.assign(_abs_target=np.abs(pd.to_numeric(df[target_col], errors="coerce")))
        .sort_values("_abs_target", ascending=False)
        .head(20)
    )
    cols = [c for c in ["session_id", "timestamp", target_col, "_abs_target"] if c in worst_abs.columns]
    worst_abs[cols].to_csv(out_dir / "worst_target_rows.csv", index=False)
    return summary


def save_worst_prediction_errors(
    artifacts: ModelArtifacts,
    out_dir: Path,
    model_name: str,
) -> Path:
    """Save top 20 rows by absolute prediction error for a model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = artifacts.X_test_raw.copy()
    df["actual_power_watts"] = artifacts.y_test
    df["pred_power_watts"] = artifacts.y_pred
    df["abs_error"] = np.abs(df["actual_power_watts"] - df["pred_power_watts"])
    worst = df.sort_values("abs_error", ascending=False).head(20)
    path = out_dir / f"{model_name}_worst_prediction_rows.csv"
    cols_front = [c for c in ["session_id", "timestamp", "actual_power_watts", "pred_power_watts", "abs_error"] if c in worst.columns]
    other_cols = [c for c in worst.columns if c not in cols_front]
    worst[cols_front + other_cols].to_csv(path, index=False)
    return path
