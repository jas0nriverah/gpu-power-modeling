from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterGrid

from .config import DEFAULT_SWEEP_SPACE
from .evaluation import metrics_table
from .train import train_and_evaluate


def feature_set_registry(df: pd.DataFrame) -> Dict[str, list[str]]:
    """Return curated feature families excluding target and leakage-like columns."""
    all_cols = [c for c in df.columns if c not in {"power_watts", "timestamp"}]
    all_cols = [
        c
        for c in all_cols
        if not any(tok in c.lower() for tok in ["power", "energy", "watt"])
    ]
    gpu_cols = [c for c in all_cols if "gpu" in c or "memory_" in c or "clock" in c]
    thermal_cols = [c for c in all_cols if "temp" in c or "ambient" in c]
    system_cols = [
        c
        for c in all_cols
        if c.startswith(("cpu", "freq", "load", "PSU", "mem_", "cycles", "cache"))
    ]
    return {
        "all_features": all_cols,
        "gpu_only": sorted(set(gpu_cols + [c for c in all_cols if c == "workload_type"])),
        "gpu_plus_thermal": sorted(set(gpu_cols + thermal_cols + [c for c in all_cols if c == "workload_type"])),
        "system_only": sorted(set(system_cols + thermal_cols)),
    }


def run_feature_ablation_experiments(
    df: pd.DataFrame,
    out_dir: Path,
    include_mlp: bool = False,
    include_torch_mlp: bool = False,
    include_xgboost: bool = False,
    include_lightgbm: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "random",
) -> pd.DataFrame:
    """Run feature-family ablations and save summary tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = feature_set_registry(df)

    rows = []
    for feature_set_name, features in feature_sets.items():
        results = train_and_evaluate(
            df=df,
            test_size=test_size,
            random_state=random_state,
            include_mlp=include_mlp,
            include_torch_mlp=include_torch_mlp,
            include_xgboost=include_xgboost,
            include_lightgbm=include_lightgbm,
            selected_features=features,
            split_strategy=split_strategy,
        )
        metric_df = metrics_table(results)
        for _, row in metric_df.iterrows():
            model_name = row["model"]
            diag = results[model_name].residual_diagnostics
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "r2": row["r2"],
                    "residual_bias": diag["residual_bias"],
                    "residual_std": diag["residual_std"],
                    "residual_p95_abs": diag["residual_p95_abs"],
                    "feature_count": len(features),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["feature_set", "rmse"]).reset_index(drop=True)
    summary.to_csv(out_dir / "feature_ablation_summary.csv", index=False)

    best = summary.sort_values("rmse").groupby("feature_set", as_index=False).first()
    best.to_csv(out_dir / "feature_ablation_best_by_set.csv", index=False)
    return summary


def run_hyperparameter_sweep(
    df: pd.DataFrame,
    out_dir: Path,
    random_state: int = 42,
    test_size: float = 0.2,
    split_strategy: str = "random",
    search_space: Optional[Dict[str, Dict[str, List[Any]]]] = None,
) -> pd.DataFrame:
    """Run a lightweight sweep over tree-model hyperparameters.

    The grid defaults to :data:`config.DEFAULT_SWEEP_SPACE` but can be
    overridden (e.g. from a YAML experiment config).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    search_space = search_space or {k: dict(v) for k, v in DEFAULT_SWEEP_SPACE.items()}

    rows = []
    for model_name, grid in search_space.items():
        for combo in ParameterGrid(grid):
            results = train_and_evaluate(
                df=df,
                test_size=test_size,
                random_state=random_state,
                split_strategy=split_strategy,
                model_params={model_name: combo},
                model_names=[model_name],
            )
            metrics = metrics_table(results).iloc[0].to_dict()
            diag = results[model_name].residual_diagnostics
            rows.append(
                {
                    "model": model_name,
                    "params": str(combo),
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "r2": float(metrics["r2"]),
                    "residual_p95_abs": diag["residual_p95_abs"],
                }
            )
    summary = pd.DataFrame(rows).sort_values(["model", "rmse"]).reset_index(drop=True)
    summary.to_csv(out_dir / "hyperparameter_sweep.csv", index=False)
    best = summary.groupby("model", as_index=False).first()
    best.to_csv(out_dir / "hyperparameter_sweep_best.csv", index=False)
    return summary


def run_split_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    random_state: int = 42,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Compare random, time, and grouped-session split behavior."""
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = ["random", "time", "grouped"]
    rows = []
    for strategy in strategies:
        if strategy == "grouped" and "session_id" not in df.columns:
            continue
        results = train_and_evaluate(
            df=df,
            test_size=test_size,
            random_state=random_state,
            split_strategy=strategy,
        )
        table = metrics_table(results)
        for _, r in table.iterrows():
            rows.append(
                {
                    "split_strategy": strategy,
                    "model": r["model"],
                    "mae": float(r["mae"]),
                    "rmse": float(r["rmse"]),
                    "r2": float(r["r2"]),
                }
            )
    out = pd.DataFrame(rows).sort_values(["split_strategy", "rmse"]).reset_index(drop=True)
    out.to_csv(out_dir / "split_comparison.csv", index=False)
    best = out.groupby("split_strategy", as_index=False).first()
    best.to_csv(out_dir / "split_comparison_best.csv", index=False)
    return out


def run_grouped_blocked_cv(
    df: pd.DataFrame,
    out_dir: Path,
    random_state: int = 42,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Run small grouped blocked CV using session_id as group boundary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if "session_id" not in df.columns:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "grouped_blocked_cv.csv", index=False)
        return empty

    groups = df["session_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "grouped_blocked_cv.csv", index=False)
        return empty

    fold_count = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=fold_count)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        results = train_and_evaluate(
            df=df,
            split_strategy="grouped",
            random_state=random_state,
            model_names=["linear_regression", "random_forest"],
            train_indices=np.asarray(train_idx),
            test_indices=np.asarray(test_idx),
        )
        table = metrics_table(results)
        for _, r in table.iterrows():
            rows.append(
                {
                    "fold": fold_idx,
                    "model": r["model"],
                    "mae": float(r["mae"]),
                    "rmse": float(r["rmse"]),
                    "r2": float(r["r2"]),
                }
            )
    detail = pd.DataFrame(rows).sort_values(["fold", "rmse"]).reset_index(drop=True)
    detail.to_csv(out_dir / "grouped_blocked_cv.csv", index=False)
    summary = (
        detail.groupby("model", as_index=False)[["mae", "rmse", "r2"]]
        .mean()
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    summary.to_csv(out_dir / "grouped_blocked_cv_summary.csv", index=False)
    return detail
