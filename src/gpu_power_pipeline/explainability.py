from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .train import ModelArtifacts

SHAP_SUMMARY_FILENAME_SUFFIX = "_shap_summary.csv"


def save_shap_summary(
    artifacts: ModelArtifacts,
    out_dir: str | Path,
    max_rows: int = 500,
) -> Optional[Path]:
    """Save mean absolute SHAP values for tree-style models when SHAP is installed."""
    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("SHAP is not installed. Install optional boosting dependencies first.") from exc

    model = artifacts.pipeline.named_steps["model"]
    model_name = artifacts.model_name
    supported = {"random_forest", "gradient_boosting", "xgboost", "lightgbm"}
    if model_name not in supported:
        return None

    preprocessor = artifacts.pipeline.named_steps["preprocess"]
    sample = artifacts.X_test_raw.head(max_rows).copy()
    transformed = preprocessor.transform(sample)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    if dense.size == 0:
        return None

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(dense)
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    importance = np.abs(arr).mean(axis=0)
    summary = pd.DataFrame(
        {
            "feature": artifacts.feature_names,
            "mean_abs_shap": importance,
            "model": model_name,
            "sample_rows": len(sample),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}{SHAP_SUMMARY_FILENAME_SUFFIX}"
    summary.to_csv(path, index=False)
    return path
