from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .train import ModelArtifacts


def metrics_table(results: Dict[str, ModelArtifacts]) -> pd.DataFrame:
    rows = []
    for model_name, artifacts in results.items():
        row = {"model": model_name}
        row.update(artifacts.metrics)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    return df


def save_metrics(results: Dict[str, ModelArtifacts], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = metrics_table(results)
    table.to_csv(out_dir / "metrics.csv", index=False)
    return table
