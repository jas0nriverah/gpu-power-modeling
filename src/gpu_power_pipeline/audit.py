from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def audit_dataset(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Flag potential leakage via feature names and target correlation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    if "power_watts" not in df.columns:
        raise ValueError("Dataset audit requires 'power_watts'.")

    numeric = df.select_dtypes(include=["number"])
    target = df["power_watts"]

    for col in numeric.columns:
        if col == "power_watts":
            continue
        valid = pd.concat([numeric[col], target], axis=1).dropna()
        corr = float(valid.corr().iloc[0, 1]) if len(valid) > 8 else np.nan
        suspicious_name = any(tok in col.lower() for tok in ["power", "energy", "watt"])
        suspicious_corr = np.isfinite(corr) and abs(corr) > 0.995
        rows.append(
            {
                "feature": col,
                "abs_corr_with_target": abs(corr) if np.isfinite(corr) else np.nan,
                "name_leakage_risk": suspicious_name,
                "corr_leakage_risk": bool(suspicious_corr),
            }
        )

    report = pd.DataFrame(rows).sort_values("abs_corr_with_target", ascending=False)
    report.to_csv(out_dir / "data_audit.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_features_numeric": len(numeric.columns) - 1,
                "n_name_leakage_flags": int(report["name_leakage_risk"].sum()),
                "n_corr_leakage_flags": int(report["corr_leakage_risk"].sum()),
            }
        ]
    )
    summary.to_csv(out_dir / "data_audit_summary.csv", index=False)
    return report
