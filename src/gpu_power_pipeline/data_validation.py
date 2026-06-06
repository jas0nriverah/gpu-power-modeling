from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import DatasetSpec, get_dataset_spec

DATASET_SUMMARY_FILENAME = "dataset_summary.csv"
DATASET_WARNINGS_FILENAME = "dataset_warnings.csv"
DATASET_REPORT_FILENAME = "DATASET_REPORT.md"

MIN_ROWS_FOR_MODELING = 500
MIN_GROUPS_FOR_GROUPED_VALIDATION = 3
MIN_ROWS_PER_GROUP = 20

EXPECTED_RANGES = {
    "gpu_utilization_pct": (0.0, 100.0),
    "memory_utilization_pct": (0.0, 100.0),
    "temperature_c": (-20.0, 120.0),
    "ambient_c": (-20.0, 60.0),
    "graphics_clock_mhz": (0.0, 10000.0),
    "memory_clock_mhz": (0.0, 12000.0),
    "power_watts": (0.0, 2000.0),
}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _warning(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def summarize_dataset(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return dataset summary and suitability warnings for a normalized dataset."""
    spec = get_dataset_spec(source)
    summary = _build_summary(df, spec)
    warnings = _build_warnings(df, spec, summary.iloc[0].to_dict())
    return summary, pd.DataFrame(warnings, columns=["code", "severity", "message"])


def _build_summary(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    session_col = spec.grouped_validation_column
    target_col = spec.target_column
    timestamp_col = spec.timestamp_column

    if session_col in df.columns:
        group_sizes = df[session_col].astype(str).value_counts()
        unique_sessions = int(group_sizes.shape[0])
        min_rows_per_session = int(group_sizes.min()) if not group_sizes.empty else 0
        median_rows_per_session = _safe_float(group_sizes.median())
        max_rows_per_session = int(group_sizes.max()) if not group_sizes.empty else 0
    else:
        unique_sessions = 0
        min_rows_per_session = 0
        median_rows_per_session = None
        max_rows_per_session = 0

    target_valid = 0
    target_min = None
    target_max = None
    target_missing_rate = None
    if target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce")
        valid = target.dropna()
        target_valid = int(valid.shape[0])
        target_missing_rate = _safe_float(target.isna().mean())
        target_min = _safe_float(valid.min())
        target_max = _safe_float(valid.max())

    recommended_present = [col for col in spec.recommended_columns if col in df.columns]
    recommended_missing = [col for col in spec.recommended_columns if col not in df.columns]
    grouped_ready = unique_sessions >= MIN_GROUPS_FOR_GROUPED_VALIDATION and min_rows_per_session >= MIN_ROWS_PER_GROUP

    return pd.DataFrame(
        [
            {
                "source": spec.name,
                "data_kind": spec.data_kind,
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "n_numeric_columns": int(len(numeric_cols)),
                "n_categorical_columns": int(len(categorical_cols)),
                "has_target": bool(target_col in df.columns),
                "target_valid_rows": target_valid,
                "target_missing_rate": target_missing_rate,
                "target_min": target_min,
                "target_max": target_max,
                "has_timestamp": bool(timestamp_col in df.columns),
                "has_session_id": bool(session_col in df.columns),
                "unique_sessions": unique_sessions,
                "min_rows_per_session": min_rows_per_session,
                "median_rows_per_session": median_rows_per_session,
                "max_rows_per_session": max_rows_per_session,
                "grouped_validation_ready": bool(grouped_ready),
                "recommended_columns_present": int(len(recommended_present)),
                "recommended_columns_missing": ",".join(recommended_missing),
            }
        ]
    )


def _build_warnings(df: pd.DataFrame, spec: DatasetSpec, summary: dict[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    columns = set(df.columns)

    missing_required = [col for col in spec.required_columns if col not in columns]
    if missing_required:
        warnings.append(
            _warning(
                "missing_required_columns",
                "error",
                f"Missing required normalized column(s): {', '.join(missing_required)}.",
            )
        )

    missing_recommended = [col for col in spec.recommended_columns if col not in columns]
    for col in missing_recommended:
        warnings.append(
            _warning(
                "missing_recommended_column",
                "warning",
                f"Recommended column '{col}' is missing; modeling can run but diagnostics may be weaker.",
            )
        )

    if int(summary["n_rows"]) < MIN_ROWS_FOR_MODELING:
        warnings.append(
            _warning(
                "small_dataset",
                "warning",
                f"Dataset has {summary['n_rows']} rows; results are unstable below {MIN_ROWS_FOR_MODELING} rows.",
            )
        )

    if spec.data_kind == "synthetic":
        warnings.append(
            _warning(
                "synthetic_data_mode",
                "warning",
                "Synthetic data is useful for pipeline tests and demos, not for proving real-world accuracy.",
            )
        )

    if not bool(summary["grouped_validation_ready"]):
        warnings.append(
            _warning(
                "grouped_validation_not_ready",
                "warning",
                "Grouped validation needs at least "
                f"{MIN_GROUPS_FOR_GROUPED_VALIDATION} sessions with about {MIN_ROWS_PER_GROUP}+ rows each.",
            )
        )

    warnings.extend(_unit_range_warnings(df))
    return warnings


def _unit_range_warnings(df: pd.DataFrame, max_out_of_range_share: float = 0.05) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for col, (lo, hi) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            warnings.append(
                _warning("no_valid_unit_values", "warning", f"Column '{col}' has no valid numeric values.")
            )
            continue
        outside_share = float(((valid < lo) | (valid > hi)).mean())
        if outside_share > max_out_of_range_share:
            warnings.append(
                _warning(
                    "unit_range_check",
                    "warning",
                    f"Column '{col}' has {outside_share:.1%} values outside expected range [{lo}, {hi}].",
                )
            )
    return warnings


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    sub = df.fillna("").copy()
    cols = sub.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in sub.iterrows()]
    return "\n".join([header, sep] + body)


def write_dataset_report(
    summary: pd.DataFrame,
    warnings: pd.DataFrame,
    out_dir: str | Path,
) -> Path:
    """Write a concise markdown report for dataset suitability."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    blocking = warnings[warnings["severity"] == "error"] if not warnings.empty else pd.DataFrame()
    warning_rows = warnings[warnings["severity"] == "warning"] if not warnings.empty else pd.DataFrame()
    status = "error" if not blocking.empty else "warning" if not warning_rows.empty else "ok"
    report = f"""# Dataset Report

## Summary
- Status: `{status}`
- Source: `{summary.iloc[0].get("source", "unknown") if not summary.empty else "unknown"}`
- Data kind: `{summary.iloc[0].get("data_kind", "unknown") if not summary.empty else "unknown"}`
- Rows: `{summary.iloc[0].get("n_rows", "unknown") if not summary.empty else "unknown"}`
- Grouped validation ready: `{summary.iloc[0].get("grouped_validation_ready", "unknown") if not summary.empty else "unknown"}`

## Suitability Checks
{_to_markdown_table(warnings)}

## Dataset Summary
{_to_markdown_table(summary)}
"""
    path = out_path / DATASET_REPORT_FILENAME
    path.write_text(report, encoding="utf-8")
    return path


def save_dataset_summary(df: pd.DataFrame, source: str, out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Save dataset summary CSVs and a markdown suitability report."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary, warnings = summarize_dataset(df, source=source)
    summary.to_csv(out_path / DATASET_SUMMARY_FILENAME, index=False)
    warnings.to_csv(out_path / DATASET_WARNINGS_FILENAME, index=False)
    report_path = write_dataset_report(summary, warnings, out_dir=out_path)
    return summary, warnings, report_path
