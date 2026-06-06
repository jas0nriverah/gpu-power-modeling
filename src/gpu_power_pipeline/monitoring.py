from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

REFERENCE_PROFILE_FILENAME = "reference_profile.json"
DRIFT_REPORT_FILENAME = "drift_report.csv"
MONITORING_REPORT_FILENAME = "monitoring_report.md"


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _json_default(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def build_reference_profile(
    df: pd.DataFrame,
    feature_order: Iterable[str],
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    min_rows: int = 20,
) -> dict[str, Any]:
    """Summarize training feature distributions for later input drift checks."""
    features = list(feature_order)
    numeric = list(numeric_features)
    categorical = list(categorical_features)
    warnings: list[str] = []
    if len(df) < min_rows:
        warnings.append(f"Reference data has {len(df)} rows; drift checks are weak below {min_rows} rows.")

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        warnings.append(f"Reference data is missing expected features: {', '.join(missing_features)}.")

    numeric_stats: dict[str, dict[str, Any]] = {}
    for col in numeric:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        numeric_stats[col] = {
            "count": int(valid.shape[0]),
            "missing_rate": _safe_float(series.isna().mean()),
            "mean": _safe_float(valid.mean()),
            "std": _safe_float(valid.std(ddof=0)),
            "min": _safe_float(valid.min()),
            "p05": _safe_float(valid.quantile(0.05)),
            "p50": _safe_float(valid.quantile(0.50)),
            "p95": _safe_float(valid.quantile(0.95)),
            "max": _safe_float(valid.max()),
        }

    categorical_stats: dict[str, dict[str, Any]] = {}
    for col in categorical:
        if col not in df.columns:
            continue
        series = df[col].astype("object")
        non_missing = series.dropna().astype(str)
        counts = non_missing.value_counts()
        categories = counts.head(100).index.tolist()
        categorical_stats[col] = {
            "count": int(non_missing.shape[0]),
            "missing_rate": _safe_float(series.isna().mean()),
            "unique_count": int(counts.shape[0]),
            "categories": categories,
            "categories_truncated": bool(counts.shape[0] > len(categories)),
            "top_values": {str(k): int(v) for k, v in counts.head(20).items()},
        }

    return {
        "profile_version": 1,
        "row_count": int(len(df)),
        "feature_order": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "numeric": numeric_stats,
        "categorical": categorical_stats,
        "warnings": warnings,
    }


def save_reference_profile(profile: dict[str, Any], out_dir: str | Path) -> Path:
    """Write a reference profile to ``reference_profile.json``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / REFERENCE_PROFILE_FILENAME
    path.write_text(json.dumps(profile, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return path


def _profile_from_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data.get("reference_profile")
    if not profile:
        raise ValueError(f"No reference_profile found in {path}")
    return profile


def load_reference_profile(reference: str | Path) -> dict[str, Any]:
    """Load reference stats from a run directory, monitoring directory, or model bundle."""
    ref_path = Path(reference)
    if ref_path.is_file():
        if ref_path.name == REFERENCE_PROFILE_FILENAME:
            return json.loads(ref_path.read_text(encoding="utf-8"))
        if ref_path.name == "metadata.json":
            return _profile_from_metadata(ref_path)
    candidates = [
        ref_path / REFERENCE_PROFILE_FILENAME,
        ref_path / "monitoring" / REFERENCE_PROFILE_FILENAME,
        ref_path / "metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.name == "metadata.json":
                return _profile_from_metadata(candidate)
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No reference profile found under {ref_path}. Expected reference_profile.json "
        "or metadata.json with reference_profile."
    )


def validate_input_schema(
    frame: pd.DataFrame,
    feature_order: Iterable[str],
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> pd.DataFrame:
    """Return schema validation rows for prediction or monitoring inputs."""
    if frame.empty:
        raise ValueError("No input rows provided.")
    expected = list(feature_order)
    numeric = set(numeric_features)
    categorical = set(categorical_features)
    rows: list[dict[str, Any]] = []

    for col in expected:
        if col not in frame.columns:
            rows.append(
                {
                    "feature": col,
                    "feature_type": "numeric" if col in numeric else "categorical" if col in categorical else "unknown",
                    "check": "missing_feature",
                    "severity": "warning",
                    "message": f"Input is missing expected feature '{col}'. Prediction will impute or ignore it.",
                }
            )
            continue
        if col in numeric:
            raw = frame[col]
            coerced = pd.to_numeric(raw, errors="coerce")
            invalid = raw.notna() & coerced.isna()
            if invalid.any():
                rows.append(
                    {
                        "feature": col,
                        "feature_type": "numeric",
                        "check": "non_numeric_values",
                        "severity": "warning",
                        "message": f"Feature '{col}' has {int(invalid.sum())} non-numeric value(s).",
                    }
                )

    extras = [col for col in frame.columns if col not in expected]
    for col in extras:
        rows.append(
            {
                "feature": col,
                "feature_type": "extra",
                "check": "extra_feature",
                "severity": "info",
                "message": f"Input includes extra feature '{col}' that the model does not use.",
            }
        )
    return pd.DataFrame(rows)


def _severity(warn: bool) -> str:
    return "warning" if warn else "ok"


def generate_drift_report(
    reference_profile: dict[str, Any],
    input_frame: pd.DataFrame,
    min_rows: int = 20,
    mean_shift_threshold: float = 2.0,
    range_share_threshold: float = 0.20,
    missing_delta_threshold: float = 0.20,
    unseen_share_threshold: float = 0.20,
) -> pd.DataFrame:
    """Compare new prediction inputs with training/reference feature distributions."""
    if input_frame.empty:
        raise ValueError("No input rows provided.")
    rows: list[dict[str, Any]] = []
    n_rows = len(input_frame)
    if n_rows < min_rows:
        rows.append(
            {
                "feature": "__dataset__",
                "feature_type": "dataset",
                "check": "small_input_sample",
                "reference_value": reference_profile.get("row_count"),
                "input_value": n_rows,
                "threshold": min_rows,
                "severity": "warning",
                "message": f"Input has {n_rows} rows; drift estimates are unstable below {min_rows} rows.",
            }
        )

    schema = validate_input_schema(
        input_frame,
        feature_order=reference_profile.get("feature_order", []),
        numeric_features=reference_profile.get("numeric_features", []),
        categorical_features=reference_profile.get("categorical_features", []),
    )
    rows.extend(schema.to_dict(orient="records"))

    for feature, ref in reference_profile.get("numeric", {}).items():
        if feature not in input_frame.columns:
            continue
        series = pd.to_numeric(input_frame[feature], errors="coerce")
        valid = series.dropna()
        input_missing = _safe_float(series.isna().mean()) or 0.0
        ref_missing = float(ref.get("missing_rate") or 0.0)
        missing_delta = input_missing - ref_missing
        warn_missing = missing_delta > missing_delta_threshold
        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "check": "missing_rate_delta",
                "reference_value": ref_missing,
                "input_value": input_missing,
                "threshold": missing_delta_threshold,
                "severity": _severity(warn_missing),
                "message": (
                    f"Missing rate increased by {missing_delta:.3f}."
                    if warn_missing
                    else "Missing rate is close to reference."
                ),
            }
        )
        if valid.empty:
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "check": "no_valid_numeric_values",
                    "reference_value": ref.get("count"),
                    "input_value": 0,
                    "threshold": 1,
                    "severity": "warning",
                    "message": f"Feature '{feature}' has no valid numeric values in input.",
                }
            )
            continue

        ref_mean = ref.get("mean")
        ref_std = ref.get("std")
        input_mean = _safe_float(valid.mean())
        if ref_mean is not None and input_mean is not None:
            denom = max(float(ref_std or 0.0), 1e-9)
            shift = abs(input_mean - float(ref_mean)) / denom
            warn_shift = shift > mean_shift_threshold
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "check": "mean_shift_std",
                    "reference_value": ref_mean,
                    "input_value": input_mean,
                    "threshold": mean_shift_threshold,
                    "severity": _severity(warn_shift),
                    "message": (
                        f"Input mean shifted by {shift:.2f} reference std devs."
                        if warn_shift
                        else "Input mean is close to reference."
                    ),
                }
            )

        ref_min = ref.get("min")
        ref_max = ref.get("max")
        if ref_min is not None and ref_max is not None:
            outside = (valid < float(ref_min)) | (valid > float(ref_max))
            outside_share = _safe_float(outside.mean()) or 0.0
            warn_range = outside_share > range_share_threshold
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "check": "outside_reference_range_share",
                    "reference_value": f"[{ref_min}, {ref_max}]",
                    "input_value": outside_share,
                    "threshold": range_share_threshold,
                    "severity": _severity(warn_range),
                    "message": (
                        f"{outside_share:.1%} of values are outside the reference min/max range."
                        if warn_range
                        else "Values are mostly within the reference min/max range."
                    ),
                }
            )

    for feature, ref in reference_profile.get("categorical", {}).items():
        if feature not in input_frame.columns:
            continue
        series = input_frame[feature].astype("object")
        valid = series.dropna().astype(str)
        input_missing = _safe_float(series.isna().mean()) or 0.0
        ref_missing = float(ref.get("missing_rate") or 0.0)
        known = set(ref.get("categories", []))
        if valid.empty:
            unseen_share = 0.0
        elif known and not ref.get("categories_truncated", False):
            unseen_share = float((~valid.isin(known)).mean())
        else:
            unseen_share = 0.0
        warn_unseen = unseen_share > unseen_share_threshold
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "check": "missing_rate_delta",
                "reference_value": ref_missing,
                "input_value": input_missing,
                "threshold": missing_delta_threshold,
                "severity": _severity(input_missing - ref_missing > missing_delta_threshold),
                "message": "Categorical missing rate compared with reference.",
            }
        )
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "check": "unseen_category_share",
                "reference_value": len(known),
                "input_value": unseen_share,
                "threshold": unseen_share_threshold,
                "severity": _severity(warn_unseen),
                "message": (
                    f"{unseen_share:.1%} of values were not observed in the reference profile."
                    if warn_unseen
                    else "Categories are covered by the reference profile or too high-cardinality to flag."
                ),
            }
        )

    return pd.DataFrame(rows)


def _to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    sub = df.head(max_rows).fillna("").copy()
    cols = sub.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in sub.iterrows()]
    return "\n".join([header, sep] + body)


def write_monitoring_report(
    drift_report: pd.DataFrame,
    out_dir: str | Path,
    reference_profile: dict[str, Any],
    input_path: Optional[str | Path] = None,
) -> Path:
    """Write a concise markdown monitoring report."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    warnings = drift_report[drift_report["severity"] == "warning"] if not drift_report.empty else pd.DataFrame()
    status = "warning" if not warnings.empty else "ok"
    warning_text = "\n".join(f"- {row['feature']}: {row['message']}" for _, row in warnings.iterrows())
    if not warning_text:
        warning_text = "- No warnings."
    report = f"""# Monitoring Report

## Summary
- Status: `{status}`
- Reference rows: `{reference_profile.get("row_count", "unknown")}`
- Input file: `{input_path or "not recorded"}`
- Warning count: `{len(warnings)}`

## Warnings
{warning_text}

## Drift Checks
{_to_markdown_table(drift_report)}
"""
    path = out_path / MONITORING_REPORT_FILENAME
    path.write_text(report, encoding="utf-8")
    return path


def run_monitoring(
    reference_profile: dict[str, Any],
    input_frame: pd.DataFrame,
    out_dir: str | Path,
    input_path: Optional[str | Path] = None,
    min_rows: int = 20,
) -> tuple[pd.DataFrame, Path, Path]:
    """Generate CSV and markdown monitoring reports for new prediction inputs."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    drift = generate_drift_report(reference_profile, input_frame, min_rows=min_rows)
    drift_path = out_path / DRIFT_REPORT_FILENAME
    drift.to_csv(drift_path, index=False)
    report_path = write_monitoring_report(
        drift_report=drift,
        out_dir=out_path,
        reference_profile=reference_profile,
        input_path=input_path,
    )
    return drift, drift_path, report_path


def append_prediction_log(
    log_path: str | Path,
    model_name: str,
    model_version: str,
    input_frame: pd.DataFrame,
    predictions: Iterable[float],
) -> Path:
    """Append prediction events as JSON lines for batch or API inference."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_list = [float(p) for p in predictions]
    if len(predictions_list) != len(input_frame):
        raise ValueError("Prediction count does not match input row count.")
    logged_at = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        for row_idx, (_, row) in enumerate(input_frame.iterrows()):
            inputs = {
                str(key): (None if pd.isna(value) else value)
                for key, value in row.to_dict().items()
            }
            event = {
                "logged_at_utc": logged_at,
                "model_name": model_name,
                "model_version": model_version,
                "row_index": row_idx,
                "predicted_power_watts": predictions_list[row_idx],
                "inputs": inputs,
            }
            handle.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")
    return path
