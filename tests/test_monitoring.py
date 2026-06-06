import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.monitoring import (
    append_prediction_log,
    build_reference_profile,
    generate_drift_report,
    run_monitoring,
    validate_input_schema,
)
from gpu_power_pipeline.persistence import ModelMetadata, new_version_id, save_model_bundle
from gpu_power_pipeline.preprocessing import TARGET_COL, infer_feature_columns
from gpu_power_pipeline.train import fit_full_pipeline


def _reference_profile(n_samples: int = 120) -> tuple[pd.DataFrame, dict]:
    df = load_dataset(source="synthetic", n_samples=n_samples, random_state=12)
    numeric, categorical = infer_feature_columns(df, target_col=TARGET_COL)
    numeric = [c for c in numeric if c != "timestamp"]
    categorical = [c for c in categorical if c != "timestamp"]
    profile = build_reference_profile(
        df=df,
        feature_order=numeric + categorical,
        numeric_features=numeric,
        categorical_features=categorical,
    )
    return df, profile


def test_reference_profile_creation_has_training_stats():
    _, profile = _reference_profile()
    assert profile["row_count"] == 120
    assert "gpu_utilization_pct" in profile["numeric"]
    assert profile["numeric"]["gpu_utilization_pct"]["mean"] is not None
    assert "workload_type" in profile["categorical"]


def test_input_schema_validation_flags_missing_and_bad_numeric():
    _, profile = _reference_profile()
    frame = pd.DataFrame([{"gpu_utilization_pct": "bad", "workload_type": "training"}])
    checks = validate_input_schema(
        frame,
        feature_order=profile["feature_order"],
        numeric_features=profile["numeric_features"],
        categorical_features=profile["categorical_features"],
    )
    assert "missing_feature" in checks["check"].tolist()
    assert "non_numeric_values" in checks["check"].tolist()


def test_drift_report_generation_warns_for_shift_and_small_input(tmp_path: Path):
    df, profile = _reference_profile()
    new_inputs = df.drop(columns=[TARGET_COL]).head(6).copy()
    new_inputs["gpu_utilization_pct"] = 999.0

    drift, drift_path, report_path = run_monitoring(
        reference_profile=profile,
        input_frame=new_inputs,
        out_dir=tmp_path / "monitoring",
        input_path="new_inputs.csv",
    )
    assert drift_path.exists()
    assert report_path.exists()
    warnings = drift.loc[drift["severity"] == "warning", "check"].tolist()
    assert "small_input_sample" in warnings
    assert "outside_reference_range_share" in warnings
    assert "mean_shift_std" in warnings
    assert "Monitoring Report" in report_path.read_text(encoding="utf-8")


def test_generate_drift_report_handles_fixture_size_warning():
    df, profile = _reference_profile(n_samples=30)
    tiny_input = df.drop(columns=[TARGET_COL]).head(3)
    drift = generate_drift_report(profile, tiny_input, min_rows=20)
    row = drift.loc[drift["check"] == "small_input_sample"].iloc[0]
    assert row["severity"] == "warning"


def test_prediction_log_jsonl_format(tmp_path: Path):
    df, profile = _reference_profile(n_samples=80)
    pipeline, numeric, categorical = fit_full_pipeline(df, "random_forest", random_state=3)
    metadata = ModelMetadata(
        model_name="random_forest",
        target=TARGET_COL,
        numeric_features=numeric,
        categorical_features=categorical,
        feature_order=numeric + categorical,
        metrics={"rmse": 1.0},
        split_strategy="time",
        source="synthetic",
        version=new_version_id(),
        reference_profile=profile,
    )
    save_model_bundle(pipeline, metadata, registry_dir=tmp_path / "artifacts")
    input_frame = df[metadata.feature_order].head(2)
    predictions = pipeline.predict(input_frame)

    log_path = append_prediction_log(
        log_path=tmp_path / "prediction_log.jsonl",
        model_name=metadata.model_name,
        model_version=metadata.version,
        input_frame=input_frame,
        predictions=predictions,
    )
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert set(events[0]).issuperset(
        {"logged_at_utc", "model_name", "model_version", "row_index", "predicted_power_watts", "inputs"}
    )
    assert np.isfinite(events[0]["predicted_power_watts"])
