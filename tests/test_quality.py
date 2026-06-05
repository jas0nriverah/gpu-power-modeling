from pathlib import Path

import numpy as np
import pandas as pd

from gpu_power_pipeline.audit import audit_dataset
from gpu_power_pipeline.experiments import feature_set_registry
from gpu_power_pipeline.report import generate_one_page_report
from gpu_power_pipeline.train import get_train_test_indices


def test_audit_flags_name_and_corr_leakage(tmp_path: Path):
    n = 120
    target = np.linspace(10, 100, n)
    df = pd.DataFrame(
        {
            "power_watts": target,
            "power_proxy": target * 1.0,  # should trigger name + corr leakage
            "safe_feature": np.random.default_rng(0).normal(size=n),
        }
    )
    report = audit_dataset(df, out_dir=tmp_path / "audit")
    flagged = report.loc[report["feature"] == "power_proxy"].iloc[0]
    assert bool(flagged["name_leakage_risk"]) is True
    assert bool(flagged["corr_leakage_risk"]) is True


def test_grouped_split_separates_session_groups():
    n = 900
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="s"),
            "session_id": [f"s{i//150}" for i in range(n)],
            "x": np.random.default_rng(1).normal(size=n),
            "power_watts": np.random.default_rng(2).normal(loc=100, scale=5, size=n),
        }
    )
    train_idx, test_idx = get_train_test_indices(
        df=df,
        split_strategy="grouped",
        test_size=0.25,
        random_state=7,
    )
    train_sessions = set(df.iloc[train_idx]["session_id"].tolist())
    test_sessions = set(df.iloc[test_idx]["session_id"].tolist())
    assert train_sessions.isdisjoint(test_sessions)


def test_report_generation_includes_primary_metric_section(tmp_path: Path):
    out_dir = tmp_path / "run"
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)
    (out_dir / "audit").mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"model": "random_forest", "mae": 1.0, "rmse": 2.0, "r2": 0.9}]).to_csv(
        out_dir / "metrics.csv", index=False
    )
    pd.DataFrame([{"split_strategy": "random", "model": "linear_regression", "mae": 1, "rmse": 1, "r2": 1}]).to_csv(
        out_dir / "validation" / "split_comparison_best.csv", index=False
    )
    pd.DataFrame([{"model": "random_forest", "mae": 3.0, "rmse": 4.0, "r2": 0.5}]).to_csv(
        out_dir / "validation" / "grouped_blocked_cv_summary.csv", index=False
    )
    pd.DataFrame([{"n_rows": 10, "n_features_numeric": 3, "n_name_leakage_flags": 0, "n_corr_leakage_flags": 0}]).to_csv(
        out_dir / "audit" / "data_audit_summary.csv", index=False
    )

    report_path = generate_one_page_report(out_dir=out_dir, source="synthetic")
    text = report_path.read_text(encoding="utf-8")
    assert "Recommended Primary Metric (Use This)" in text
    assert "Secondary Diagnostic Metrics (Optimistic)" in text


def test_feature_registry_excludes_target_and_leakage_like_columns():
    df = pd.DataFrame(
        {
            "power_watts": [1.0, 2.0, 3.0],
            "gpu_utilization_pct": [10.0, 20.0, 30.0],
            "temp_sensor": [40.0, 42.0, 43.0],
            "board_power_like": [4.0, 5.0, 6.0],
        }
    )
    registry = feature_set_registry(df)
    for _, cols in registry.items():
        assert "power_watts" not in cols
        assert "board_power_like" not in cols
