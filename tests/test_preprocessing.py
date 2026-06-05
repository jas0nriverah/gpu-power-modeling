import numpy as np
import pandas as pd

from gpu_power_pipeline.preprocessing import (
    TARGET_COL,
    build_preprocessor,
    infer_feature_columns,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gpu_utilization_pct": [10.0, 20.0, np.nan, 40.0],
            "temperature_c": [40.0, 41.0, 42.0, 43.0],
            "workload_type": ["idle", "training", "idle", "mixed"],
            TARGET_COL: [100.0, 150.0, 120.0, 200.0],
        }
    )


def test_infer_feature_columns_splits_numeric_and_categorical():
    df = _frame()
    numeric, categorical = infer_feature_columns(df)
    assert "gpu_utilization_pct" in numeric
    assert "temperature_c" in numeric
    assert "workload_type" in categorical
    assert TARGET_COL not in numeric + categorical


def test_build_preprocessor_imputes_and_encodes():
    df = _frame()
    numeric, categorical = infer_feature_columns(df)
    pre = build_preprocessor(numeric_cols=numeric, categorical_cols=categorical)
    transformed = pre.fit_transform(df[numeric + categorical])
    assert transformed.shape[0] == len(df)
    # No NaNs should remain after imputation.
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    assert np.isfinite(dense).all()
