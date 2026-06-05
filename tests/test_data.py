import pandas as pd

from gpu_power_pipeline.data import generate_synthetic_gpu_power_data


def test_synthetic_data_has_expected_columns():
    df = generate_synthetic_gpu_power_data(n_samples=250, random_state=7)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 250
    expected = {
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "temperature_c",
        "ambient_c",
        "workload_type",
        "power_watts",
    }
    assert expected.issubset(set(df.columns))
    assert df["power_watts"].min() >= 0
    assert "session_id" in df.columns
