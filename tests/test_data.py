import pandas as pd

from gpu_power_pipeline.collection import collect_nvidia_smi_samples
from gpu_power_pipeline.data import (
    available_dataset_sources,
    generate_synthetic_gpu_power_data,
    get_dataset_spec,
    load_dataset,
)
from gpu_power_pipeline.data_validation import save_dataset_summary, summarize_dataset


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


def test_dataset_specs_register_public_safe_modes():
    sources = available_dataset_sources()
    assert sources == ["bmcdata_public", "local_nvidia_smi", "mit_supercloud", "nrel_eagle", "synthetic"]
    assert get_dataset_spec("synthetic").data_kind == "synthetic"
    assert get_dataset_spec("bmcdata_public").data_kind == "public_real_trace"
    assert get_dataset_spec("mit_supercloud").data_kind == "public_real_trace"
    assert get_dataset_spec("local_nvidia_smi").data_kind == "local_measured_trace"


def test_dataset_summary_flags_synthetic_and_small_data():
    df = generate_synthetic_gpu_power_data(n_samples=80, random_state=4)
    summary, warnings = summarize_dataset(df, source="synthetic")
    assert summary.iloc[0]["source"] == "synthetic"
    assert summary.iloc[0]["data_kind"] == "synthetic"
    warning_codes = set(warnings["code"].tolist())
    assert "synthetic_data_mode" in warning_codes
    assert "small_dataset" in warning_codes
    assert "grouped_validation_not_ready" in warning_codes


def test_dataset_summary_unit_range_warning():
    df = generate_synthetic_gpu_power_data(n_samples=600, random_state=5)
    df.loc[:100, "gpu_utilization_pct"] = 150.0
    _, warnings = summarize_dataset(df, source="synthetic")
    assert "unit_range_check" in warnings["code"].tolist()


def test_save_dataset_summary_writes_report(tmp_path):
    df = generate_synthetic_gpu_power_data(n_samples=120, random_state=8)
    summary, warnings, report_path = save_dataset_summary(df, source="synthetic", out_dir=tmp_path / "data")
    assert (tmp_path / "data" / "dataset_summary.csv").exists()
    assert (tmp_path / "data" / "dataset_warnings.csv").exists()
    assert report_path.exists()
    assert not summary.empty
    assert not warnings.empty
    assert "Dataset Report" in report_path.read_text(encoding="utf-8")


def test_mit_supercloud_loader_normalizes_dcgm_columns(tmp_path):
    data_dir = tmp_path / "mit_supercloud"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "job_id": [101, 101, 102],
            "timestamp": ["2022-01-01 00:00:00", "2022-01-01 00:00:01", "2022-01-01 00:00:02"],
            "gpu": [0, 0, 1],
            "utilization gpu pct": [50.0, 75.0, 25.0],
            "utilization memory pct": [40.0, 45.0, 20.0],
            "temperature gpu": [62.0, 65.0, 55.0],
            "power draw W": [180.0, 220.0, 140.0],
            "clocks sm MHz": [1300.0, 1350.0, 1100.0],
            "clocks mem MHz": [5000.0, 5000.0, 4500.0],
        }
    ).to_csv(data_dir / "dcgm.csv", index=False)

    df = load_dataset(source="mit_supercloud", data_path=str(data_dir), max_rows=2)
    assert len(df) == 2
    assert {
        "power_watts",
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "temperature_c",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "session_id",
    }.issubset(df.columns)
    assert df["workload_type"].eq("mit_supercloud_gpu_trace").all()


def test_collect_nvidia_smi_writes_normalized_csv(tmp_path):
    def fake_runner(_command):
        return "2026/06/05 20:00:00.000, 0, NVIDIA A100, GPU-abc, 72, 55, 1410, 1215, 64, 215.5\n"

    out_path = collect_nvidia_smi_samples(
        out_path=tmp_path / "gpu_telemetry.csv",
        duration_seconds=0.01,
        interval_seconds=1.0,
        session_id="makerspace_test",
        workload_type="kernel_benchmark",
        runner=fake_runner,
    )
    df = pd.read_csv(out_path)
    assert len(df) >= 1
    assert df.iloc[0]["power_watts"] == 215.5
    assert df.iloc[0]["gpu_utilization_pct"] == 72
    assert df.iloc[0]["session_id"] == "makerspace_test_gpu_0"


def test_local_nvidia_smi_loader_reads_collected_csv(tmp_path):
    pd.DataFrame(
        {
            "timestamp": ["2026-06-05 20:00:00"],
            "gpu_id": [0],
            "gpu_name": ["NVIDIA A100"],
            "gpu_uuid": ["GPU-abc"],
            "gpu_utilization_pct": [70.0],
            "memory_utilization_pct": [60.0],
            "graphics_clock_mhz": [1410.0],
            "memory_clock_mhz": [1215.0],
            "temperature_c": [63.0],
            "power_watts": [210.0],
            "session_id": ["makerspace_run_gpu_0"],
            "workload_type": ["kernel_benchmark"],
        }
    ).to_csv(tmp_path / "gpu_telemetry.csv", index=False)

    df = load_dataset(source="local_nvidia_smi", data_path=str(tmp_path))
    assert len(df) == 1
    assert df.iloc[0]["power_watts"] == 210.0
    assert df.iloc[0]["workload_type"] == "local_nvidia_smi_gpu_trace"
