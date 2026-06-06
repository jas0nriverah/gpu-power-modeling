from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd

DatasetSource = Literal["synthetic", "nrel_eagle", "bmcdata_public", "mit_supercloud", "local_nvidia_smi"]


@dataclass(frozen=True)
class DatasetSpec:
    """Small adapter contract for normalized modeling data."""

    name: str
    data_kind: str
    description: str
    required_columns: tuple[str, ...]
    recommended_columns: tuple[str, ...]
    grouped_validation_column: str = "session_id"
    timestamp_column: str = "timestamp"
    target_column: str = "power_watts"


NORMALIZED_REQUIRED_COLUMNS = ("power_watts",)
NORMALIZED_RECOMMENDED_COLUMNS = (
    "timestamp",
    "session_id",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "temperature_c",
    "workload_type",
)

DATASET_SPECS: dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec(
        name="synthetic",
        data_kind="synthetic",
        description="Generated telemetry for pipeline development and CI, not real-world accuracy claims.",
        required_columns=NORMALIZED_REQUIRED_COLUMNS,
        recommended_columns=NORMALIZED_RECOMMENDED_COLUMNS,
    ),
    "bmcdata_public": DatasetSpec(
        name="bmcdata_public",
        data_kind="public_real_trace",
        description="Public BMC telemetry traces normalized into the common modeling schema.",
        required_columns=NORMALIZED_REQUIRED_COLUMNS,
        recommended_columns=NORMALIZED_RECOMMENDED_COLUMNS,
    ),
    "mit_supercloud": DatasetSpec(
        name="mit_supercloud",
        data_kind="public_real_trace",
        description="MIT Supercloud HPCA22 public NVIDIA GPU telemetry normalized from dcgm.csv or nvidia_smi.csv.",
        required_columns=NORMALIZED_REQUIRED_COLUMNS,
        recommended_columns=NORMALIZED_RECOMMENDED_COLUMNS,
    ),
    "local_nvidia_smi": DatasetSpec(
        name="local_nvidia_smi",
        data_kind="local_measured_trace",
        description="Local measured NVIDIA GPU telemetry collected with nvidia-smi on an allowed machine.",
        required_columns=NORMALIZED_REQUIRED_COLUMNS,
        recommended_columns=NORMALIZED_RECOMMENDED_COLUMNS,
    ),
    "nrel_eagle": DatasetSpec(
        name="nrel_eagle",
        data_kind="public_manual_trace",
        description="Public NREL Eagle long-format telemetry loaded from a user-provided CSV.",
        required_columns=NORMALIZED_REQUIRED_COLUMNS,
        recommended_columns=NORMALIZED_RECOMMENDED_COLUMNS,
    ),
}


def get_dataset_spec(source: str) -> DatasetSpec:
    """Return the normalized schema expectations for a dataset source."""
    if source not in DATASET_SPECS:
        raise ValueError(f"Unsupported source: {source}")
    return DATASET_SPECS[source]


def available_dataset_sources() -> list[str]:
    """List registered dataset adapters."""
    return sorted(DATASET_SPECS)


def generate_synthetic_gpu_power_data(
    n_samples: int = 12000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic telemetry and power data.

    This intentionally uses public, generic assumptions only.
    """
    rng = np.random.default_rng(random_state)
    workloads = np.array(
        ["idle", "memory_bound", "compute_bound", "mixed", "inference", "training"]
    )
    workload_type = rng.choice(
        workloads, size=n_samples, p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
    )
    time_idx = np.arange(n_samples)

    util_base = {
        "idle": 8,
        "memory_bound": 42,
        "compute_bound": 78,
        "mixed": 62,
        "inference": 54,
        "training": 84,
    }
    mem_util_base = {
        "idle": 15,
        "memory_bound": 77,
        "compute_bound": 43,
        "mixed": 63,
        "inference": 57,
        "training": 71,
    }

    gpu_util = np.array([util_base[w] for w in workload_type]) + rng.normal(0, 9, n_samples)
    mem_util = np.array([mem_util_base[w] for w in workload_type]) + rng.normal(
        0, 10, n_samples
    )
    regime = np.sin(time_idx / 900.0) * 6 + np.where((time_idx % 2000) < 400, 8, 0)
    gpu_util = gpu_util + regime * 0.7
    mem_util = mem_util + regime * 0.5
    gpu_util = np.clip(gpu_util, 0, 100)
    mem_util = np.clip(mem_util, 0, 100)

    graphics_clock_mhz = 900 + 8.5 * gpu_util + rng.normal(0, 75, n_samples)
    mem_clock_mhz = 3200 + 13 * mem_util + rng.normal(0, 120, n_samples)
    graphics_clock_mhz = np.clip(graphics_clock_mhz, 600, 2100)
    mem_clock_mhz = np.clip(mem_clock_mhz, 1800, 7500)

    ambient_c = rng.normal(23.0, 2.5, n_samples)
    temp_c = (
        ambient_c
        + 0.24 * gpu_util
        + 0.06 * mem_util
        + 0.002 * (graphics_clock_mhz - 1000)
        + rng.normal(0, 2.0, n_samples)
    )
    temp_c = np.clip(temp_c, 22, 92)

    power_watts = (
        25
        + 1.15 * gpu_util
        + 0.45 * mem_util
        + 0.018 * (graphics_clock_mhz - 900)
        + 0.008 * (mem_clock_mhz - 3200)
        + 0.75 * np.maximum(temp_c - 45, 0)
        + 0.0025 * gpu_util * mem_util
        + rng.normal(0, 7.5, n_samples)
    )
    power_watts = np.clip(power_watts, 18, 420)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n_samples, freq="s"),
            "gpu_utilization_pct": gpu_util.round(3),
            "memory_utilization_pct": mem_util.round(3),
            "graphics_clock_mhz": graphics_clock_mhz.round(3),
            "memory_clock_mhz": mem_clock_mhz.round(3),
            "temperature_c": temp_c.round(3),
            "ambient_c": ambient_c.round(3),
            "workload_type": workload_type,
            "power_watts": power_watts.round(3),
        }
    )
    # Session blocks emulate separate trace captures.
    session_span = max(100, n_samples // 6)
    session_idx = (np.arange(len(df)) // session_span).astype(int)
    df["session_id"] = [f"synthetic_session_{i}" for i in session_idx]
    return df


def _load_nrel_eagle_long_csv(path: Path) -> pd.DataFrame:
    """Load NREL Eagle telemetry if user provides downloaded CSV."""
    df_long = pd.read_csv(path)
    expected_cols = {"ts", "dv", "mt", "vl"}
    missing = expected_cols.difference(df_long.columns)
    if missing:
        raise ValueError(
            f"NREL long-format CSV must contain columns {sorted(expected_cols)}; "
            f"missing {sorted(missing)}."
        )

    df_long["ts"] = pd.to_datetime(df_long["ts"], errors="coerce")
    df_long = df_long.dropna(subset=["ts", "dv", "mt", "vl"])

    wide = (
        df_long.pivot_table(index=["ts", "dv"], columns="mt", values="vl", aggfunc="mean")
        .reset_index()
        .sort_values(["dv", "ts"])
    )
    wide.columns.name = None

    target_col = "gpu0_power_usage_report"
    if target_col not in wide.columns:
        raise ValueError(
            "NREL CSV did not include target metric 'gpu0_power_usage_report'. "
            "Use a CSV containing GPU power metrics."
        )
    wide = wide.rename(columns={target_col: "power_watts"})

    if "gpu0_mem_util" in wide.columns:
        wide = wide.rename(columns={"gpu0_mem_util": "memory_utilization_pct"})
    if "gpu0_graphics_clock_report" in wide.columns:
        wide = wide.rename(columns={"gpu0_graphics_clock_report": "graphics_clock_mhz"})
    if "gpu0_temp" in wide.columns:
        wide = wide.rename(columns={"gpu0_temp": "temperature_c"})
    if "gpu0_encoder_util" in wide.columns:
        wide = wide.rename(columns={"gpu0_encoder_util": "gpu_utilization_pct"})

    wide["timestamp"] = wide["ts"]
    wide["session_id"] = wide["dv"].astype(str)
    wide["workload_type"] = "unknown"
    return wide


def _clean_column_name(name: object) -> str:
    """Normalize source-specific telemetry headers to stable snake_case."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return cleaned.strip("_")


def _first_existing(columns: set[str], candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _resolve_gpu_telemetry_csv(path: Path, preferred_files: list[str], source_name: str) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"{source_name} path does not exist: {path}")
    for filename in preferred_files:
        candidate = path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Expected one of {preferred_files} under {path}. "
        f"Prepare {source_name} data in this directory first."
    )


def _load_gpu_telemetry_csv(
    path: Path,
    source_name: str,
    preferred_files: list[str],
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load GPU telemetry and normalize common DCGM/nvidia-smi fields."""
    csv_path = _resolve_gpu_telemetry_csv(path, preferred_files=preferred_files, source_name=source_name)
    df = pd.read_csv(csv_path, nrows=max_rows)
    if df.empty:
        raise ValueError(f"{source_name} CSV is empty: {csv_path}")

    original_columns = list(df.columns)
    renamed = {_clean_column_name(col): col for col in original_columns}
    clean = df.rename(columns={original: _clean_column_name(original) for original in original_columns})
    columns = set(clean.columns)

    target_col = _first_existing(
        columns,
        [
            "power_draw_w",
            "power_draw_watts",
            "power_watts",
            "gpu_power_w",
            "gpu_power",
            "power",
        ],
    )
    if target_col is None:
        raise ValueError(
            f"{source_name} CSV must include a GPU power column such as "
            "'power draw W', 'power_draw_w', or 'power_watts'."
        )

    out = pd.DataFrame()
    out["power_watts"] = pd.to_numeric(clean[target_col], errors="coerce")

    timestamp_col = _first_existing(columns, ["timestamp", "time", "datetime", "ts"])
    if timestamp_col:
        out["timestamp"] = pd.to_datetime(clean[timestamp_col], errors="coerce")

    job_col = _first_existing(columns, ["job_id", "jobid", "slurm_job_id", "jid"])
    node_col = _first_existing(columns, ["node", "hostname", "host", "node_id"])
    gpu_col = _first_existing(columns, ["gpu", "gpu_id", "device_id", "minor_number", "index"])
    if job_col:
        out["session_id"] = clean[job_col].astype(str)
    elif node_col:
        out["session_id"] = clean[node_col].astype(str)
    else:
        out["session_id"] = csv_path.stem
    if gpu_col:
        out["gpu_id"] = clean[gpu_col].astype(str)
        out["session_id"] = out["session_id"].astype(str) + "_gpu_" + out["gpu_id"].astype(str)

    feature_map = {
        "gpu_utilization_pct": [
            "utilization_gpu_pct",
            "gpu_utilization_pct",
            "utilization_gpu",
            "gpu_utilization",
            "sm_utilization",
            "sm_utilization_pct",
        ],
        "memory_utilization_pct": [
            "utilization_memory_pct",
            "memory_utilization_pct",
            "utilization_memory",
            "memory_utilization",
        ],
        "temperature_c": [
            "temperature_gpu",
            "temperature_gpu_c",
            "gpu_temperature",
            "gpu_temperature_c",
            "temperature_c",
        ],
        "memory_temperature_c": [
            "temperature_memory",
            "temperature_memory_c",
            "memory_temperature",
            "memory_temperature_c",
        ],
        "memory_used_mib": [
            "memory_used_mib",
            "memory_used_mb",
            "gpu_memory_used_mib",
            "gpu_memory_used_mb",
        ],
        "memory_free_mib": [
            "memory_free_mib",
            "memory_free_mb",
            "gpu_memory_free_mib",
            "gpu_memory_free_mb",
        ],
        "graphics_clock_mhz": [
            "clocks_sm_mhz",
            "sm_clock_mhz",
            "graphics_clock_mhz",
            "clock_sm_mhz",
        ],
        "memory_clock_mhz": [
            "clocks_mem_mhz",
            "memory_clock_mhz",
            "clock_memory_mhz",
            "mem_clock_mhz",
        ],
        "pcie_tx_mbps": ["pcie_tx_mbps", "pcie_tx_bytes", "pcie_tx"],
        "pcie_rx_mbps": ["pcie_rx_mbps", "pcie_rx_bytes", "pcie_rx"],
    }
    for output_col, candidates in feature_map.items():
        source_col = _first_existing(columns, candidates)
        if source_col:
            out[output_col] = pd.to_numeric(clean[source_col], errors="coerce")

    out["workload_type"] = f"{source_name}_gpu_trace"
    out["source_file"] = csv_path.name
    out = out.dropna(subset=["power_watts"])
    out = out[(out["power_watts"] > 0) & (out["power_watts"] < 2000)]
    if out.empty:
        raise ValueError(f"{source_name} CSV has no valid positive GPU power rows after normalization.")

    missing_recommended = [
        name
        for name in ["gpu_utilization_pct", "memory_utilization_pct", "temperature_c"]
        if name not in out.columns
    ]
    if missing_recommended:
        original = ", ".join(sorted(renamed.values()))
        raise ValueError(
            f"{source_name} CSV is missing key GPU telemetry columns after normalization: "
            f"{missing_recommended}. Available columns: {original}"
        )
    return out.reset_index(drop=True)


def _load_mit_supercloud_csv(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load public MIT Supercloud GPU telemetry."""
    return _load_gpu_telemetry_csv(
        path=path,
        source_name="mit_supercloud",
        preferred_files=["dcgm.csv", "nvidia_smi.csv"],
        max_rows=max_rows,
    )


def _load_local_nvidia_smi_csv(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load local measured NVIDIA GPU telemetry collected by this project."""
    return _load_gpu_telemetry_csv(
        path=path,
        source_name="local_nvidia_smi",
        preferred_files=["gpu_telemetry.csv", "nvidia_smi.csv"],
        max_rows=max_rows,
    )


def _fetch_github_repo_file_listing(owner: str, repo: str, path: str, ref: str = "master") -> list[dict]:
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    with urlopen(api) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("Unexpected GitHub API response for file listing.")
    return data


def download_bmcdata_public(
    download_dir: str | Path,
    max_files: int = 10,
    random_state: int = 42,
) -> list[Path]:
    """Download a small subset of public bmcdata traces from GitHub."""
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    listing = _fetch_github_repo_file_listing("arealuser", "bmcdata", "data", ref="master")
    files = [entry for entry in listing if entry.get("name", "").endswith(".csv")]
    if not files:
        raise ValueError("No CSV files found in public bmcdata repository.")

    rng = np.random.default_rng(random_state)
    if len(files) > max_files:
        selected_idx = rng.choice(len(files), size=max_files, replace=False)
        files = [files[i] for i in sorted(selected_idx)]

    downloaded: list[Path] = []
    for file_info in files:
        url = file_info.get("download_url")
        if not url:
            continue
        out_file = download_path / file_info["name"]
        if not out_file.exists():
            with urlopen(url) as response:
                content = response.read()
            out_file.write_bytes(content)
        downloaded.append(out_file)
    if not downloaded:
        raise ValueError("Failed to download bmcdata files.")
    return downloaded


def _load_bmcdata_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", engine="python", on_bad_lines="skip")
    to_drop = [c for c in ["result", "table", "_start", "_stop"] if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    df["source_file"] = path.stem
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if col in {"_time", "_measurement", "device_id", "name_gpu0", "source_file"}:
            continue
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    return df2


def _normalize_bmc_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known scaled telemetry units in some bmcdata sessions."""
    out = df.copy()
    for col in out.columns:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        med = float(vals.median()) if vals.notna().any() else np.nan
        if not np.isfinite(med):
            continue

        lc = col.lower()
        if any(tok in lc for tok in ["temp", "inlet", "outlet"]) and med > 500:
            out[col] = vals / 1000.0
        elif any(tok in lc for tok in ["vin", "cin"]) and med > 2000:
            out[col] = vals / 1000.0
        elif "power" in lc and med > 1_000_000:
            out[col] = vals / 1_000_000.0
        elif "power" in lc and med > 5_000:
            out[col] = vals / 1000.0
    return out


def _drop_bad_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop numeric columns that are mostly missing or constant."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    drop_cols: list[str] = []
    for col in numeric_cols:
        series = out[col]
        if series.notna().sum() < max(20, int(0.05 * len(out))):
            drop_cols.append(col)
            continue
        if series.nunique(dropna=True) <= 1:
            drop_cols.append(col)
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    return out


def _drop_corr_leakage_features(df: pd.DataFrame, target_col: str = "power_watts", threshold: float = 0.995) -> pd.DataFrame:
    """Drop non-target numeric features with near-perfect target correlation."""
    out = df.copy()
    if target_col not in out.columns:
        return out
    numeric = out.select_dtypes(include=["number"])
    target = out[target_col]
    leak_cols: list[str] = []
    for col in numeric.columns:
        if col == target_col:
            continue
        valid = pd.concat([numeric[col], target], axis=1).dropna()
        if len(valid) < 12:
            continue
        corr = valid.corr().iloc[0, 1]
        if np.isfinite(corr) and abs(float(corr)) >= threshold:
            leak_cols.append(col)
    if leak_cols:
        out = out.drop(columns=sorted(set(leak_cols)), errors="ignore")
    return out


def _build_bmc_power_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    if "_time" not in raw_df.columns:
        raise ValueError("Expected '_time' column in bmcdata files.")

    raw_df["_time"] = pd.to_datetime(raw_df["_time"], errors="coerce")
    raw_df = raw_df.dropna(subset=["_time"])
    raw_df = _coerce_numeric(raw_df)
    raw_df = _normalize_bmc_units(raw_df)

    numeric_cols = raw_df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in bmcdata files.")

    group_keys = ["_time"]
    if "source_file" in raw_df.columns:
        group_keys = ["source_file", "_time"]
    agg = raw_df.groupby(group_keys, as_index=False)[numeric_cols].mean()
    agg = agg.sort_values(group_keys).reset_index(drop=True)

    if "power_gpu0" in agg.columns:
        target = "power_gpu0"
    elif {"PSU1_Total_Power", "PSU2_Total_Power"}.issubset(set(agg.columns)):
        agg["power_gpu0"] = agg["PSU1_Total_Power"].fillna(0) + agg["PSU2_Total_Power"].fillna(0)
        target = "power_gpu0"
    elif "PSU1_Total_Power" in agg.columns:
        target = "PSU1_Total_Power"
    else:
        raise ValueError("Could not infer target power column from bmcdata.")

    rename_map = {
        "util_gpu0": "gpu_utilization_pct",
        "temp_gpu0": "temperature_c",
        "power_gpu0": "power_watts",
    }
    for old, new in rename_map.items():
        if old in agg.columns:
            agg = agg.rename(columns={old: new})
    if target != "power_gpu0":
        agg = agg.rename(columns={target: "power_watts"})

    if "memory_utilization_pct" not in agg.columns:
        mem_total = agg.get("mem_total_gpu0")
        mem_used = agg.get("mem_used_gpu0")
        if mem_total is not None and mem_used is not None:
            agg["memory_utilization_pct"] = np.where(
                mem_total > 0, (mem_used / mem_total) * 100.0, np.nan
            )

    # Keep columns with enough data density.
    min_non_null = max(40, int(0.25 * len(agg)))
    keep_cols = [c for c in agg.columns if agg[c].notna().sum() >= min_non_null]
    agg = agg[keep_cols]

    # Leakage guard: remove direct/near-direct power channels from features.
    forbidden_feature_tokens = [
        "power",
        "energy",
        "watts",
        "psu",
        "_cin",
        "_vin",
    ]
    leakage_cols = [
        c
        for c in agg.columns
        if c != "power_watts" and any(token in c.lower() for token in forbidden_feature_tokens)
    ]
    if leakage_cols:
        agg = agg.drop(columns=leakage_cols, errors="ignore")

    # Basic row filtering and cleanup.
    agg = agg.dropna(subset=["power_watts"])
    agg = agg[(agg["power_watts"] > 0) & (agg["power_watts"] < 2000)]
    agg = _drop_bad_numeric_columns(agg)
    agg = _drop_corr_leakage_features(agg, target_col="power_watts", threshold=0.995)
    agg = agg.rename(columns={"_time": "timestamp"})
    if "source_file" in agg.columns:
        agg["session_id"] = agg["source_file"].astype(str)
    else:
        agg["session_id"] = "bmc_session_0"
    agg["workload_type"] = "bmc_real_trace"
    return agg


def load_bmcdata_public(
    data_dir: str | Path,
    max_files: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load public BMC traces and build a leakage-guarded modeling frame."""
    paths = download_bmcdata_public(data_dir, max_files=max_files, random_state=random_state)
    frames = [_load_bmcdata_csv(path) for path in paths]
    combined = pd.concat(frames, axis=0, ignore_index=True)
    return _build_bmc_power_dataframe(combined)


def load_dataset(
    source: DatasetSource = "synthetic",
    data_path: Optional[str] = None,
    n_samples: int = 12000,
    random_state: int = 42,
    bmcdata_dir: Optional[str] = None,
    bmcdata_max_files: int = 10,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load dataset by source using public or synthetic inputs only."""
    if source == "synthetic":
        return generate_synthetic_gpu_power_data(
            n_samples=n_samples,
            random_state=random_state,
        )
    if source == "nrel_eagle":
        if not data_path:
            raise ValueError("data_path is required when source='nrel_eagle'.")
        return _load_nrel_eagle_long_csv(Path(data_path))
    if source == "mit_supercloud":
        if not data_path:
            raise ValueError("data_path is required when source='mit_supercloud'.")
        return _load_mit_supercloud_csv(Path(data_path), max_rows=max_rows)
    if source == "local_nvidia_smi":
        if not data_path:
            raise ValueError("data_path is required when source='local_nvidia_smi'.")
        return _load_local_nvidia_smi_csv(Path(data_path), max_rows=max_rows)
    if source == "bmcdata_public":
        if not bmcdata_dir:
            raise ValueError("bmcdata_dir is required when source='bmcdata_public'.")
        return load_bmcdata_public(
            data_dir=bmcdata_dir,
            max_files=bmcdata_max_files,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported source: {source}")
