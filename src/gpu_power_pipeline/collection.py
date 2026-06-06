from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

NVIDIA_SMI_QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
]

OUTPUT_COLUMNS = [
    "timestamp",
    "gpu_id",
    "gpu_name",
    "gpu_uuid",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "temperature_c",
    "power_watts",
    "session_id",
    "workload_type",
]

Runner = Callable[[list[str]], str]


def _default_runner(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def _parse_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "[Not Supported]", "N/A", "Not Supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _query_nvidia_smi(runner: Runner = _default_runner) -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_SMI_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    output = runner(command)
    rows: list[dict[str, object]] = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        values = [part.strip() for part in raw_line.split(",")]
        if len(values) != len(NVIDIA_SMI_QUERY_FIELDS):
            raise ValueError(f"Unexpected nvidia-smi row with {len(values)} fields: {raw_line}")
        rows.append(
            {
                "timestamp": values[0],
                "gpu_id": values[1],
                "gpu_name": values[2],
                "gpu_uuid": values[3],
                "gpu_utilization_pct": _parse_float(values[4]),
                "memory_utilization_pct": _parse_float(values[5]),
                "graphics_clock_mhz": _parse_float(values[6]),
                "memory_clock_mhz": _parse_float(values[7]),
                "temperature_c": _parse_float(values[8]),
                "power_watts": _parse_float(values[9]),
            }
        )
    return rows


def collect_nvidia_smi_samples(
    out_path: str | Path,
    duration_seconds: float,
    interval_seconds: float = 1.0,
    session_id: Optional[str] = None,
    workload_type: str = "local_nvidia_smi_trace",
    runner: Runner = _default_runner,
) -> Path:
    """Collect local NVIDIA GPU telemetry into this project's normalized CSV schema."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = session_id or datetime.now(timezone.utc).strftime("local_gpu_%Y%m%dT%H%M%SZ")
    deadline = time.monotonic() + duration_seconds
    wrote_any = False

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        while True:
            rows = _query_nvidia_smi(runner=runner)
            for row in rows:
                row["session_id"] = f"{session}_gpu_{row.get('gpu_id', 'unknown')}"
                row["workload_type"] = workload_type
                writer.writerow(row)
                wrote_any = True
            if time.monotonic() >= deadline:
                break
            time.sleep(interval_seconds)

    if not wrote_any:
        raise RuntimeError("nvidia-smi returned no GPU rows.")
    return output_path
