"""Inference utilities: load a persisted model and predict on new telemetry.

Wraps a saved :class:`~sklearn.pipeline.Pipeline` so callers can pass plain
dicts/records and get power predictions back, with input validation and
column alignment handled consistently between the CLI and the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .persistence import (
    ModelMetadata,
    load_latest_bundle,
    load_model_bundle,
    resolve_model_dir,
)

Record = Mapping[str, Any]
Records = Union[Record, Sequence[Record], pd.DataFrame]


class PowerModel:
    """A loaded model bundle ready for prediction."""

    def __init__(self, pipeline: Pipeline, metadata: ModelMetadata) -> None:
        self.pipeline = pipeline
        self.metadata = metadata

    @classmethod
    def from_dir(cls, version_dir: str | Path) -> "PowerModel":
        pipeline, metadata = load_model_bundle(version_dir)
        return cls(pipeline, metadata)

    @classmethod
    def from_registry(
        cls,
        registry_dir: str | Path,
        model_name: str,
        version: str = "latest",
    ) -> "PowerModel":
        if version == "latest":
            pipeline, metadata = load_latest_bundle(registry_dir, model_name)
        else:
            pipeline, metadata = load_model_bundle(
                resolve_model_dir(registry_dir, model_name, version=version)
            )
        return cls(pipeline, metadata)

    @property
    def feature_order(self) -> List[str]:
        return list(self.metadata.feature_order)

    def _align_frame(self, records: Records) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, Mapping):
            frame = pd.DataFrame([dict(records)])
        else:
            frame = pd.DataFrame([dict(r) for r in records])
        if frame.empty:
            raise ValueError("No records provided for prediction.")

        for col in self.metadata.feature_order:
            if col not in frame.columns:
                frame[col] = np.nan
        for col in self.metadata.numeric_features:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame[self.metadata.feature_order]

    def predict(self, records: Records) -> np.ndarray:
        """Predict power (watts) for one or more telemetry records."""
        frame = self._align_frame(records)
        preds = self.pipeline.predict(frame)
        return np.asarray(preds, dtype=float)

    def predict_one(self, record: Record) -> float:
        """Predict power for a single telemetry record."""
        return float(self.predict(record)[0])


def load_model(
    registry_dir: str | Path,
    model_name: str,
    version: str = "latest",
) -> PowerModel:
    """Convenience entry point used by the CLI and API."""
    return PowerModel.from_registry(registry_dir, model_name, version=version)


def predict_records(model: PowerModel, records: Records) -> List[Dict[str, Any]]:
    """Return prediction rows pairing inputs with predicted power."""
    frame = model._align_frame(records)
    preds = model.pipeline.predict(frame)
    output: List[Dict[str, Any]] = []
    for i, pred in enumerate(np.asarray(preds, dtype=float)):
        output.append({"index": i, "predicted_power_watts": float(pred)})
    return output
