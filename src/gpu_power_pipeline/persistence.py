"""Model persistence and a lightweight versioned artifact registry.

A *model bundle* is a fitted scikit-learn :class:`~sklearn.pipeline.Pipeline`
plus :class:`ModelMetadata` describing the features it expects and how it was
trained. Bundles are written under a registry directory laid out as::

    <registry>/
        registry.json                # index of models -> versions
        <model_name>/
            <version>/
                model.joblib
                metadata.json

This makes trained models reproducible and serveable without retraining.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import sklearn
from sklearn.pipeline import Pipeline

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
REGISTRY_INDEX = "registry.json"


@dataclass
class ModelMetadata:
    """Describes a persisted model bundle and its training context."""

    model_name: str
    target: str
    numeric_features: List[str]
    categorical_features: List[str]
    feature_order: List[str]
    metrics: Dict[str, float]
    split_strategy: str
    source: str
    version: str
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=platform.python_version)
    sklearn_version: str = field(default_factory=lambda: sklearn.__version__)
    git_commit: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_model_bundle(
    pipeline: Pipeline,
    metadata: ModelMetadata,
    registry_dir: str | Path,
) -> Path:
    """Persist a fitted pipeline + metadata under the registry.

    Returns the version directory that was written and updates the registry
    index so the bundle can later be resolved as the ``latest`` for its name.
    """
    registry_path = Path(registry_dir)
    version_dir = registry_path / metadata.model_name / metadata.version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, version_dir / MODEL_FILENAME)
    (version_dir / METADATA_FILENAME).write_text(metadata.to_json(), encoding="utf-8")

    _update_registry_index(registry_path, metadata.model_name, metadata.version)
    return version_dir


def _update_registry_index(registry_path: Path, model_name: str, version: str) -> None:
    registry_path.mkdir(parents=True, exist_ok=True)
    index_path = registry_path / REGISTRY_INDEX
    index: Dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            index = {}
    entry = index.get(model_name, {"versions": [], "latest": None})
    if version not in entry["versions"]:
        entry["versions"].append(version)
    entry["versions"].sort()
    entry["latest"] = entry["versions"][-1]
    index[model_name] = entry
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def load_model_bundle(version_dir: str | Path) -> Tuple[Pipeline, ModelMetadata]:
    """Load a pipeline + metadata from an explicit version directory."""
    version_path = Path(version_dir)
    model_path = version_path / MODEL_FILENAME
    metadata_path = version_path / METADATA_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"No model file at {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata file at {metadata_path}")
    pipeline = joblib.load(model_path)
    metadata = ModelMetadata.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
    return pipeline, metadata


def resolve_model_dir(registry_dir: str | Path, model_name: str, version: str = "latest") -> Path:
    """Resolve a registry entry to a concrete version directory."""
    registry_path = Path(registry_dir)
    if version != "latest":
        return registry_path / model_name / version
    index_path = registry_path / REGISTRY_INDEX
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entry = index.get(model_name)
        if entry and entry.get("latest"):
            return registry_path / model_name / entry["latest"]
    # Fall back to highest-sorted version directory on disk.
    model_root = registry_path / model_name
    if model_root.is_dir():
        versions = sorted(p.name for p in model_root.iterdir() if p.is_dir())
        if versions:
            return model_root / versions[-1]
    raise FileNotFoundError(f"No persisted versions for model '{model_name}' under {registry_path}")


def load_latest_bundle(registry_dir: str | Path, model_name: str) -> Tuple[Pipeline, ModelMetadata]:
    """Convenience loader for the latest version of a named model."""
    return load_model_bundle(resolve_model_dir(registry_dir, model_name, version="latest"))


def new_version_id() -> str:
    """Return a fresh UTC-based version identifier (exposed for callers)."""
    return _new_version()
