"""Experiment configuration loaded from YAML instead of hardcoded values.

Keeps tunable knobs (data source, split strategy, model params, sweep grid)
in version-controlled config files so experiments are reproducible and easy to
diff. All fields have defaults, so an empty/partial YAML still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml

    YAML_AVAILABLE = True
except Exception:  # pragma: no cover - yaml is a declared dependency
    YAML_AVAILABLE = False


DEFAULT_SWEEP_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "random_forest": {
        "n_estimators": [120, 220],
        "max_depth": [8, 14],
        "min_samples_leaf": [1, 2],
    },
    "gradient_boosting": {
        "n_estimators": [120, 220],
        "learning_rate": [0.03, 0.05],
        "max_depth": [2, 3],
    },
}


@dataclass
class DataConfig:
    source: str = "synthetic"
    data_path: Optional[str] = None
    n_samples: int = 12000
    max_rows: Optional[int] = None
    bmcdata_dir: str = "data/bmcdata_public"
    bmcdata_max_files: int = 10


@dataclass
class SplitConfig:
    strategy: str = "time"
    test_size: float = 0.2
    cv_folds: int = 3


@dataclass
class ModelConfig:
    include_mlp: bool = False
    include_torch_mlp: bool = False
    include_xgboost: bool = False
    include_lightgbm: bool = False
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExperimentToggles:
    run_ablation: bool = False
    run_sweep: bool = False
    run_validation: bool = False
    generate_report: bool = False
    run_shap: bool = False
    track_mlflow: bool = False
    track_wandb: bool = False


@dataclass
class TrackingConfig:
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "gpu-power-modeling"
    wandb_project: str = "gpu-power-modeling"
    wandb_mode: str = "offline"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default"
    random_state: int = 42
    outdir: str = "outputs"
    registry_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    experiments: ExperimentToggles = field(default_factory=ExperimentToggles)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    sweep_space: Dict[str, Dict[str, List[Any]]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_SWEEP_SPACE.items()}
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        data = dict(data or {})
        return cls(
            name=data.get("name", "default"),
            random_state=int(data.get("random_state", 42)),
            outdir=data.get("outdir", "outputs"),
            registry_dir=data.get("registry_dir", "artifacts"),
            data=DataConfig(**(data.get("data") or {})),
            split=SplitConfig(**(data.get("split") or {})),
            models=ModelConfig(**(data.get("models") or {})),
            experiments=ExperimentToggles(**(data.get("experiments") or {})),
            tracking=TrackingConfig(**(data.get("tracking") or {})),
            sweep_space=data.get("sweep_space") or {k: dict(v) for k, v in DEFAULT_SWEEP_SPACE.items()},
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        if not YAML_AVAILABLE:
            raise RuntimeError("PyYAML is required to load YAML configs. Install pyyaml.")
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw or {})
