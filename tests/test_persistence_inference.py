from pathlib import Path

import numpy as np

from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.inference import PowerModel, predict_records
from gpu_power_pipeline.persistence import (
    ModelMetadata,
    load_latest_bundle,
    new_version_id,
    save_model_bundle,
)
from gpu_power_pipeline.train import fit_full_pipeline


def _train_and_save(tmp_path: Path, model_name: str = "random_forest"):
    df = load_dataset(source="synthetic", n_samples=400, random_state=1)
    pipeline, numeric, categorical = fit_full_pipeline(df, model_name, random_state=1)
    metadata = ModelMetadata(
        model_name=model_name,
        target="power_watts",
        numeric_features=numeric,
        categorical_features=categorical,
        feature_order=numeric + categorical,
        metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        split_strategy="time",
        source="synthetic",
        version=new_version_id(),
    )
    version_dir = save_model_bundle(pipeline, metadata, registry_dir=tmp_path / "artifacts")
    return df, version_dir


def test_save_load_predict_roundtrip(tmp_path: Path):
    df, version_dir = _train_and_save(tmp_path)
    assert (version_dir / "model.joblib").exists()
    assert (version_dir / "metadata.json").exists()

    model = PowerModel.from_dir(version_dir)
    record = df.drop(columns=["power_watts"]).iloc[0].to_dict()
    pred = model.predict_one(record)
    assert np.isfinite(pred)


def test_registry_latest_resolution(tmp_path: Path):
    _train_and_save(tmp_path)
    pipeline, metadata = load_latest_bundle(tmp_path / "artifacts", "random_forest")
    assert metadata.model_name == "random_forest"
    assert pipeline is not None


def test_predict_tolerates_missing_columns(tmp_path: Path):
    df, version_dir = _train_and_save(tmp_path)
    model = PowerModel.from_dir(version_dir)
    rows = predict_records(model, [{"gpu_utilization_pct": 55.0, "workload_type": "training"}])
    assert len(rows) == 1
    assert np.isfinite(rows[0]["predicted_power_watts"])
