from pathlib import Path

import pytest

from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.persistence import ModelMetadata, new_version_id, save_model_bundle
from gpu_power_pipeline.train import fit_full_pipeline

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from gpu_power_pipeline.api import create_app  # noqa: E402


def _seed_registry(tmp_path: Path, model_name: str = "random_forest") -> Path:
    df = load_dataset(source="synthetic", n_samples=400, random_state=1)
    pipeline, numeric, categorical = fit_full_pipeline(df, model_name, random_state=1)
    metadata = ModelMetadata(
        model_name=model_name,
        target="power_watts",
        numeric_features=numeric,
        categorical_features=categorical,
        feature_order=numeric + categorical,
        metrics={"rmse": 1.0},
        split_strategy="time",
        source="synthetic",
        version=new_version_id(),
    )
    save_model_bundle(pipeline, metadata, registry_dir=tmp_path / "artifacts")
    return tmp_path / "artifacts"


def test_health_and_predict(tmp_path: Path):
    registry = _seed_registry(tmp_path)
    app = create_app(registry_dir=str(registry), model_name="random_forest")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    payload = {
        "records": [
            {
                "gpu_utilization_pct": 70,
                "memory_utilization_pct": 60,
                "graphics_clock_mhz": 1500,
                "memory_clock_mhz": 5000,
                "temperature_c": 65,
                "ambient_c": 23,
                "workload_type": "training",
            }
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["predicted_power_watts"] > 0


def test_health_degraded_when_no_model(tmp_path: Path):
    app = create_app(registry_dir=str(tmp_path / "empty"), model_name="random_forest")
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    # Predict should fail clearly when no model is available.
    resp = client.post("/predict", json={"records": [{"gpu_utilization_pct": 50}]})
    assert resp.status_code == 503
