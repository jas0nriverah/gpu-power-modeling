"""FastAPI inference service for power predictions.

Loads a persisted model bundle from the artifact registry and exposes:

- ``GET /health``  - readiness + loaded model identity
- ``GET /model``   - metadata describing the loaded model
- ``POST /predict`` - batch power predictions from telemetry records

Configure via environment variables (with sensible defaults):

- ``GPU_POWER_REGISTRY_DIR`` (default ``artifacts``)
- ``GPU_POWER_MODEL_NAME``   (default ``random_forest``)
- ``GPU_POWER_MODEL_VERSION`` (default ``latest``)
- ``GPU_POWER_PREDICTION_LOG`` (optional JSONL prediction log path)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .inference import PowerModel, load_model
from .monitoring import append_prediction_log


class TelemetryRecord(BaseModel):
    """A single telemetry observation. Unknown fields are allowed and ignored
    unless the loaded model expects them."""

    model_config = ConfigDict(extra="allow")

    gpu_utilization_pct: Optional[float] = None
    memory_utilization_pct: Optional[float] = None
    graphics_clock_mhz: Optional[float] = None
    memory_clock_mhz: Optional[float] = None
    temperature_c: Optional[float] = None
    ambient_c: Optional[float] = None
    workload_type: Optional[str] = None


class PredictRequest(BaseModel):
    records: List[TelemetryRecord] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    index: int
    predicted_power_watts: float


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None


def create_app(
    registry_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI app, attempting to load the configured model."""
    registry_dir = registry_dir or os.environ.get("GPU_POWER_REGISTRY_DIR", "artifacts")
    model_name = model_name or os.environ.get("GPU_POWER_MODEL_NAME", "random_forest")
    version = version or os.environ.get("GPU_POWER_MODEL_VERSION", "latest")
    prediction_log = os.environ.get("GPU_POWER_PREDICTION_LOG")

    app = FastAPI(
        title="GPU Power Modeling API",
        version="0.1.0",
        description="Predict power (watts) from telemetry using a trained model bundle.",
    )

    state: Dict[str, Any] = {"model": None, "error": None}

    def _try_load() -> Optional[PowerModel]:
        try:
            model = load_model(registry_dir, model_name, version=version)
            state["model"] = model
            state["error"] = None
            return model
        except Exception as exc:  # noqa: BLE001 - surfaced via /health and 503s
            state["model"] = None
            state["error"] = str(exc)
            return None

    _try_load()

    def _get_model() -> PowerModel:
        model = state["model"] or _try_load()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded: {state.get('error') or 'unavailable'}",
            )
        return model

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model = state["model"]
        return HealthResponse(
            status="ok" if model is not None else "degraded",
            model_loaded=model is not None,
            model_name=model.metadata.model_name if model else None,
            model_version=model.metadata.version if model else None,
        )

    @app.get("/model")
    def model_info() -> Dict[str, Any]:
        model = _get_model()
        meta = model.metadata
        return {
            "model_name": meta.model_name,
            "version": meta.version,
            "target": meta.target,
            "feature_order": meta.feature_order,
            "metrics": meta.metrics,
            "split_strategy": meta.split_strategy,
            "source": meta.source,
            "created_utc": meta.created_utc,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model = _get_model()
        records = [r.model_dump(exclude_none=False) for r in request.records]
        try:
            preds = model.predict(records)
        except Exception as exc:  # noqa: BLE001 - bad input -> 400
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
        if prediction_log:
            try:
                append_prediction_log(
                    log_path=prediction_log,
                    model_name=model.metadata.model_name,
                    model_version=model.metadata.version,
                    input_frame=model._align_frame(records),
                    predictions=preds,
                )
            except Exception as exc:  # noqa: BLE001 - logging should not block inference
                state["prediction_log_error"] = str(exc)
        return PredictResponse(
            model_name=model.metadata.model_name,
            model_version=model.metadata.version,
            predictions=[
                PredictionItem(index=i, predicted_power_watts=float(p))
                for i, p in enumerate(preds)
            ],
        )

    return app


app = create_app()
