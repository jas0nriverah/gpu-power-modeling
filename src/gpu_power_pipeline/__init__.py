"""GPU Power Modeling pipeline.

A leakage-aware regression pipeline that predicts ``power_watts`` from public
or synthetic telemetry, with grouped session cross-validation, experiment
tooling, model persistence, and a FastAPI inference layer.
"""

__version__ = "0.1.0"

__all__ = [
    "audit",
    "cli",
    "config",
    "data",
    "evaluation",
    "experiments",
    "inference",
    "persistence",
    "plotting",
    "preprocessing",
    "quality",
    "report",
    "train",
]
