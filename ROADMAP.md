# Roadmap

Planned improvements for the GPU Power Modeling project.

## Done

* Python package with data loading, preprocessing, training, evaluation, and reporting
* Synthetic telemetry generation
* Support for BMC, MIT Supercloud, local `nvidia-smi`, and NREL-style CSV inputs
* Dataset checks and leakage filtering
* Random, time-based, and grouped validation splits
* Feature ablations and hyperparameter sweeps
* Residual diagnostics, feature importance, and target-quality checks
* One-page report generation
* Model saving/loading with a versioned artifact layout
* CLI commands for `run`, `train`, `evaluate`, `predict`, `monitor`, and `serve`
* YAML experiment configs
* FastAPI inference service
* Dockerfile for serving
* Lightweight drift checks and prediction logs
* Optional MLflow / W&B tracking
* Optional XGBoost / LightGBM models and SHAP summaries
* GitHub Actions CI with tests, linting, and smoke runs
* MIT license

## Current focus

* Keep the README concise and accurate
* Keep examples reproducible
* Make the pipeline easy to run from a fresh clone
* Avoid overstating results from noisy real-world traces

## Future work

* Add one pinned example run under `examples/`
* Add batch or async inference
* Improve workload labels for real telemetry
* Test monitoring on larger datasets
* Add more real-trace examples when data quality is good enough

## Non-goals

* Proprietary or confidential hardware data
* Internal architecture assumptions
* Production-deployment claims without serving benchmarks
* Fabricated metrics or undocumented datasets
