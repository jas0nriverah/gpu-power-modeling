# Roadmap

This file tracks planned improvements for the **GPU Power Modeling** portfolio project.
It replaces the earlier `PLAN.md`, which described a different CUDA kernel analyzer and no longer matched this repository.

## Completed

- Modular Python package (`src/gpu_power_pipeline/`) with data loading, preprocessing, training, and evaluation
- Public dataset support (`arealuser/bmcdata`, MIT Supercloud HPCA22), local `nvidia-smi` collection, and synthetic telemetry generation
- Leakage guards (name-based + correlation-based feature filtering, dataset audit)
- Split strategies: random, time-ordered, and grouped by `session_id`
- Grouped blocked cross-validation as the primary evaluation metric
- Experiment tooling: feature ablations, hyperparameter sweeps, split comparison
- Residual diagnostics, permutation importance, target-quality checks, one-page report generator
- GitHub Actions CI for `pytest` and a synthetic smoke run
- Run metadata logging (`run_metadata.json`)

### Phase 1 - Project clarity and packaging

- [x] README rewrite: concise summary, architecture diagram, data labels
- [x] Add `LICENSE` (MIT)
- [x] Remove stale / conflicting planning docs
- [x] Initialize git repository and clean ignored build artifacts

### Phase 2 - ML engineering workflow

- [x] Persist trained models (`joblib`) with versioned artifact layout
- [x] Separate CLI commands: `train`, `evaluate`, `predict`, `serve`
- [x] YAML experiment configs instead of hardcoded sweep/feature values
- [x] Document reproducible train, evaluate, and report workflow
- [x] Dataset adapter registry and dataset suitability reports

### Phase 3 - Production-style features

- [x] FastAPI inference endpoint (`/health`, `/model`, `/predict`)
- [x] `Dockerfile` for consistent serving
- [x] Expand CI: linting (ruff) + train/predict smoke + inference tests
- [x] Tests for preprocessing, model load/save, inference, evaluation, and API
- [x] Lightweight monitoring: reference profiles, input drift checks, prediction logs

### Phase 4 - Documentation polish

- [x] README: design notes for models, validation, leakage, limitations, and deployment

## Remaining / future

- [ ] Pin one reproducible run bundle under `examples/`
- [ ] Experiment tracking (MLflow or Weights & Biases)
- [ ] Batch/async inference
- [ ] Stronger monitoring on larger real datasets
- [ ] Optional XGBoost/LightGBM models with SHAP interpretability

## Non-goals

- Proprietary hardware data, internal architecture assumptions, or confidential workflows
- Claiming production-grade deployment without measured serving benchmarks
- Fabricated metrics or datasets in documentation
