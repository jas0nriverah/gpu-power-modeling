"""Command-line interface.

Subcommands:

- ``run``      end-to-end experiment (data -> train -> evaluate -> artifacts)
- ``train``    fit the best (or chosen) model on all data and save a bundle
- ``evaluate`` score a saved/freshly trained model and write metrics
- ``predict``  load a saved model and predict on a CSV/JSON of telemetry
- ``serve``    launch the FastAPI inference service

For backward compatibility, invoking with no subcommand (e.g.
``python -m gpu_power_pipeline --source synthetic``) behaves like ``run``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .audit import audit_dataset
from .config import ExperimentConfig
from .data import load_dataset
from .evaluation import metrics_table, save_metrics
from .experiments import (
    run_feature_ablation_experiments,
    run_grouped_blocked_cv,
    run_hyperparameter_sweep,
    run_split_comparison,
)
from .inference import PowerModel, predict_records
from .persistence import ModelMetadata, new_version_id, save_model_bundle
from .plotting import (
    plot_feature_importance,
    plot_predicted_vs_actual,
    plot_residual_distribution,
    plot_residuals,
    save_dataset_preview,
    save_feature_importance_table,
    save_permutation_importance_table,
)
from .preprocessing import TARGET_COL
from .quality import save_target_quality_summary, save_worst_prediction_errors
from .report import generate_one_page_report
from .train import (
    SPLIT_GROUPED_SESSION,
    SPLIT_RANDOM,
    SPLIT_TIME,
    ModelArtifacts,
    fit_full_pipeline,
    train_and_evaluate,
)

SUBCOMMANDS = {"run", "train", "evaluate", "predict", "serve"}


def _get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _write_run_metadata(extra: dict, source: str, out_dir: Path) -> None:
    metadata = {
        "source": source,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "git_commit": _get_git_commit(),
        **extra,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["synthetic", "nrel_eagle", "bmcdata_public"], default="synthetic")
    parser.add_argument("--data-path", type=str, default=None, help="CSV path for source=nrel_eagle.")
    parser.add_argument("--n-samples", type=int, default=12000, help="Synthetic sample count.")
    parser.add_argument("--bmcdata-dir", type=str, default="data/bmcdata_public")
    parser.add_argument("--bmcdata-max-files", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--include-mlp", action="store_true", help="Include sklearn MLP baseline.")
    parser.add_argument("--include-torch-mlp", action="store_true", help="Include PyTorch MLP if installed.")
    parser.add_argument(
        "--split-strategy",
        choices=[SPLIT_RANDOM, SPLIT_TIME, SPLIT_GROUPED_SESSION],
        default="time",
    )
    parser.add_argument("--test-size", type=float, default=0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu_power_pipeline", description="GPU Power Modeling pipeline")
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="End-to-end experiment run.")
    _add_common_data_args(run_p)
    _add_model_args(run_p)
    run_p.add_argument("--config", type=str, default=None, help="YAML experiment config.")
    run_p.add_argument("--outdir", type=str, default="outputs")
    run_p.add_argument("--registry-dir", type=str, default="artifacts")
    run_p.add_argument("--run-ablation", action="store_true")
    run_p.add_argument("--run-dual-datasets", action="store_true")
    run_p.add_argument("--run-sweep", action="store_true")
    run_p.add_argument("--generate-report", action="store_true")
    run_p.add_argument("--run-validation", action="store_true")
    run_p.add_argument("--cv-folds", type=int, default=3)
    run_p.add_argument("--save-model", action="store_true", help="Persist best model to the registry.")

    # train
    train_p = sub.add_parser("train", help="Fit a model on all data and save a bundle.")
    _add_common_data_args(train_p)
    _add_model_args(train_p)
    train_p.add_argument("--config", type=str, default=None)
    train_p.add_argument("--registry-dir", type=str, default="artifacts")
    train_p.add_argument("--model-name", type=str, default=None, help="Model to save (default: best by RMSE).")
    train_p.add_argument("--outdir", type=str, default="outputs")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="Train + evaluate and write metrics only.")
    _add_common_data_args(eval_p)
    _add_model_args(eval_p)
    eval_p.add_argument("--outdir", type=str, default="outputs_eval")

    # predict
    pred_p = sub.add_parser("predict", help="Predict from a saved model.")
    pred_p.add_argument("--registry-dir", type=str, default="artifacts")
    pred_p.add_argument("--model-name", type=str, required=True)
    pred_p.add_argument("--model-version", type=str, default="latest")
    pred_p.add_argument("--input", type=str, required=True, help="CSV or JSON of telemetry records.")
    pred_p.add_argument("--output", type=str, default=None, help="Optional CSV path for predictions.")

    # serve
    serve_p = sub.add_parser("serve", help="Launch the FastAPI inference service.")
    serve_p.add_argument("--registry-dir", type=str, default="artifacts")
    serve_p.add_argument("--model-name", type=str, default="random_forest")
    serve_p.add_argument("--model-version", type=str, default="latest")
    serve_p.add_argument("--host", type=str, default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    return parser


def _config_from_run_args(args: argparse.Namespace) -> ExperimentConfig:
    if getattr(args, "config", None):
        config = ExperimentConfig.from_yaml(args.config)
        return config
    config = ExperimentConfig()
    config.random_state = args.random_state
    config.outdir = getattr(args, "outdir", "outputs")
    config.registry_dir = getattr(args, "registry_dir", "artifacts")
    config.data.source = args.source
    config.data.data_path = args.data_path
    config.data.n_samples = args.n_samples
    config.data.bmcdata_dir = args.bmcdata_dir
    config.data.bmcdata_max_files = args.bmcdata_max_files
    config.split.strategy = args.split_strategy
    config.split.test_size = args.test_size
    config.split.cv_folds = getattr(args, "cv_folds", 3)
    config.models.include_mlp = args.include_mlp
    config.models.include_torch_mlp = args.include_torch_mlp
    config.experiments.run_ablation = getattr(args, "run_ablation", False)
    config.experiments.run_sweep = getattr(args, "run_sweep", False)
    config.experiments.run_validation = getattr(args, "run_validation", False)
    config.experiments.generate_report = getattr(args, "generate_report", False)
    return config


# --------------------------------------------------------------------------- #
# Shared pipeline logic
# --------------------------------------------------------------------------- #
def _load(config: ExperimentConfig, source: str) -> pd.DataFrame:
    return load_dataset(
        source=source,
        data_path=config.data.data_path,
        n_samples=config.data.n_samples,
        random_state=config.random_state,
        bmcdata_dir=config.data.bmcdata_dir,
        bmcdata_max_files=config.data.bmcdata_max_files,
    )


def _save_diagnostics(results: dict[str, ModelArtifacts], out_dir: Path) -> None:
    for model_name, artifacts in results.items():
        model_dir = out_dir / model_name
        plot_predicted_vs_actual(artifacts, model_dir)
        plot_residuals(artifacts, model_dir)
        plot_residual_distribution(artifacts, model_dir)
        plot_feature_importance(artifacts, model_dir, top_k=15)
        save_feature_importance_table(artifacts, model_dir)
        save_permutation_importance_table(artifacts, model_dir)
        save_worst_prediction_errors(artifacts, out_dir=out_dir / "quality", model_name=model_name)


def _build_metadata(
    artifacts: ModelArtifacts,
    source: str,
    split_strategy: str,
) -> ModelMetadata:
    return ModelMetadata(
        model_name=artifacts.model_name,
        target=TARGET_COL,
        numeric_features=artifacts.numeric_features,
        categorical_features=artifacts.categorical_features,
        feature_order=artifacts.input_features,
        metrics=artifacts.metrics,
        split_strategy=split_strategy,
        source=source,
        version=new_version_id(),
        git_commit=_get_git_commit(),
    )


def _save_best_model(
    df: pd.DataFrame,
    results: dict[str, ModelArtifacts],
    config: ExperimentConfig,
    source: str,
    model_name: Optional[str] = None,
) -> Path:
    table = metrics_table(results)
    chosen = model_name or table.iloc[0]["model"]
    artifacts = results[chosen]
    pipeline, _, _ = fit_full_pipeline(
        df=df,
        model_name=chosen,
        random_state=config.random_state,
        include_mlp=config.models.include_mlp,
        include_torch_mlp=config.models.include_torch_mlp,
        model_params=config.models.params,
    )
    metadata = _build_metadata(artifacts, source=source, split_strategy=config.split.strategy)
    version_dir = save_model_bundle(pipeline, metadata, registry_dir=config.registry_dir)
    print(f"Saved model '{chosen}' -> {version_dir}")
    return version_dir


def _run_single_dataset(
    config: ExperimentConfig,
    source: str,
    out_dir: Path,
    save_model: bool = False,
    metadata_extra: Optional[dict] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(extra=metadata_extra or {}, source=source, out_dir=out_dir)

    df = _load(config, source)
    save_dataset_preview(df, out_dir=out_dir)
    save_target_quality_summary(df, out_dir=out_dir / "quality")
    audit_dataset(df, out_dir=out_dir / "audit")

    results = train_and_evaluate(
        df,
        test_size=config.split.test_size,
        random_state=config.random_state,
        include_mlp=config.models.include_mlp,
        include_torch_mlp=config.models.include_torch_mlp,
        split_strategy=config.split.strategy,
        model_params=config.models.params or None,
    )
    metrics_df = save_metrics(results, out_dir=out_dir)
    _save_diagnostics(results, out_dir)

    if config.experiments.run_ablation:
        run_feature_ablation_experiments(
            df=df,
            out_dir=out_dir / "ablation",
            include_mlp=config.models.include_mlp,
            include_torch_mlp=config.models.include_torch_mlp,
            test_size=config.split.test_size,
            random_state=config.random_state,
            split_strategy=config.split.strategy,
        )
    if config.experiments.run_sweep:
        run_hyperparameter_sweep(
            df=df,
            out_dir=out_dir / "sweep",
            random_state=config.random_state,
            test_size=config.split.test_size,
            split_strategy=config.split.strategy,
            search_space=config.sweep_space,
        )
    if config.experiments.run_validation:
        run_split_comparison(
            df=df,
            out_dir=out_dir / "validation",
            random_state=config.random_state,
            test_size=config.split.test_size,
        )
        run_grouped_blocked_cv(
            df=df,
            out_dir=out_dir / "validation",
            random_state=config.random_state,
            n_splits=config.split.cv_folds,
        )
    if config.experiments.generate_report:
        generate_one_page_report(out_dir=out_dir, source=source)

    if save_model:
        _save_best_model(df=df, results=results, config=config, source=source)

    print(f"\nDataset source: {source}")
    print("Model metrics (lower MAE/RMSE is better; higher R2 is better):")
    print(metrics_df.to_string(index=False))
    print(f"Artifacts written to: {out_dir.resolve()}")


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> None:
    config = _config_from_run_args(args)
    root_out = Path(config.outdir)
    root_out.mkdir(parents=True, exist_ok=True)
    meta_extra = {"cli_args": vars(args), "config_name": config.name}

    if getattr(args, "run_dual_datasets", False):
        for source in ["synthetic", "bmcdata_public"]:
            _run_single_dataset(
                config=config,
                source=source,
                out_dir=root_out / source,
                save_model=getattr(args, "save_model", False),
                metadata_extra=meta_extra,
            )
        return

    _run_single_dataset(
        config=config,
        source=config.data.source,
        out_dir=root_out,
        save_model=getattr(args, "save_model", False),
        metadata_extra=meta_extra,
    )


def cmd_train(args: argparse.Namespace) -> None:
    config = _config_from_run_args(args)
    source = config.data.source
    df = _load(config, source)
    results = train_and_evaluate(
        df,
        test_size=config.split.test_size,
        random_state=config.random_state,
        include_mlp=config.models.include_mlp,
        include_torch_mlp=config.models.include_torch_mlp,
        split_strategy=config.split.strategy,
        model_params=config.models.params or None,
    )
    print("Evaluation before saving (held-out split):")
    print(metrics_table(results).to_string(index=False))
    _save_best_model(
        df=df,
        results=results,
        config=config,
        source=source,
        model_name=getattr(args, "model_name", None),
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = _config_from_run_args(args)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load(config, config.data.source)
    results = train_and_evaluate(
        df,
        test_size=config.split.test_size,
        random_state=config.random_state,
        include_mlp=config.models.include_mlp,
        include_torch_mlp=config.models.include_torch_mlp,
        split_strategy=config.split.strategy,
    )
    table = save_metrics(results, out_dir=out_dir)
    print(table.to_string(index=False))
    print(f"Metrics written to: {(out_dir / 'metrics.csv').resolve()}")


def _read_records(input_path: Path) -> list[dict]:
    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("records", [data])
        return list(data)
    frame = pd.read_csv(input_path)
    return frame.to_dict(orient="records")


def cmd_predict(args: argparse.Namespace) -> None:
    model = PowerModel.from_registry(args.registry_dir, args.model_name, version=args.model_version)
    records = _read_records(Path(args.input))
    rows = predict_records(model, records)
    out_df = pd.DataFrame(rows)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Predictions written to: {out_path.resolve()}")
    else:
        print(out_df.to_string(index=False))


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ["GPU_POWER_REGISTRY_DIR"] = args.registry_dir
    os.environ["GPU_POWER_MODEL_NAME"] = args.model_name
    os.environ["GPU_POWER_MODEL_VERSION"] = args.model_version
    uvicorn.run("gpu_power_pipeline.api:app", host=args.host, port=args.port, reload=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Backward compatibility: no subcommand -> treat as `run`.
    if not argv or (argv[0] not in SUBCOMMANDS and argv[0].startswith("-")):
        argv = ["run", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "run": cmd_run,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "serve": cmd_serve,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return
    handler(args)


if __name__ == "__main__":
    main()
