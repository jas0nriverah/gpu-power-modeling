from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _to_markdown_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df.empty:
        return "_No rows._"
    sub = df.head(max_rows).copy()
    cols = sub.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in sub.iterrows():
        body.append("| " + " | ".join([str(row[c]) for c in cols]) + " |")
    return "\n".join([header, sep] + body)


def generate_one_page_report(out_dir: Path, source: str) -> Path:
    """Generate concise markdown report with primary vs secondary metric policy."""
    metrics_path = out_dir / "metrics.csv"
    ablation_path = out_dir / "ablation" / "feature_ablation_best_by_set.csv"
    sweep_path = out_dir / "sweep" / "hyperparameter_sweep_best.csv"
    split_best_path = out_dir / "validation" / "split_comparison_best.csv"
    cv_summary_path = out_dir / "validation" / "grouped_blocked_cv_summary.csv"
    audit_path = out_dir / "audit" / "data_audit_summary.csv"
    target_quality_path = out_dir / "quality" / "target_summary.csv"

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    ablation = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
    sweep = pd.read_csv(sweep_path) if sweep_path.exists() else pd.DataFrame()
    split_best = pd.read_csv(split_best_path) if split_best_path.exists() else pd.DataFrame()
    cv_summary = pd.read_csv(cv_summary_path) if cv_summary_path.exists() else pd.DataFrame()
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    target_quality = pd.read_csv(target_quality_path) if target_quality_path.exists() else pd.DataFrame()

    best_model = metrics.sort_values("rmse").iloc[0]["model"] if not metrics.empty else "n/a"
    best_plot = f"{best_model}/{best_model}_pred_vs_actual.png" if best_model != "n/a" else ""
    residual_plot = f"{best_model}/{best_model}_residual_distribution.png" if best_model != "n/a" else ""
    grouped_primary = cv_summary.head(1) if not cv_summary.empty else pd.DataFrame()
    secondary = split_best[split_best["split_strategy"].isin(["random", "time"])] if not split_best.empty else pd.DataFrame()
    grouped_invalid_warning = ""
    if not grouped_primary.empty:
        r2_val = float(grouped_primary.iloc[0].get("r2", np.nan))
        rmse_val = float(grouped_primary.iloc[0].get("rmse", np.nan))
        risk = False
        if not target_quality.empty and "invalid_grouped_metric_risk" in target_quality.columns:
            risk = bool(target_quality.iloc[0]["invalid_grouped_metric_risk"])
        if risk or (np.isfinite(r2_val) and r2_val < -1.0) or (np.isfinite(rmse_val) and rmse_val > 5000):
            grouped_invalid_warning = (
                "**Warning:** grouped metrics appear dominated by outliers or data-quality issues. "
                "Check `quality/target_summary.csv` and `quality/*worst_prediction_rows.csv` before trusting headline numbers."
            )

    report_md = f"""# One-Page Experiment Report

## Context
- Dataset source: `{source}`
- Objective: predict `power_watts` from public/synthetic telemetry.

## Model Comparison
{_to_markdown_table(metrics)}

## Target Quality Summary
{_to_markdown_table(target_quality)}

## Recommended Primary Metric (Use This)
Grouped blocked CV is the primary reported metric for this project.
It holds out complete sessions/files and best approximates deployment on unseen traces.
{_to_markdown_table(grouped_primary)}
{grouped_invalid_warning}

## Secondary Diagnostic Metrics (Optimistic)
Random/time split results are retained only as secondary diagnostics and can be optimistic.
{_to_markdown_table(secondary)}

## Best Ablation Per Feature Set
{_to_markdown_table(ablation)}

## Best Hyperparameter Sweep Results
{_to_markdown_table(sweep)}

## Split Strategy Comparison (Best Per Strategy)
{_to_markdown_table(split_best)}

## Grouped Blocked CV Summary
{_to_markdown_table(cv_summary)}

Grouped split is the most realistic because it holds out entire trace sessions rather than nearby rows.  
That prevents temporal/session memorization and better estimates performance on unseen workload captures.

## Leakage/Audit Snapshot
{_to_markdown_table(audit)}

## Selected Plots
![best_pred_vs_actual]({best_plot})
![best_residual_distribution]({residual_plot})
"""
    out = out_dir / "ONE_PAGE_REPORT.md"
    out.write_text(report_md, encoding="utf-8")
    return out
