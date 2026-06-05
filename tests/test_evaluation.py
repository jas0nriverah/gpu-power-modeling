from pathlib import Path

from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.evaluation import metrics_table, save_metrics
from gpu_power_pipeline.train import train_and_evaluate


def test_metrics_table_sorted_by_rmse():
    df = load_dataset(source="synthetic", n_samples=400, random_state=2)
    results = train_and_evaluate(df, test_size=0.25, random_state=2)
    table = metrics_table(results)
    assert list(table.columns) == ["model", "mae", "rmse", "r2"]
    rmse_values = table["rmse"].tolist()
    assert rmse_values == sorted(rmse_values)


def test_save_metrics_writes_csv(tmp_path: Path):
    df = load_dataset(source="synthetic", n_samples=300, random_state=4)
    results = train_and_evaluate(df, test_size=0.25, random_state=4)
    table = save_metrics(results, out_dir=tmp_path)
    assert (tmp_path / "metrics.csv").exists()
    assert not table.empty
