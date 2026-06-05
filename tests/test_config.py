from pathlib import Path

from gpu_power_pipeline.config import ExperimentConfig


def test_defaults_are_sane():
    config = ExperimentConfig()
    assert config.data.source == "synthetic"
    assert config.split.strategy in {"random", "time", "grouped"}
    assert "random_forest" in config.sweep_space


def test_from_dict_partial_override():
    config = ExperimentConfig.from_dict(
        {
            "name": "exp1",
            "data": {"source": "bmcdata_public", "bmcdata_max_files": 3},
            "split": {"strategy": "grouped"},
            "experiments": {"run_validation": True},
        }
    )
    assert config.name == "exp1"
    assert config.data.source == "bmcdata_public"
    assert config.data.bmcdata_max_files == 3
    assert config.split.strategy == "grouped"
    assert config.experiments.run_validation is True
    # Untouched fields keep defaults.
    assert config.random_state == 42


def test_from_yaml_roundtrip(tmp_path: Path):
    yaml_text = """
name: yaml_exp
random_state: 7
data:
  source: synthetic
  n_samples: 500
split:
  strategy: time
  test_size: 0.3
experiments:
  run_ablation: true
"""
    path = tmp_path / "exp.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    config = ExperimentConfig.from_yaml(path)
    assert config.name == "yaml_exp"
    assert config.random_state == 7
    assert config.data.n_samples == 500
    assert config.split.test_size == 0.3
    assert config.experiments.run_ablation is True
