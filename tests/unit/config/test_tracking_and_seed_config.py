"""ExperimentConfig.tracking / random_seed / deterministic: serialization, validation,
config hash, and how they reach the training stack (PipelineConfig)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.experiment import ExperimentConfig, TrackingSection
from src.core.config import TRACKING_BACKENDS


def _config(**overrides) -> ExperimentConfig:
    base = {"run_id": "fixed_run", "output_dir": "experiments/runs"}
    base.update(overrides)
    return ExperimentConfig(**base)


class TestTrackingSectionSerialization:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.tracking == TrackingSection(backend="none")
        assert cfg.deterministic is False

    @pytest.mark.parametrize("backend", TRACKING_BACKENDS)
    def test_dict_round_trip(self, backend: str) -> None:
        cfg = _config(random_seed=7, deterministic=True)
        cfg.tracking = TrackingSection(
            backend=backend, tracking_uri="http://mlflow:5000", experiment_name="mes_exp"
        )
        restored = ExperimentConfig.from_dict(cfg.to_dict())
        assert restored.tracking == cfg.tracking
        assert restored.random_seed == 7
        assert restored.deterministic is True
        assert restored.to_dict() == cfg.to_dict()

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        cfg = _config()
        cfg.tracking = TrackingSection(backend="local", tracking_uri=str(tmp_path / "trk"))
        path = tmp_path / "exp.yaml"
        cfg.save_yaml(path)
        restored = ExperimentConfig.from_yaml(path)
        assert restored.tracking == cfg.tracking
        assert restored.to_dict() == cfg.to_dict()

    def test_config_without_tracking_section_loads_defaults(self) -> None:
        data = _config().to_dict()
        del data["tracking"], data["deterministic"]
        restored = ExperimentConfig.from_dict(data)
        assert restored.tracking.backend == "none"
        assert restored.deterministic is False

    def test_unknown_backend_rejected(self) -> None:
        data = _config().to_dict()
        data["tracking"] = {"backend": "wandb"}
        with pytest.raises(ValueError, match="tracking.backend"):
            ExperimentConfig.from_dict(data)

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValueError, match="random_seed"):
            _config(random_seed=-1)


class TestConfigHash:
    def test_stable_and_hex(self) -> None:
        h = _config().config_hash()
        assert h == _config().config_hash()
        assert len(h) == 64 and int(h, 16) >= 0

    def test_ignores_run_identity_and_tracking(self) -> None:
        base = _config().config_hash()
        other = _config(run_id="another_run", output_dir="elsewhere", name="renamed", verbose=0)
        other.description = "a note"
        other.tracking = TrackingSection(backend="local")
        assert other.config_hash() == base

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda c: setattr(c, "random_seed", 7),
            lambda c: setattr(c, "deterministic", True),
            lambda c: setattr(c.training, "models", ["xgboost", "logistic"]),
            lambda c: setattr(c.data, "symbol", "MGC"),
            lambda c: setattr(c.training.optuna, "n_trials", 5),
        ],
    )
    def test_changes_with_result_affecting_fields(self, mutate) -> None:
        cfg = _config()
        base = cfg.config_hash()
        mutate(cfg)
        assert cfg.config_hash() != base


class TestReachesPipelineConfig:
    def test_seed_and_determinism(self) -> None:
        cfg = _config(random_seed=123, deterministic=True)
        pc = cfg.to_pipeline_config(cv_gaps=(15, 30))
        assert pc.random_state == 123
        assert pc.deterministic is True

    def test_tracking_none_by_default(self) -> None:
        pc = _config().to_pipeline_config(cv_gaps=(15, 30))
        assert pc.tracking_backend == "none"
        assert pc.tracking_uri is None
        assert pc.tracking_parent_run_id is None

    def test_local_backend_defaults_next_to_the_run_dirs(self, tmp_path: Path) -> None:
        cfg = _config(output_dir=tmp_path / "runs")
        cfg.tracking = TrackingSection(backend="local")
        pc = cfg.to_pipeline_config(cv_gaps=(15, 30), tracking_parent_run_id="parent123")
        assert pc.tracking_backend == "local"
        assert pc.tracking_uri == str(tmp_path / "runs" / "tracking")
        assert pc.tracking_experiment == cfg.name
        assert pc.tracking_parent_run_id == "parent123"

    def test_explicit_uri_and_experiment(self) -> None:
        cfg = _config()
        cfg.tracking = TrackingSection(
            backend="mlflow", tracking_uri="http://mlflow:5000", experiment_name="mes"
        )
        pc = cfg.to_pipeline_config(cv_gaps=(15, 30))
        assert (pc.tracking_uri, pc.tracking_experiment) == ("http://mlflow:5000", "mes")

    def test_mlflow_without_uri_uses_mlflow_default(self) -> None:
        cfg = _config()
        cfg.tracking = TrackingSection(backend="mlflow")
        assert cfg.tracking_location() is None
