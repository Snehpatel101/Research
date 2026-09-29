"""The run seed (ExperimentConfig.random_seed -> PipelineConfig.random_state) reaches
every consumer: model configs (random_state / random_seed), the Optuna sampler and
every trial's model, and the stacking meta-learner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig, TrackingSection
from src.core.reproducibility import apply_model_seed
from src.models.config import TrainerConfig
from src.models.training.services.model_training import ModelTrainingRequest
from src.models.training.trainer import Trainer
from tests.helpers import tiny_prepared_data


class TestApplyModelSeed:
    def test_sets_both_keys(self) -> None:
        assert apply_model_seed({}, 7) == {"random_state": 7, "random_seed": 7}

    def test_explicit_seed_wins(self) -> None:
        cfg = apply_model_seed({"random_state": 3}, 7)
        assert cfg == {"random_state": 3, "random_seed": 7}


class TestTrainerSeedsModel:
    @pytest.mark.parametrize("model_name", ["xgboost", "logistic", "random_forest"])
    def test_run_seed_reaches_model(self, model_name: str, tmp_path: Path) -> None:
        trainer = Trainer(TrainerConfig(model_name=model_name, random_seed=7, output_dir=tmp_path))
        assert trainer.model.config["random_state"] == 7
        assert trainer.model.config["random_seed"] == 7

    def test_explicit_model_seed_kept(self, tmp_path: Path) -> None:
        trainer = Trainer(
            TrainerConfig(
                model_name="xgboost",
                random_seed=7,
                output_dir=tmp_path,
                model_config={"random_state": 3},
            )
        )
        assert trainer.model.config["random_state"] == 3

    def test_deterministic_mode_bridged(self, tmp_path: Path) -> None:
        trainer = Trainer(
            TrainerConfig(model_name="lstm", deterministic_mode=True, output_dir=tmp_path)
        )
        assert trainer.model.config["deterministic_mode"] is True


class TestTrainingRequestFromPipelineConfig:
    def _pipeline_config(self, tmp_path: Path) -> Any:
        cfg = ExperimentConfig(run_id="r", output_dir=tmp_path, random_seed=99, deterministic=True)
        cfg.tracking = TrackingSection(backend="local")
        return cfg.to_pipeline_config(cv_gaps=(15, 30), tracking_parent_run_id="parent")

    def test_run_settings_carried(self, tmp_path: Path) -> None:
        pc = self._pipeline_config(tmp_path)
        request = ModelTrainingRequest.from_pipeline_config(
            pc,
            model_name="xgboost",
            horizon=5,
            prepared_data=tiny_prepared_data(),
            output_dir=tmp_path,
        )
        assert request.random_seed == 99
        assert request.deterministic is True
        assert request.tracking_backend == "local"
        assert request.tracking_uri == pc.tracking_uri
        assert request.tracking_parent_run_id == "parent"
        assert (request.purge_bars, request.embargo_bars) == (15, 30)
        assert request.batch_size == pc.batch_size
        assert request.n_classes == pc.n_classes

    def test_overrides_win(self, tmp_path: Path) -> None:
        request = ModelTrainingRequest.from_pipeline_config(
            self._pipeline_config(tmp_path),
            model_name="xgboost",
            horizon=5,
            prepared_data=tiny_prepared_data(),
            output_dir=tmp_path,
            batch_size=16,
            optimize_hyperparams=False,
        )
        assert request.batch_size == 16
        assert request.optimize_hyperparams is False


class TestTunerSeed:
    def test_sampler_and_trial_models_seeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import optuna.samplers

        from src.validation.cv import (
            PurgedKFold,
            PurgedKFoldConfig,
            TimeSeriesOptunaTuner,
            cv_tuner,
        )

        sampler_seeds: list[int | None] = []
        real_sampler = optuna.samplers.TPESampler

        def recording_sampler(*args: Any, **kwargs: Any) -> Any:
            sampler_seeds.append(kwargs.get("seed"))
            return real_sampler(*args, **kwargs)

        model_configs: list[dict[str, Any]] = []
        real_create = cv_tuner.ModelRegistry.create

        def recording_create(name: str, config: dict[str, Any] | None = None) -> Any:
            model_configs.append(dict(config or {}))
            return real_create(name, config=config)

        monkeypatch.setattr(optuna.samplers, "TPESampler", recording_sampler)
        monkeypatch.setattr(cv_tuner.ModelRegistry, "create", staticmethod(recording_create))

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(300, 4)), columns=[f"f{i}" for i in range(4)])
        y = pd.Series(rng.choice([-1, 0, 1], size=300))
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=2, embargo_bars=2))
        tuner = TimeSeriesOptunaTuner(model_name="logistic", cv=cv, n_trials=1, seed=1234)
        tuner.tune(X, y)

        assert sampler_seeds == [1234]
        assert model_configs, "no trial model was built"
        assert all(c["random_state"] == 1234 for c in model_configs)


def test_meta_learner_takes_random_state() -> None:
    from src.models.ensemble import get_meta_learner

    meta = get_meta_learner("ridge_meta", n_classes=3, random_state=5)
    assert meta.config["random_state"] == 5
