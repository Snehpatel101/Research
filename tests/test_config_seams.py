"""Regression tests for config seam fixes.

Covers:
1. early_stopping_patience threading (request -> TrainerConfig -> model_config)
   and removal of the old ``max_epochs // 2`` heuristic.
2. Optuna timeout reaching the TimeSeriesOptunaTuner constructor.
3. ExperimentConfig.to_pipeline_config seams (optuna_timeout, optimize_features,
   mtf_timeframes).
4. YAML round-trip (safe_dump/safe_load) and ScalerConfig clip_range list->tuple.
5. TrainerConfig field-driven to_dict round-trip.
6. OptunaConfig n_trials=0 valid, negative rejected.
7. PipelineConfig regime_adx_threshold None-auto per symbol.

Uses only tiny synthetic 2D data; no model is ever trained (Trainer and the
Optuna tuner are monkeypatched to capture their configs).
"""

from __future__ import annotations

import numpy as np

from src.config.data import ScalerConfig
from src.config.experiment import ExperimentConfig
from src.config.training import OptunaConfig
from src.data.adapters.preparation import PreparedData

# =============================================================================
# HELPERS
# =============================================================================


def _tiny_prepared_data(n_train: int = 120, n_val: int = 40, n_features: int = 4) -> PreparedData:
    """Build a tiny 2D PreparedData for xgboost-style tabular training."""
    rng = np.random.RandomState(42)
    return PreparedData(
        X_train=rng.normal(size=(n_train, n_features)).astype(np.float32),
        y_train=rng.choice([-1, 0, 1], size=n_train).astype(np.int64),
        X_val=rng.normal(size=(n_val, n_features)).astype(np.float32),
        y_val=rng.choice([-1, 0, 1], size=n_val).astype(np.int64),
        model_name="xgboost",
        adapter_type="tabular",
        data_rank=2,
        feature_names=[f"f{i}" for i in range(n_features)],
    )


class _FakeTrainer:
    """Captures the TrainerConfig instead of training anything."""

    last_config = None

    def __init__(self, config):
        _FakeTrainer.last_config = config

    def run(self, container):
        return {"evaluation_metrics": {}}

    def run_prepared(self, prepared):
        return {"evaluation_metrics": {}}


def _run_train_model(monkeypatch, tmp_path, *, patience, optimize_hyperparams=False):
    """Drive ModelTrainingService.train_model with a fake Trainer, capture config."""
    import src.models as models_pkg
    from src.models.training.services.model_training import (
        ModelTrainingRequest,
        ModelTrainingService,
    )

    _FakeTrainer.last_config = None
    monkeypatch.setattr(models_pkg, "Trainer", _FakeTrainer)

    request = ModelTrainingRequest(
        model_name="xgboost",
        horizon=5,
        prepared_data=_tiny_prepared_data(),
        output_dir=tmp_path / "out",
        max_epochs=100,
        early_stopping_patience=patience,
        optimize_hyperparams=optimize_hyperparams,
        hyperparam_trials=1,
        n_splits=2,
        optuna_timeout=1234,
    )
    result = ModelTrainingService().train_model(request)
    return _FakeTrainer.last_config, result


class _FakeTuner:
    """Captures constructor kwargs instead of running Optuna."""

    captured: dict = {}

    def __init__(self, **kwargs):
        _FakeTuner.captured = dict(kwargs)

    def tune(self, X, y, sample_weights=None, param_space=None, data_rank=2):
        return {"best_params": {"n_estimators": 10}, "best_value": 0.5}


# =============================================================================
# 1. PATIENCE THREADING
# =============================================================================


class TestPatienceThreading:
    def test_explicit_patience_reaches_trainer_config_and_model_config(self, monkeypatch, tmp_path):
        config, result = _run_train_model(monkeypatch, tmp_path, patience=7)

        assert config is not None, "Fake Trainer never received a config"
        assert config.early_stopping_patience == 7
        assert config.model_config["early_stopping_patience"] == 7
        assert result.model_name == "xgboost"

    def test_none_patience_does_not_preset_model_config(self, monkeypatch, tmp_path):
        """With patience=None the old max_epochs//2 heuristic must NOT inject 50."""
        config, _ = _run_train_model(monkeypatch, tmp_path, patience=None)

        assert config is not None
        assert "early_stopping_patience" not in config.model_config, (
            "model_config must not pre-set early_stopping_patience when the "
            "request leaves it None (old max_epochs//2 heuristic resurfaced)"
        )
        # The deleted heuristic would have produced 100 // 2 == 50.
        assert config.early_stopping_patience != 50
        assert config.max_epochs == 100


# =============================================================================
# 2. OPTUNA TIMEOUT -> TUNER CONSTRUCTOR
# =============================================================================


class TestOptunaTimeoutThreading:
    def test_tuning_request_timeout_reaches_tuner_constructor(self, monkeypatch):
        """A tuning request carrying optuna_timeout=1234 passes it to the tuner."""
        import src.models.training.services.hyperparameter_tuning as ht

        monkeypatch.setattr(ht, "TimeSeriesOptunaTuner", _FakeTuner)
        _FakeTuner.captured = {}

        request = ht.TuningRequest(
            model_name="xgboost",
            horizon=5,
            prepared_data=_tiny_prepared_data(),
            n_splits=2,
            n_trials=1,
        )
        # The optimize() seam reads getattr(request, "optuna_timeout", None).
        request.optuna_timeout = 1234

        result = ht.HyperparameterTuningService().optimize(request)

        assert _FakeTuner.captured.get("timeout") == 1234
        assert _FakeTuner.captured.get("model_name") == "xgboost"
        assert result.best_params == {"n_estimators": 10}

    def test_model_training_request_timeout_reaches_tuner(self, monkeypatch, tmp_path):
        """End-to-end: ModelTrainingRequest(optuna_timeout=1234) -> tuner timeout."""
        import src.models.training.services.hyperparameter_tuning as ht

        monkeypatch.setattr(ht, "TimeSeriesOptunaTuner", _FakeTuner)
        _FakeTuner.captured = {}

        _run_train_model(monkeypatch, tmp_path, patience=None, optimize_hyperparams=True)

        assert _FakeTuner.captured.get("timeout") == 1234

    def test_model_training_request_has_optuna_timeout_field(self):
        from src.models.training.services.model_training import ModelTrainingRequest

        request = ModelTrainingRequest(
            model_name="xgboost",
            horizon=5,
            prepared_data=_tiny_prepared_data(),
            optuna_timeout=1234,
        )
        assert request.optuna_timeout == 1234


# =============================================================================
# 3. ExperimentConfig.to_pipeline_config SEAMS
# =============================================================================


def _experiment_config(**kwargs) -> ExperimentConfig:
    cfg = ExperimentConfig(run_id="fixed_run", **kwargs)
    cfg.data.symbol = "MES"
    cfg.data.data_path = "dummy.parquet"
    cfg.training.models = ["xgboost"]
    cfg.training.horizons = [5]
    cfg.training.purge_bars = 5
    cfg.training.embargo_bars = 60
    return cfg


class TestToPipelineConfig:
    def test_optuna_timeout_matches_training_optuna_timeout(self):
        cfg = _experiment_config()
        cfg.training.optuna.timeout = 777

        pipeline = cfg.to_pipeline_config()

        assert pipeline.optuna_timeout == 777
        assert pipeline.optuna_timeout == cfg.training.optuna.timeout

    def test_optimize_features_follows_selection_enabled_with_zero_trials(self):
        cfg = _experiment_config()
        cfg.training.optuna.n_trials = 0
        cfg.data.features.selection_enabled = True

        pipeline = cfg.to_pipeline_config()

        # Feature selection is MDA-based, independent of Optuna trial count.
        assert pipeline.optimize_features is True
        assert pipeline.optimize_hyperparams is False

    def test_optimize_features_disabled_when_selection_disabled(self):
        cfg = _experiment_config()
        cfg.training.optuna.n_trials = 0
        cfg.data.features.selection_enabled = False

        pipeline = cfg.to_pipeline_config()

        assert pipeline.optimize_features is False

    def test_mtf_disabled_yields_empty_timeframes(self):
        cfg = _experiment_config()
        cfg.data.mtf.enabled = False

        pipeline = cfg.to_pipeline_config()

        assert pipeline.mtf_timeframes == []

    def test_mtf_enabled_passes_timeframes_through(self):
        cfg = _experiment_config()
        cfg.data.mtf.enabled = True
        cfg.data.mtf.timeframes = ["15min", "60min"]

        pipeline = cfg.to_pipeline_config()

        assert pipeline.mtf_timeframes == ["15min", "60min"]


# =============================================================================
# 4. YAML ROUND-TRIP + ScalerConfig NORMALIZATION
# =============================================================================


class TestYamlRoundTrip:
    def test_save_yaml_from_yaml_round_trip(self, tmp_path):
        cfg = ExperimentConfig(run_id="fixed_run", output_dir=str(tmp_path / "runs"))
        cfg.data.symbol = "MGC"
        cfg.training.models = ["xgboost", "lightgbm"]
        cfg.training.horizons = [5, 10]

        path = tmp_path / "config.yaml"
        cfg.save_yaml(path)

        restored = ExperimentConfig.from_yaml(path)

        assert restored.to_dict() == cfg.to_dict()

    def test_save_yaml_uses_safe_load_compatible_output(self, tmp_path):
        """safe_dump must not emit python-specific tags (e.g. !!python/tuple)."""
        cfg = ExperimentConfig(run_id="fixed_run", output_dir=str(tmp_path / "runs"))
        path = tmp_path / "config.yaml"
        cfg.save_yaml(path)

        text = path.read_text()
        assert "!!python" not in text

    def test_scaler_config_normalizes_list_clip_range_to_tuple(self):
        scaler = ScalerConfig(clip_range=[-3, 3])

        assert isinstance(scaler.clip_range, tuple)
        assert scaler.clip_range == (-3.0, 3.0)


# =============================================================================
# 5. TrainerConfig FIELD-DRIVEN to_dict ROUND-TRIP
# =============================================================================


class TestTrainerConfigRoundTrip:
    def test_to_dict_contains_previously_dropped_fields(self, tmp_path):
        from src.models.config.trainer_config import TrainerConfig

        config = TrainerConfig(
            model_name="xgboost",
            horizon=5,
            pipeline_run_id="run_123",
            output_dir=tmp_path,
        )
        d = config.to_dict()

        assert d["pipeline_run_id"] == "run_123"
        assert "feature_selection_min_frequency" in d
        assert d["feature_selection_min_frequency"] == 0.6

    def test_from_dict_to_dict_round_trip(self, tmp_path):
        from src.models.config.trainer_config import TrainerConfig

        config = TrainerConfig(
            model_name="xgboost",
            horizon=5,
            pipeline_run_id="run_123",
            early_stopping_patience=7,
            output_dir=tmp_path,
        )
        d = config.to_dict()

        restored = TrainerConfig.from_dict(d)

        assert restored.to_dict() == d
        assert restored.early_stopping_patience == 7


# =============================================================================
# 6. OptunaConfig VALIDATION
# =============================================================================


class TestOptunaConfigValidation:
    def test_zero_trials_is_valid(self):
        assert OptunaConfig(n_trials=0).validate() == []

    def test_negative_trials_rejected(self):
        issues = OptunaConfig(n_trials=-1).validate()

        assert issues, "n_trials=-1 must produce validation issues"
        assert any("n_trials" in issue for issue in issues)


# =============================================================================
# 7. PipelineConfig regime_adx_threshold NONE-AUTO
# =============================================================================


def _pipeline_config(tmp_path, symbol, **kwargs):
    from src.core import PipelineConfig

    return PipelineConfig(
        symbol=symbol,
        data_path=str(tmp_path / "data.parquet"),
        output_dir=str(tmp_path / "out"),
        models=["xgboost"],
        horizons=[5],
        **kwargs,
    )


class TestRegimeAdxThreshold:
    def test_none_auto_resolves_mes_preset(self, tmp_path):
        config = _pipeline_config(tmp_path, "MES")

        assert config.regime_adx_threshold == 20.0

    def test_none_auto_resolves_mgc_preset(self, tmp_path):
        config = _pipeline_config(tmp_path, "MGC")

        assert config.regime_adx_threshold == 23.0

    def test_explicit_value_honored_over_preset(self, tmp_path):
        config = _pipeline_config(tmp_path, "MES", regime_adx_threshold=25.0)

        assert config.regime_adx_threshold == 25.0
