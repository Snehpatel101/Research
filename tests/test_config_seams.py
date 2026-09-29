"""Regression tests for config seam fixes.

Covers:
1. early_stopping_patience threading (request -> TrainerConfig -> model_config)
   and removal of the old ``max_epochs // 2`` heuristic.
2. Optuna timeout reaching the TimeSeriesOptunaTuner constructor.
3. ExperimentConfig.to_pipeline_config seams (optuna_timeout, optimize_features,
   mtf_timeframes).
4. YAML round-trip (safe_dump/safe_load); old YAML with removed/unknown keys
   loads with a warning instead of crashing.
5. TrainerConfig field-driven to_dict round-trip.
6. OptunaConfig n_trials=0 valid, negative rejected.
7. PipelineConfig regime_adx_threshold None-auto per symbol.
8. Phase 116 wiring: split ratios, calibration enabled/method (down to
   TrainerConfig) and ExperimentConfig.verbose reach the code that runs.

Uses only tiny synthetic 2D data; no model is ever trained (Trainer and the
Optuna tuner are monkeypatched to capture their configs).
"""

from __future__ import annotations

import numpy as np

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


def _run_train_model(monkeypatch, tmp_path, *, patience, optimize_hyperparams=False, **extra):
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
        **extra,
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
# 4. YAML ROUND-TRIP + BACKWARD-COMPATIBLE LOADING
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

    def test_removed_and_unknown_keys_warn_and_are_ignored(self, caplog):
        """YAML written before Phase 116 still loads; dropped keys are reported."""
        old = {
            "run_id": "fixed_run",
            "data": {
                "symbol": "MGC",
                "scaler": {"scaler_type": "robust", "clip_range": [-5, 5]},
                "features": {"mode": "full", "selection_enabled": False},
                "labeling": {"method": "triple_barrier", "upper_mult": 2.0},
            },
            "training": {
                "device": "cuda",
                "checkpoint": {"enabled": True},
                "optuna": {"n_trials": 3, "n_startup_trials": 5},
                "walk_forward": {"n_windows": 3, "gap_bars": 5},
            },
            "evaluation": {"compute_shap": True, "run_backtest": True},
            "bundling": {"bundle_format": "tar.gz"},
        }

        with caplog.at_level("WARNING", logger="src.config.experiment"):
            cfg = ExperimentConfig.from_dict(old)

        # Known values survive
        assert cfg.data.symbol == "MGC"
        assert cfg.data.features.selection_enabled is False
        assert cfg.data.labeling.upper_mult == 2.0
        assert cfg.training.optuna.n_trials == 3
        assert cfg.training.walk_forward.n_windows == 3
        assert cfg.evaluation.run_backtest is True
        # Every dropped key is named in a warning
        text = caplog.text
        for key in (
            "scaler",
            "mode",
            "method",
            "device",
            "checkpoint",
            "n_startup_trials",
            "gap_bars",
            "compute_shap",
            "bundle_format",
        ):
            assert key in text, f"no warning for dropped key {key!r}"

    def test_empty_section_uses_defaults(self):
        cfg = ExperimentConfig.from_dict({"run_id": "r", "data": {"mtf": None}})

        assert cfg.data.mtf.enabled is True


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


# =============================================================================
# 8. PHASE 116 WIRING
# =============================================================================


class TestPhase116Wiring:
    def test_split_ratios_reach_pipeline_config(self):
        cfg = _experiment_config()
        cfg.data.splits.train_ratio = 0.6
        cfg.data.splits.val_ratio = 0.25
        cfg.data.splits.test_ratio = 0.15

        pipeline = cfg.to_pipeline_config()

        assert (pipeline.train_ratio, pipeline.val_ratio, pipeline.test_ratio) == (
            0.6,
            0.25,
            0.15,
        )

    def test_calibration_settings_reach_pipeline_config(self):
        cfg = _experiment_config()
        cfg.training.calibration.enabled = False
        cfg.training.calibration.method = "isotonic"

        pipeline = cfg.to_pipeline_config()

        assert pipeline.auto_calibrate is False
        assert pipeline.calibration_method == "isotonic"

    def test_calibration_settings_reach_trainer_config(self, monkeypatch, tmp_path):
        """The Trainer self-calibrates from TrainerConfig — it must not fall back
        to global.yaml when the experiment disabled calibration."""
        config, _ = _run_train_model(
            monkeypatch,
            tmp_path,
            patience=None,
            use_calibration=False,
            calibration_method="sigmoid",
        )

        assert config.use_calibration is False
        assert config.calibration_method == "sigmoid"

    def test_experiment_verbose_is_factory_default(self, tmp_path):
        from src.factory import MLFactory

        cfg = ExperimentConfig(run_id="fixed_run", output_dir=str(tmp_path), verbose=0)

        assert MLFactory(cfg, enable_checkpoints=False).verbose == 0
        assert MLFactory(cfg, verbose=2, enable_checkpoints=False).verbose == 2
