"""Tests for ML for Dummies smart configuration system."""

import pytest
from pathlib import Path

from src.config.smart_config import (
    MODEL_DEFAULTS,
    FEATURE_SETS,
    OPTIMIZATION_DEFAULTS,
    SmartConfig,
    ResolvedModelConfig,
    resolve_model_config,
    resolve_config,
    list_models,
    describe_model,
    show_defaults,
)


class TestModelDefaults:
    """Tests for MODEL_DEFAULTS dictionary."""

    def test_all_23_models_defined(self):
        """Verify all 23 models have defaults defined."""
        # Boosting (3)
        assert "xgboost" in MODEL_DEFAULTS
        assert "lightgbm" in MODEL_DEFAULTS
        assert "catboost" in MODEL_DEFAULTS

        # Classical (3)
        assert "random_forest" in MODEL_DEFAULTS
        assert "logistic" in MODEL_DEFAULTS
        assert "svm" in MODEL_DEFAULTS

        # Neural RNNs (2)
        assert "lstm" in MODEL_DEFAULTS
        assert "gru" in MODEL_DEFAULTS

        # Neural CNNs (3)
        assert "tcn" in MODEL_DEFAULTS
        assert "inceptiontime" in MODEL_DEFAULTS
        assert "resnet1d" in MODEL_DEFAULTS

        # Transformers (5)
        assert "transformer" in MODEL_DEFAULTS
        assert "patchtst" in MODEL_DEFAULTS
        assert "itransformer" in MODEL_DEFAULTS
        assert "tft" in MODEL_DEFAULTS
        assert "nbeats" in MODEL_DEFAULTS

        # Meta-learners (4)
        assert "ridge_meta" in MODEL_DEFAULTS
        assert "mlp_meta" in MODEL_DEFAULTS
        assert "calibrated_meta" in MODEL_DEFAULTS
        assert "xgboost_meta" in MODEL_DEFAULTS

        # Ensembles (3)
        assert "voting" in MODEL_DEFAULTS
        assert "stacking" in MODEL_DEFAULTS
        assert "blending" in MODEL_DEFAULTS

    def test_boosting_models_use_15min_timeframe(self):
        """Boosting models should default to 15min timeframe."""
        for model in ["xgboost", "lightgbm", "catboost"]:
            assert MODEL_DEFAULTS[model]["timeframe"] == "15min"
            assert MODEL_DEFAULTS[model]["family"] == "boosting"

    def test_sequence_models_use_5min_timeframe(self):
        """RNN/CNN models should default to 5min timeframe."""
        for model in ["lstm", "gru", "tcn"]:
            assert MODEL_DEFAULTS[model]["timeframe"] == "5min"
            assert MODEL_DEFAULTS[model]["sequence_length"] is not None

    def test_transformers_use_1min_timeframe(self):
        """Transformers should default to 1min timeframe for raw OHLCV."""
        for model in ["transformer", "patchtst", "itransformer"]:
            assert MODEL_DEFAULTS[model]["timeframe"] == "1min"
            assert MODEL_DEFAULTS[model]["features"] == "raw"

    def test_all_models_have_required_fields(self):
        """All models should have the required default fields."""
        required_fields = ["family", "timeframe", "features", "sequence_length", "batch_size", "description"]
        for model_name, defaults in MODEL_DEFAULTS.items():
            for field in required_fields:
                assert field in defaults, f"{model_name} missing field: {field}"

    def test_sequence_models_have_positive_sequence_length(self):
        """Models that need sequences should have positive sequence_length."""
        sequence_models = ["lstm", "gru", "tcn", "transformer", "patchtst", "itransformer", "tft", "nbeats", "inceptiontime", "resnet1d"]
        for model in sequence_models:
            seq_len = MODEL_DEFAULTS[model]["sequence_length"]
            assert seq_len is not None and seq_len > 0, f"{model} should have sequence_length > 0"

    def test_tabular_models_have_no_sequence_length(self):
        """Tabular models should have None for sequence_length."""
        tabular_models = ["xgboost", "lightgbm", "catboost", "random_forest", "logistic", "svm"]
        for model in tabular_models:
            assert MODEL_DEFAULTS[model]["sequence_length"] is None


class TestSmartConfig:
    """Tests for SmartConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SmartConfig()
        assert config.models == ["xgboost"]
        assert config.symbol == "MES"
        assert config.horizons == [20]
        assert config.optimize == "none"
        assert config.ensemble is False

    def test_single_model_string_converted_to_list(self):
        """Single model string should be converted to list."""
        config = SmartConfig(models="lstm")
        assert config.models == ["lstm"]

    def test_single_horizon_int_converted_to_list(self):
        """Single horizon int should be converted to list."""
        config = SmartConfig(horizons=15)
        assert config.horizons == [15]

    def test_multiple_models(self):
        """Multiple models should be accepted."""
        config = SmartConfig(models=["xgboost", "lstm", "patchtst"])
        assert len(config.models) == 3

    def test_invalid_model_raises_error(self):
        """Unknown model should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            SmartConfig(models=["xgboost", "invalid_model"])

    def test_path_string_converted_to_path(self):
        """String paths should be converted to Path objects."""
        config = SmartConfig(data_dir="/some/path", output_dir="/other/path")
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.output_dir, Path)

    def test_optimize_options(self):
        """All optimize options should be valid."""
        for opt in ["none", "features", "hyperparameters", "both"]:
            config = SmartConfig(optimize=opt)
            assert config.optimize == opt


class TestResolveModelConfig:
    """Tests for resolve_model_config function."""

    def test_xgboost_defaults(self):
        """Test XGBoost resolves with correct defaults."""
        resolved = resolve_model_config("xgboost")
        assert resolved.name == "xgboost"
        assert resolved.family == "boosting"
        assert resolved.timeframe == "15min"
        assert resolved.features == "full"
        assert resolved.sequence_length is None
        assert resolved.optimize_features is False
        assert resolved.optimize_hyperparams is False

    def test_lstm_defaults(self):
        """Test LSTM resolves with correct defaults."""
        resolved = resolve_model_config("lstm")
        assert resolved.name == "lstm"
        assert resolved.family == "neural"
        assert resolved.timeframe == "5min"
        assert resolved.features == "sequence"
        assert resolved.sequence_length == 60
        assert resolved.batch_size == 256

    def test_patchtst_defaults(self):
        """Test PatchTST resolves with correct defaults."""
        resolved = resolve_model_config("patchtst")
        assert resolved.name == "patchtst"
        assert resolved.family == "transformer"
        assert resolved.timeframe == "1min"
        assert resolved.features == "raw"
        assert resolved.sequence_length == 96
        assert resolved.batch_size == 64

    def test_optimize_features_flag(self):
        """Test optimize='features' sets correct flags."""
        resolved = resolve_model_config("xgboost", optimize="features")
        assert resolved.optimize_features is True
        assert resolved.optimize_hyperparams is False

    def test_optimize_hyperparameters_flag(self):
        """Test optimize='hyperparameters' sets correct flags."""
        resolved = resolve_model_config("xgboost", optimize="hyperparameters")
        assert resolved.optimize_features is False
        assert resolved.optimize_hyperparams is True

    def test_optimize_both_flag(self):
        """Test optimize='both' sets both flags."""
        resolved = resolve_model_config("xgboost", optimize="both")
        assert resolved.optimize_features is True
        assert resolved.optimize_hyperparams is True

    def test_override_sequence_length(self):
        """Test overriding sequence_length."""
        resolved = resolve_model_config("lstm", overrides={"sequence_length": 120})
        assert resolved.sequence_length == 120
        assert resolved.overrides_applied == {"sequence_length": 120}

    def test_override_timeframe(self):
        """Test overriding timeframe."""
        resolved = resolve_model_config("xgboost", overrides={"timeframe": "5min"})
        assert resolved.timeframe == "5min"

    def test_override_batch_size(self):
        """Test overriding batch_size."""
        resolved = resolve_model_config("lstm", overrides={"batch_size": 128})
        assert resolved.batch_size == 128

    def test_to_dict(self):
        """Test to_dict method returns correct format."""
        resolved = resolve_model_config("xgboost")
        d = resolved.to_dict()
        assert d["name"] == "xgboost"
        assert d["family"] == "boosting"
        assert "optimize_features" in d


class TestResolveConfig:
    """Tests for resolve_config function."""

    def test_single_model_resolution(self):
        """Test resolving config with single model."""
        config = SmartConfig(models=["xgboost"], symbol="MGC", horizons=[15])
        resolved = resolve_config(config)

        assert resolved["symbol"] == "MGC"
        assert resolved["horizons"] == [15]
        assert len(resolved["models"]) == 1
        assert resolved["models"][0].name == "xgboost"

    def test_multiple_model_resolution(self):
        """Test resolving config with multiple models."""
        config = SmartConfig(models=["xgboost", "lstm", "patchtst"])
        resolved = resolve_config(config)

        assert len(resolved["models"]) == 3
        names = [m.name for m in resolved["models"]]
        assert "xgboost" in names
        assert "lstm" in names
        assert "patchtst" in names

    def test_ensemble_settings_resolved(self):
        """Test ensemble settings are resolved correctly."""
        config = SmartConfig(
            models=["xgboost", "lstm"],
            ensemble=True,
            meta_learner="mlp_meta"
        )
        resolved = resolve_config(config)

        assert resolved["ensemble"]["enabled"] is True
        assert resolved["ensemble"]["meta_learner"] == "mlp_meta"

    def test_per_model_overrides(self):
        """Test per-model overrides are applied correctly."""
        config = SmartConfig(
            models=["xgboost", "lstm"],
            overrides={
                "xgboost": {"timeframe": "5min"},
                "lstm": {"sequence_length": 120},
            }
        )
        resolved = resolve_config(config)

        xgb = next(m for m in resolved["models"] if m.name == "xgboost")
        lstm = next(m for m in resolved["models"] if m.name == "lstm")

        assert xgb.timeframe == "5min"
        assert lstm.sequence_length == 120


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_list_models_all(self):
        """Test listing all models."""
        models = list_models()
        assert len(models) >= 23  # At least 23 models
        assert "xgboost" in models
        assert "lstm" in models
        assert "patchtst" in models

    def test_list_models_by_family(self):
        """Test listing models by family."""
        boosting = list_models("boosting")
        assert "xgboost" in boosting
        assert "lightgbm" in boosting
        assert "catboost" in boosting
        assert "lstm" not in boosting

        neural = list_models("neural")
        assert "lstm" in neural
        assert "gru" in neural
        assert "xgboost" not in neural

    def test_describe_model(self):
        """Test getting model description."""
        desc = describe_model("xgboost")
        assert "boosting" in desc.lower() or "gradient" in desc.lower()

        desc = describe_model("lstm")
        assert "lstm" in desc.lower() or "sequence" in desc.lower()

    def test_describe_model_unknown(self):
        """Test describe_model raises for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            describe_model("invalid_model")

    def test_show_defaults(self):
        """Test showing model defaults."""
        defaults = show_defaults("lstm")
        assert defaults["family"] == "neural"
        assert defaults["timeframe"] == "5min"
        assert defaults["sequence_length"] == 60

    def test_show_defaults_unknown(self):
        """Test show_defaults raises for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            show_defaults("invalid_model")


class TestFeatureSets:
    """Tests for feature set definitions."""

    def test_all_feature_sets_have_descriptions(self):
        """All feature sets should have descriptions."""
        expected_sets = ["full", "standard", "sequence", "minimal", "raw", "minimal_raw", "tft"]
        for feature_set in expected_sets:
            assert feature_set in FEATURE_SETS
            assert len(FEATURE_SETS[feature_set]) > 0


class TestOptimizationDefaults:
    """Tests for optimization default settings."""

    def test_features_optimization_defaults(self):
        """Test feature optimization defaults."""
        assert "features" in OPTIMIZATION_DEFAULTS
        assert "n_trials" in OPTIMIZATION_DEFAULTS["features"]
        assert OPTIMIZATION_DEFAULTS["features"]["n_trials"] > 0

    def test_hyperparameters_optimization_defaults(self):
        """Test hyperparameter optimization defaults."""
        assert "hyperparameters" in OPTIMIZATION_DEFAULTS
        assert "n_trials" in OPTIMIZATION_DEFAULTS["hyperparameters"]
        assert OPTIMIZATION_DEFAULTS["hyperparameters"]["n_trials"] > 0

    def test_both_optimization_defaults(self):
        """Test combined optimization defaults."""
        assert "both" in OPTIMIZATION_DEFAULTS
        assert "feature_trials" in OPTIMIZATION_DEFAULTS["both"]
        assert "hyperparam_trials" in OPTIMIZATION_DEFAULTS["both"]
