"""
Tests for the model configuration system.

Tests YAML loading, flattening, merging, validation, and environment detection.
"""
import pytest
from pathlib import Path
from typing import Any, Dict

from src.models.config import (
    # Classes
    TrainerConfig,
    Environment,
    ConfigValidationError,
    # Paths
    CONFIG_ROOT,
    CONFIG_DIR,
    TRAINING_CONFIG_PATH,
    CV_CONFIG_PATH,
    # Environment detection
    detect_environment,
    resolve_device,
    # Loading
    load_yaml_config,
    load_model_config,
    find_model_config,
    load_training_config,
    load_cv_config,
    flatten_model_config,
    # Building
    merge_configs,
    build_config,
    create_trainer_config,
    # Validation
    validate_model_config_structure,
    validate_config,
    validate_config_strict,
    # Utilities
    list_available_models,
    get_model_info,
)


# =============================================================================
# PATH TESTS
# =============================================================================

class TestConfigPaths:
    """Test configuration path constants."""

    def test_config_root_exists(self) -> None:
        """Config root directory should exist."""
        assert CONFIG_ROOT.exists()
        assert CONFIG_ROOT.is_dir()

    def test_config_dir_exists(self) -> None:
        """Models config directory should exist."""
        assert CONFIG_DIR.exists()
        assert CONFIG_DIR.is_dir()

    def test_training_config_exists(self) -> None:
        """Global training config should exist."""
        assert TRAINING_CONFIG_PATH.exists()
        assert TRAINING_CONFIG_PATH.is_file()

    def test_cv_config_exists(self) -> None:
        """Cross-validation config should exist."""
        assert CV_CONFIG_PATH.exists()
        assert CV_CONFIG_PATH.is_file()


# =============================================================================
# MODEL CONFIG LOADING TESTS
# =============================================================================

class TestModelConfigLoading:
    """Test loading of model configuration files."""

    @pytest.fixture
    def expected_models(self) -> list:
        """List of expected model config files."""
        return [
            "xgboost", "lightgbm", "catboost",
            "lstm", "gru", "tcn", "transformer",
            "random_forest", "logistic", "svm",
            "voting", "stacking", "blending",
        ]

    def test_list_available_models(self, expected_models: list) -> None:
        """Should list all available model configs."""
        models = list_available_models()
        for model in expected_models:
            assert model in models, f"Missing model config: {model}"

    def test_find_model_config_existing(self) -> None:
        """Should find existing model config paths."""
        path = find_model_config("xgboost")
        assert path is not None
        assert path.exists()
        assert path.name == "xgboost.yaml"

    def test_find_model_config_missing(self) -> None:
        """Should return None for missing model configs."""
        path = find_model_config("nonexistent_model")
        assert path is None

    def test_load_model_config_flattened(self) -> None:
        """Should load and flatten model config."""
        config = load_model_config("xgboost", flatten=True)

        # Check flattened keys exist
        assert "model_name" in config
        assert "model_family" in config
        assert "n_estimators" in config
        assert "device" in config

        # Check values
        assert config["model_name"] == "xgboost"
        assert config["model_family"] == "boosting"

    def test_load_model_config_raw(self) -> None:
        """Should load raw model config without flattening."""
        config = load_model_config("xgboost", flatten=False)

        # Check section structure
        assert "model" in config
        assert "defaults" in config
        assert "training" in config
        assert "device" in config

        # Check nested values
        assert config["model"]["name"] == "xgboost"
        assert config["model"]["family"] == "boosting"

    def test_load_all_model_configs(self, expected_models: list) -> None:
        """Should load all expected model configs without errors."""
        for model in expected_models:
            config = load_model_config(model)
            assert "model_name" in config or "model_family" in config


# =============================================================================
# CONFIG FLATTENING TESTS
# =============================================================================

class TestConfigFlattening:
    """Test the flatten_model_config function."""

    def test_flatten_complete_config(self) -> None:
        """Should flatten all sections correctly."""
        nested = {
            "model": {
                "name": "test_model",
                "family": "test_family",
                "description": "Test description",
            },
            "defaults": {
                "param1": 100,
                "param2": 0.5,
            },
            "training": {
                "batch_size": 256,
                "feature_set": "test_set",
            },
            "device": {
                "default": "auto",
                "mixed_precision": True,
            },
        }

        flat = flatten_model_config(nested)

        assert flat["model_name"] == "test_model"
        assert flat["model_family"] == "test_family"
        assert flat["model_description"] == "Test description"
        assert flat["param1"] == 100
        assert flat["param2"] == 0.5
        assert flat["batch_size"] == 256
        assert flat["feature_set"] == "test_set"
        assert flat["device"] == "auto"
        assert flat["mixed_precision"] is True

    def test_flatten_partial_config(self) -> None:
        """Should handle configs with missing sections."""
        nested = {
            "model": {"name": "partial", "family": "test"},
            "defaults": {"param": 42},
        }

        flat = flatten_model_config(nested)

        assert flat["model_name"] == "partial"
        assert flat["param"] == 42
        assert "device" not in flat  # No device section


# =============================================================================
# CONFIG MERGING TESTS
# =============================================================================

class TestConfigMerging:
    """Test configuration merging."""

    def test_merge_simple_override(self) -> None:
        """Override values should take precedence."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = merge_configs(base, override)

        assert result["a"] == 1
        assert result["b"] == 3  # Overridden
        assert result["c"] == 4  # Added

    def test_merge_deep_dicts(self) -> None:
        """Should merge nested dictionaries."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}

        result = merge_configs(base, override, deep=True)

        assert result["outer"]["a"] == 1
        assert result["outer"]["b"] == 3
        assert result["outer"]["c"] == 4

    def test_merge_shallow(self) -> None:
        """Shallow merge should replace entire dicts."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"c": 3}}

        result = merge_configs(base, override, deep=False)

        assert result["outer"] == {"c": 3}


# =============================================================================
# CONFIG BUILDING TESTS
# =============================================================================

class TestBuildConfig:
    """Test the build_config function."""

    def test_build_from_model_yaml(self) -> None:
        """Should build config from model YAML file."""
        config = build_config("xgboost")

        assert config["model_name"] == "xgboost"
        assert config["model_family"] == "boosting"
        assert "n_estimators" in config

    def test_build_with_cli_override(self) -> None:
        """CLI args should override YAML values."""
        cli_args = {"n_estimators": 1000, "learning_rate": 0.01}

        config = build_config("xgboost", cli_args=cli_args)

        assert config["n_estimators"] == 1000
        assert config["learning_rate"] == 0.01

    def test_build_ignores_none_cli_args(self) -> None:
        """CLI args with None values should be ignored."""
        cli_args = {"n_estimators": None, "learning_rate": 0.01}

        config = build_config("xgboost", cli_args=cli_args)

        # n_estimators should be from YAML, not None
        assert config["n_estimators"] == 500
        assert config["learning_rate"] == 0.01


# =============================================================================
# TRAINER CONFIG TESTS
# =============================================================================

class TestTrainerConfig:
    """Test TrainerConfig dataclass."""

    def test_create_basic(self) -> None:
        """Should create TrainerConfig with model name."""
        config = TrainerConfig(model_name="xgboost")

        assert config.model_name == "xgboost"
        assert config.horizon == 20  # Default
        assert config.device == "auto"  # Default

    def test_validate_positive_horizon(self) -> None:
        """Should reject non-positive horizon."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            TrainerConfig(model_name="test", horizon=0)

    def test_validate_positive_batch_size(self) -> None:
        """Should reject non-positive batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainerConfig(model_name="test", batch_size=-1)

    def test_to_dict(self) -> None:
        """Should serialize to dictionary."""
        config = TrainerConfig(model_name="test", horizon=10)
        data = config.to_dict()

        assert data["model_name"] == "test"
        assert data["horizon"] == 10
        assert isinstance(data["output_dir"], str)

    def test_from_dict(self) -> None:
        """Should deserialize from dictionary."""
        data = {
            "model_name": "test",
            "horizon": 15,
            "batch_size": 128,
        }

        config = TrainerConfig.from_dict(data)

        assert config.model_name == "test"
        assert config.horizon == 15
        assert config.batch_size == 128

    def test_create_trainer_config(self) -> None:
        """Should create TrainerConfig from model config."""
        config = create_trainer_config("xgboost", horizon=10)

        assert config.model_name == "xgboost"
        assert config.horizon == 10
        assert isinstance(config.model_config, dict)


# =============================================================================
# CONFIG VALIDATION TESTS
# =============================================================================

class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_valid_config(self) -> None:
        """Should pass for valid config."""
        config = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "device": "auto",
        }

        errors = validate_config(config, "test")
        assert len(errors) == 0

    def test_validate_out_of_range(self) -> None:
        """Should fail for out-of-range values."""
        config = {
            "n_estimators": 50000,  # > 10000
            "dropout": 1.5,  # > 1.0
        }

        errors = validate_config(config, "test")
        assert len(errors) == 2

    def test_validate_invalid_device(self) -> None:
        """Should fail for invalid device."""
        config = {"device": "tpu"}

        errors = validate_config(config, "test")
        assert len(errors) == 1
        assert "device" in errors[0].lower()

    def test_validate_structure_valid(self) -> None:
        """Should validate structured config."""
        config = {
            "model": {"name": "test", "family": "boosting"},
            "defaults": {},
            "device": {"default": "auto"},
        }

        errors = validate_model_config_structure(config)
        assert len(errors) == 0

    def test_validate_structure_missing_model(self) -> None:
        """Should fail for missing model section."""
        config = {"defaults": {}}

        errors = validate_model_config_structure(config)
        assert len(errors) >= 1
        assert any("model" in e for e in errors)

    def test_validate_strict_raises(self) -> None:
        """Should raise ConfigValidationError on strict validation."""
        config = {"n_estimators": -1}

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_strict(config, "test", raise_on_error=True)

        assert len(exc_info.value.errors) >= 1


# =============================================================================
# ENVIRONMENT DETECTION TESTS
# =============================================================================

class TestEnvironmentDetection:
    """Test environment detection utilities."""

    def test_detect_environment(self) -> None:
        """Should detect environment."""
        env = detect_environment()
        assert isinstance(env, Environment)

    def test_resolve_device_auto(self) -> None:
        """Should resolve 'auto' to actual device."""
        device = resolve_device("auto")
        assert device in ["cuda", "cpu"]

    def test_resolve_device_explicit(self) -> None:
        """Should pass through explicit device."""
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"


# =============================================================================
# GLOBAL CONFIG TESTS
# =============================================================================

class TestGlobalConfigs:
    """Test global configuration loading."""

    def test_load_training_config(self) -> None:
        """Should load training config."""
        config = load_training_config()

        assert "data" in config or "training" in config or "device" in config

    def test_load_cv_config(self) -> None:
        """Should load CV config."""
        config = load_cv_config()

        assert len(config) > 0


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestConfigUtilities:
    """Test configuration utility functions."""

    def test_get_model_info(self) -> None:
        """Should get model info from config."""
        info = get_model_info("xgboost")

        assert info["name"] == "xgboost"
        assert info["family"] == "boosting"
        assert len(info["description"]) > 0

    def test_get_model_info_missing(self) -> None:
        """Should handle missing model gracefully."""
        info = get_model_info("nonexistent")

        assert info["family"] == "unknown"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestConfigIntegration:
    """Integration tests for the configuration system."""

    def test_full_workflow_xgboost(self) -> None:
        """Test complete workflow for XGBoost config."""
        # Load config
        config = load_model_config("xgboost")
        assert config["model_name"] == "xgboost"

        # Validate
        errors = validate_config(config, "xgboost")
        assert len(errors) == 0

        # Create trainer config
        trainer = create_trainer_config("xgboost", horizon=10)
        assert trainer.model_name == "xgboost"
        assert trainer.horizon == 10

    def test_full_workflow_ensemble(self) -> None:
        """Test complete workflow for ensemble config."""
        # Load config
        config = load_model_config("voting")
        assert config["model_family"] == "ensemble"

        # Check ensemble-specific fields
        assert "base_model_names" in config or "voting" in config

        # Create trainer config
        trainer = create_trainer_config("voting", horizon=15)
        assert trainer.model_name == "voting"
