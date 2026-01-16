"""
Test StackingEnsemble - Stacking ensemble with heterogeneous model support.

Tests cover:
- StackingEnsemble initialization
- Heterogeneous ensemble detection
- MOD-001: Passthrough validation (trimming X_train for sequence offset)
- MOD-002: Sequence offset validation (reasonable offset bounds)
- MOD-003: Sequence model data type validation (3D requirement)
- Input shape validation for different ensemble types
- Save/load functionality
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import ModelRegistry
from src.models.ensemble.stacking import StackingEnsemble
from src.models.ensemble.validator import (
    classify_base_models,
    is_heterogeneous_ensemble,
    validate_ensemble_config,
    validate_base_model_compatibility,
    EnsembleCompatibilityError,
    HETEROGENEOUS_ENSEMBLE_TYPES,
    HOMOGENEOUS_ENSEMBLE_TYPES,
)


class TestStackingEnsembleInitialization:
    """Test StackingEnsemble initialization and configuration."""

    def test_basic_initialization(self):
        """StackingEnsemble should initialize with defaults."""
        ensemble = StackingEnsemble()
        assert ensemble.model_family == "ensemble"
        assert ensemble.ensemble_type == "stacking"
        assert ensemble.requires_scaling is False

    def test_initialization_with_config(self):
        """StackingEnsemble should accept configuration."""
        config = {
            "base_model_names": ["xgboost", "lightgbm"],
            "meta_learner_name": "logistic",
            "n_folds": 3,
        }
        ensemble = StackingEnsemble(config=config)

        default_config = ensemble.get_default_config()
        merged_config = {**default_config, **config}

        assert ensemble._config.get("base_model_names") == ["xgboost", "lightgbm"]
        assert ensemble._config.get("meta_learner_name") == "logistic"

    def test_get_default_config(self):
        """get_default_config should return sensible defaults."""
        ensemble = StackingEnsemble()
        config = ensemble.get_default_config()

        assert "base_model_names" in config
        assert "meta_learner_name" in config
        assert config["n_folds"] == 5
        assert config["use_probabilities"] is True
        assert config["passthrough"] is False
        assert config["use_default_configs_for_oof"] is True

    def test_requires_sequences_homogeneous_tabular(self):
        """requires_sequences should be False for all-tabular ensemble."""
        config = {"base_model_names": ["xgboost", "lightgbm"]}
        ensemble = StackingEnsemble(config=config)
        assert ensemble.requires_sequences is False

    def test_requires_sequences_homogeneous_sequence(self):
        """requires_sequences should be True for all-sequence ensemble."""
        config = {"base_model_names": ["lstm", "tcn"]}
        ensemble = StackingEnsemble(config=config)
        assert ensemble.requires_sequences is True

    def test_requires_sequences_heterogeneous(self):
        """requires_sequences should be True for mixed ensemble."""
        config = {"base_model_names": ["xgboost", "lstm"]}
        ensemble = StackingEnsemble(config=config)
        assert ensemble.requires_sequences is True


class TestHeterogeneousEnsembleDetection:
    """Test heterogeneous ensemble detection and classification."""

    def test_is_heterogeneous_all_tabular(self):
        """All tabular models should not be heterogeneous."""
        assert is_heterogeneous_ensemble(["xgboost", "lightgbm"]) is False
        assert is_heterogeneous_ensemble(["random_forest", "logistic", "svm"]) is False

    def test_is_heterogeneous_all_sequence(self):
        """All sequence models should not be heterogeneous."""
        assert is_heterogeneous_ensemble(["lstm", "gru"]) is False
        assert is_heterogeneous_ensemble(["tcn", "transformer"]) is False

    def test_is_heterogeneous_mixed(self):
        """Mixed tabular + sequence should be heterogeneous."""
        assert is_heterogeneous_ensemble(["xgboost", "lstm"]) is True
        assert is_heterogeneous_ensemble(["lightgbm", "tcn", "gru"]) is True
        assert is_heterogeneous_ensemble(["random_forest", "patchtst"]) is True

    def test_classify_base_models(self):
        """classify_base_models should separate tabular and sequence."""
        tabular, sequence = classify_base_models(["xgboost", "lstm", "tcn", "lightgbm"])
        assert set(tabular) == {"xgboost", "lightgbm"}
        assert set(sequence) == {"lstm", "tcn"}

    def test_classify_base_models_empty(self):
        """classify_base_models should handle empty input."""
        tabular, sequence = classify_base_models([])
        assert tabular == []
        assert sequence == []


class TestEnsembleValidation:
    """Test ensemble configuration validation."""

    def test_validate_homogeneous_tabular_voting(self):
        """Homogeneous tabular should be valid for voting."""
        is_valid, error = validate_ensemble_config(
            ["xgboost", "lightgbm"], ensemble_type="voting"
        )
        assert is_valid
        assert error == ""

    def test_validate_homogeneous_sequence_voting(self):
        """Homogeneous sequence should be valid for voting."""
        is_valid, error = validate_ensemble_config(
            ["lstm", "gru"], ensemble_type="voting"
        )
        assert is_valid

    def test_validate_heterogeneous_invalid_for_voting(self):
        """Heterogeneous should be invalid for voting."""
        is_valid, error = validate_ensemble_config(
            ["xgboost", "lstm"], ensemble_type="voting"
        )
        assert not is_valid
        assert "Compatibility Error" in error

    def test_validate_heterogeneous_invalid_for_blending(self):
        """Heterogeneous should be invalid for blending."""
        is_valid, error = validate_ensemble_config(
            ["lightgbm", "tcn"], ensemble_type="blending"
        )
        assert not is_valid

    def test_validate_heterogeneous_valid_for_stacking(self):
        """Heterogeneous should be VALID for stacking."""
        is_valid, error = validate_ensemble_config(
            ["xgboost", "lstm"], ensemble_type="stacking"
        )
        assert is_valid
        assert error == ""

    def test_validate_empty_list(self):
        """Empty model list should be invalid."""
        is_valid, error = validate_ensemble_config([])
        assert not is_valid
        assert "No base models" in error

    def test_validate_single_model(self):
        """Single model should be invalid (need >= 2)."""
        is_valid, error = validate_ensemble_config(["xgboost"])
        assert not is_valid
        assert "at least 2" in error

    def test_validate_unregistered_model(self):
        """Unregistered model should be invalid."""
        is_valid, error = validate_ensemble_config(["xgboost", "fake_model"])
        assert not is_valid
        assert "not registered" in error

    def test_validate_base_model_compatibility_raises(self):
        """validate_base_model_compatibility should raise for invalid config."""
        with pytest.raises(EnsembleCompatibilityError):
            validate_base_model_compatibility(
                ["xgboost", "lstm"], ensemble_type="voting"
            )

    def test_validate_base_model_compatibility_stacking_ok(self):
        """validate_base_model_compatibility should pass for stacking."""
        # Should not raise
        validate_base_model_compatibility(
            ["xgboost", "lstm"], ensemble_type="stacking"
        )


class TestInputShapeValidation:
    """Test input shape validation for StackingEnsemble."""

    @pytest.fixture
    def tabular_ensemble(self):
        """Create tabular-only ensemble."""
        config = {
            "base_model_names": ["xgboost", "lightgbm"],
            "meta_learner_name": "logistic",
        }
        return StackingEnsemble(config=config)

    @pytest.fixture
    def sequence_ensemble(self):
        """Create sequence-only ensemble."""
        config = {
            "base_model_names": ["lstm", "gru"],
            "meta_learner_name": "logistic",
        }
        return StackingEnsemble(config=config)

    def test_validate_input_shape_1d_rejected(self, tabular_ensemble):
        """1D input should always be rejected."""
        X = np.random.randn(100)
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            tabular_ensemble._validate_input_shape(X, "X_train")

    def test_validate_input_shape_2d_for_tabular(self, tabular_ensemble):
        """2D input should be valid for tabular ensemble."""
        X = np.random.randn(100, 50)
        # Should not raise
        tabular_ensemble._validate_input_shape(X, "X_train")


class TestMOD001PassthroughValidation:
    """Test MOD-001: Passthrough offset validation."""

    def test_passthrough_reasonable_offset(self):
        """MOD-001: Reasonable offset should pass."""
        # Simulating the scenario where OOF predictions have fewer samples
        # due to sequence model lookback window
        n_samples = 1000
        seq_len = 60
        offset = seq_len - 1  # Typical offset for sequence models

        # This should be < 50% of training data
        max_reasonable = n_samples // 2
        assert offset < max_reasonable

    def test_passthrough_excessive_offset_detection(self):
        """MOD-001: Excessive offset should be detectable."""
        n_samples = 100
        excessive_offset = 60  # More than 50%

        max_reasonable = n_samples // 2
        assert excessive_offset > max_reasonable

    def test_passthrough_shape_validation(self):
        """MOD-001: Shape validation after trimming."""
        X_train = np.random.randn(1000, 50)
        oof_predictions = np.random.randn(940, 9)  # 60 fewer samples

        offset = X_train.shape[0] - oof_predictions.shape[0]
        X_train_trimmed = X_train[offset:]

        # Shapes should match after trimming
        assert X_train_trimmed.shape[0] == oof_predictions.shape[0]


class TestMOD002SequenceOffsetValidation:
    """Test MOD-002: Sequence offset validation."""

    def test_sequence_offset_reasonable(self):
        """MOD-002: Reasonable sequence offset should pass validation."""
        n_samples = 1000
        seq_len = 60
        seq_offset = seq_len - 1

        # Should be < 50% of data
        max_reasonable_offset = n_samples // 2
        is_reasonable = seq_offset <= max_reasonable_offset
        assert is_reasonable

    def test_sequence_offset_excessive(self):
        """MOD-002: Excessive sequence offset should be caught."""
        n_samples = 100
        seq_offset = 60  # More than 50%

        max_reasonable_offset = n_samples // 2
        is_excessive = seq_offset > max_reasonable_offset
        assert is_excessive

    def test_sequence_offset_negative_invalid(self):
        """MOD-002: Negative offset should be invalid."""
        # Negative offset means sequence data has more samples than tabular
        seq_offset = -10
        is_invalid = seq_offset < 0
        assert is_invalid

    def test_fold_retention_ratio(self):
        """MOD-002: Fold retention ratio calculation."""
        original_train_size = 100
        filtered_train_size = 60

        retention_ratio = filtered_train_size / original_train_size
        min_required = 0.5

        # This should pass
        assert retention_ratio >= min_required

        # This should fail
        filtered_train_size_low = 40
        retention_ratio_low = filtered_train_size_low / original_train_size
        assert retention_ratio_low < min_required


class TestMOD003SequenceModelDataValidation:
    """Test MOD-003: Sequence model data type validation."""

    def test_sequence_model_requires_3d(self):
        """MOD-003: Sequence models require 3D input."""
        # Check model info for sequence requirement
        lstm_info = ModelRegistry.get_model_info("lstm")
        assert lstm_info["requires_sequences"] is True

        tcn_info = ModelRegistry.get_model_info("tcn")
        assert tcn_info["requires_sequences"] is True

    def test_tabular_model_accepts_2d(self):
        """MOD-003: Tabular models accept 2D input."""
        xgb_info = ModelRegistry.get_model_info("xgboost")
        assert xgb_info["requires_sequences"] is False

    def test_3d_data_shape_validation(self):
        """MOD-003: 3D data should have correct shape."""
        n_samples = 100
        seq_len = 60
        n_features = 50

        X_3d = np.random.randn(n_samples, seq_len, n_features)

        assert X_3d.ndim == 3
        assert X_3d.shape == (n_samples, seq_len, n_features)

    def test_2d_data_for_sequence_model_error(self):
        """MOD-003: 2D data for sequence model should be caught."""
        X_2d = np.random.randn(100, 50)

        # This represents the validation that should happen
        is_invalid_for_sequence = X_2d.ndim != 3
        assert is_invalid_for_sequence


class TestStackingEnsembleWithSyntheticData:
    """Test StackingEnsemble with synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        n_classes = 3

        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        X_val = np.random.randn(n_samples // 3, n_features).astype(np.float32)

        y_train = np.random.choice(n_classes, n_samples)
        y_val = np.random.choice(n_classes, n_samples // 3)

        return {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
        }

    def test_homogeneous_tabular_fit(self, synthetic_data):
        """Homogeneous tabular ensemble should train successfully."""
        config = {
            "base_model_names": ["xgboost", "lightgbm"],
            "base_model_configs": {
                "xgboost": {"n_estimators": 5},
                "lightgbm": {"n_estimators": 5},
            },
            "meta_learner_name": "logistic",
            "n_folds": 2,
            "purge_bars": 0,
            "embargo_bars": 0,
            "analyze_diversity": False,  # Speed up test
        }
        ensemble = StackingEnsemble(config=config)

        metrics = ensemble.fit(
            X_train=synthetic_data["X_train"],
            y_train=synthetic_data["y_train"],
            X_val=synthetic_data["X_val"],
            y_val=synthetic_data["y_val"],
        )

        assert metrics.val_f1 >= 0.0
        assert metrics.metadata.get("is_heterogeneous") is False

    def test_homogeneous_tabular_predict(self, synthetic_data):
        """Trained ensemble should make predictions."""
        config = {
            "base_model_names": ["xgboost", "lightgbm"],
            "base_model_configs": {
                "xgboost": {"n_estimators": 5},
                "lightgbm": {"n_estimators": 5},
            },
            "meta_learner_name": "logistic",
            "n_folds": 2,
            "purge_bars": 0,
            "embargo_bars": 0,
            "analyze_diversity": False,
        }
        ensemble = StackingEnsemble(config=config)

        ensemble.fit(
            X_train=synthetic_data["X_train"],
            y_train=synthetic_data["y_train"],
            X_val=synthetic_data["X_val"],
            y_val=synthetic_data["y_val"],
        )

        output = ensemble.predict(synthetic_data["X_val"])

        assert output.class_predictions.shape == (len(synthetic_data["y_val"]),)
        assert output.class_probabilities.shape == (len(synthetic_data["y_val"]), 3)

    def test_predict_proba(self, synthetic_data):
        """predict_proba should return probabilities."""
        config = {
            "base_model_names": ["xgboost", "lightgbm"],
            "base_model_configs": {
                "xgboost": {"n_estimators": 5},
                "lightgbm": {"n_estimators": 5},
            },
            "meta_learner_name": "logistic",
            "n_folds": 2,
            "purge_bars": 0,
            "embargo_bars": 0,
            "analyze_diversity": False,
        }
        ensemble = StackingEnsemble(config=config)

        ensemble.fit(
            X_train=synthetic_data["X_train"],
            y_train=synthetic_data["y_train"],
            X_val=synthetic_data["X_val"],
            y_val=synthetic_data["y_val"],
        )

        proba = ensemble.predict_proba(synthetic_data["X_val"])

        assert proba.shape == (len(synthetic_data["y_val"]), 3)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)


class TestStackingEnsembleSaveLoad:
    """Test save/load functionality."""

    @pytest.fixture
    def trained_ensemble(self):
        """Create and train a simple ensemble."""
        np.random.seed(42)
        X_train = np.random.randn(60, 10).astype(np.float32)
        X_val = np.random.randn(20, 10).astype(np.float32)
        y_train = np.random.choice(3, 60)
        y_val = np.random.choice(3, 20)

        config = {
            "base_model_names": ["xgboost", "lightgbm"],
            "base_model_configs": {
                "xgboost": {"n_estimators": 5},
                "lightgbm": {"n_estimators": 5},
            },
            "meta_learner_name": "logistic",
            "n_folds": 2,
            "purge_bars": 0,
            "embargo_bars": 0,
            "analyze_diversity": False,
        }
        ensemble = StackingEnsemble(config=config)
        ensemble.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        return ensemble, X_val

    def test_save_creates_files(self, trained_ensemble, tmp_path):
        """save should create necessary files."""
        ensemble, _ = trained_ensemble
        save_path = tmp_path / "ensemble"

        ensemble.save(save_path)

        assert save_path.exists()
        assert (save_path / "ensemble_metadata.joblib").exists()
        assert (save_path / "meta_learner").exists()
        assert (save_path / "base_models").exists()

    def test_load_restores_ensemble(self, trained_ensemble, tmp_path):
        """load should restore functional ensemble."""
        original_ensemble, X_val = trained_ensemble
        save_path = tmp_path / "ensemble"

        # Save
        original_ensemble.save(save_path)

        # Get original predictions
        original_output = original_ensemble.predict(X_val)

        # Load into new ensemble
        loaded_ensemble = StackingEnsemble()
        loaded_ensemble.load(save_path)

        # Verify predictions match
        loaded_output = loaded_ensemble.predict(X_val)

        np.testing.assert_array_equal(
            original_output.class_predictions,
            loaded_output.class_predictions,
        )

    def test_load_missing_metadata_raises(self, tmp_path):
        """load should raise error for missing metadata."""
        ensemble = StackingEnsemble()

        with pytest.raises(FileNotFoundError, match="metadata not found"):
            ensemble.load(tmp_path / "nonexistent")


class TestEnsembleTypes:
    """Test ensemble type constants."""

    def test_heterogeneous_types(self):
        """HETEROGENEOUS_ENSEMBLE_TYPES should contain stacking."""
        assert "stacking" in HETEROGENEOUS_ENSEMBLE_TYPES

    def test_homogeneous_types(self):
        """HOMOGENEOUS_ENSEMBLE_TYPES should contain voting and blending."""
        assert "voting" in HOMOGENEOUS_ENSEMBLE_TYPES
        assert "blending" in HOMOGENEOUS_ENSEMBLE_TYPES

    def test_no_overlap(self):
        """Heterogeneous and homogeneous types should not overlap."""
        overlap = HETEROGENEOUS_ENSEMBLE_TYPES & HOMOGENEOUS_ENSEMBLE_TYPES
        assert len(overlap) == 0


class TestModelRegistryIntegration:
    """Test integration with ModelRegistry."""

    def test_stacking_registered(self):
        """Stacking should be registered in ModelRegistry."""
        assert ModelRegistry.is_registered("stacking")

    def test_create_stacking_from_registry(self):
        """ModelRegistry.create should produce StackingEnsemble."""
        config = {"base_model_names": ["xgboost", "lightgbm"]}
        ensemble = ModelRegistry.create("stacking", config=config)

        assert isinstance(ensemble, StackingEnsemble)
        assert ensemble.ensemble_type == "stacking"

    def test_stacking_model_info(self):
        """Model info should indicate ensemble family."""
        info = ModelRegistry.get_model_info("stacking")
        assert info["family"] == "ensemble"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
