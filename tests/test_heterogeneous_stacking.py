"""
Test Heterogeneous Stacking Ensemble.

Tests the ability to combine models from different families:
- Tabular (2D): XGBoost/LightGBM
- Sequence/Neural (3D): TCN, LSTM
- Transformer (3D): PatchTST

The key insight: Meta-learner always receives 2D OOF predictions
regardless of base model input shapes.
"""

import numpy as np
import pytest

from src.models import ModelRegistry
from src.models.ensemble.validator import (
    classify_base_models,
    is_heterogeneous_ensemble,
    validate_ensemble_config,
)


class TestHeterogeneousValidator:
    """Test heterogeneous ensemble validation logic."""

    def test_homogeneous_tabular_valid(self):
        """All tabular models should be valid."""
        is_valid, error = validate_ensemble_config(
            ["xgboost", "lightgbm"], ensemble_type="voting"
        )
        assert is_valid, error

    def test_homogeneous_sequence_valid(self):
        """All sequence models should be valid."""
        is_valid, error = validate_ensemble_config(
            ["tcn", "lstm"], ensemble_type="voting"
        )
        assert is_valid, error

    def test_heterogeneous_invalid_for_voting(self):
        """Mixed models should be invalid for voting."""
        is_valid, error = validate_ensemble_config(
            ["xgboost", "lstm"], ensemble_type="voting"
        )
        assert not is_valid
        assert "Compatibility Error" in error

    def test_heterogeneous_valid_for_stacking(self):
        """Mixed models should be VALID for stacking."""
        is_valid, error = validate_ensemble_config(
            ["xgboost", "lstm"], ensemble_type="stacking"
        )
        assert is_valid, f"Expected valid but got: {error}"

    def test_is_heterogeneous_detection(self):
        """Test heterogeneous detection."""
        assert not is_heterogeneous_ensemble(["xgboost", "lightgbm"])
        assert not is_heterogeneous_ensemble(["lstm", "tcn"])
        assert is_heterogeneous_ensemble(["xgboost", "lstm"])
        assert is_heterogeneous_ensemble(["lightgbm", "tcn", "patchtst"])

    def test_classify_base_models(self):
        """Test model classification."""
        tabular, sequence = classify_base_models(["xgboost", "lstm", "tcn", "lightgbm"])
        assert set(tabular) == {"xgboost", "lightgbm"}
        assert set(sequence) == {"lstm", "tcn"}


class TestHeterogeneousStackingCreation:
    """Test heterogeneous stacking ensemble creation."""

    def test_create_heterogeneous_stacking(self):
        """Verify heterogeneous stacking can be created."""
        config = {
            "base_model_names": ["xgboost", "tcn"],
            "meta_learner_name": "logistic",
            "n_folds": 2,
        }
        ensemble = ModelRegistry.create("stacking", config=config)
        assert ensemble is not None
        assert ensemble.ensemble_type == "stacking"

    def test_requires_sequences_property(self):
        """Heterogeneous ensemble should require sequences if any base model does."""
        # Homogeneous tabular - no sequences
        config_tabular = {
            "base_model_names": ["xgboost", "lightgbm"],
        }
        ensemble_tabular = ModelRegistry.create("stacking", config=config_tabular)
        assert not ensemble_tabular.requires_sequences

        # Heterogeneous - has sequence models
        config_hetero = {
            "base_model_names": ["xgboost", "tcn"],
        }
        ensemble_hetero = ModelRegistry.create("stacking", config=config_hetero)
        assert ensemble_hetero.requires_sequences


class TestHeterogeneousStackingTraining:
    """Test heterogeneous stacking training with synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n_samples = 60
        n_features = 10
        seq_len = 20

        # Tabular data (2D)
        X_train_2d = np.random.randn(n_samples, n_features).astype(np.float32)
        X_val_2d = np.random.randn(n_samples // 3, n_features).astype(np.float32)

        # Sequence data (3D) - [batch, seq_len, features]
        X_train_3d = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        X_val_3d = np.random.randn(n_samples // 3, seq_len, n_features).astype(np.float32)

        # Labels in correct format: -1, 0, 1 (Down, Hold, Up)
        y_train = np.random.choice([-1, 0, 1], n_samples)
        y_val = np.random.choice([-1, 0, 1], n_samples // 3)

        return {
            "X_train_2d": X_train_2d,
            "X_val_2d": X_val_2d,
            "X_train_3d": X_train_3d,
            "X_val_3d": X_val_3d,
            "y_train": y_train,
            "y_val": y_val,
        }

    def test_homogeneous_tabular_training(self, synthetic_data):
        """Test homogeneous tabular stacking works."""
        config = {
            "base_model_names": ["xgboost", "lightgbm"],
            "base_model_configs": {
                "xgboost": {"n_estimators": 5},
                "lightgbm": {"n_estimators": 5},
            },
            "meta_learner_name": "logistic",
            "n_folds": 2,
            "purge_bars": 0,  # Reduced for small test dataset
            "embargo_bars": 0,  # Reduced for small test dataset
        }
        ensemble = ModelRegistry.create("stacking", config=config)

        metrics = ensemble.fit(
            X_train=synthetic_data["X_train_2d"],
            y_train=synthetic_data["y_train"],
            X_val=synthetic_data["X_val_2d"],
            y_val=synthetic_data["y_val"],
        )

        assert metrics.val_f1 >= 0.0
        assert metrics.metadata.get("is_heterogeneous") is False

        # Test prediction
        output = ensemble.predict(synthetic_data["X_val_2d"])
        assert output.class_probabilities.shape == (len(synthetic_data["y_val"]), 3)

    @pytest.mark.slow
    def test_heterogeneous_stacking_training(self, synthetic_data):
        """Test heterogeneous stacking with tabular + sequence models."""
        config = {
            "base_model_names": ["xgboost", "tcn"],
            "base_model_configs": {
                "xgboost": {"n_estimators": 5},
                "tcn": {"n_epochs": 1, "hidden_dim": 16},
            },
            "meta_learner_name": "logistic",
            "n_folds": 2,
            "purge_bars": 0,  # Reduced for small test dataset
            "embargo_bars": 0,  # Reduced for small test dataset
            "use_default_configs_for_oof": False,  # Use fast configs for testing
        }
        ensemble = ModelRegistry.create("stacking", config=config)

        metrics = ensemble.fit(
            X_train=synthetic_data["X_train_2d"],
            y_train=synthetic_data["y_train"],
            X_val=synthetic_data["X_val_2d"],
            y_val=synthetic_data["y_val"],
            X_train_seq=synthetic_data["X_train_3d"],
            X_val_seq=synthetic_data["X_val_3d"],
        )

        assert metrics.val_f1 >= 0.0
        assert metrics.metadata.get("is_heterogeneous") is True
        assert "xgboost" in metrics.metadata.get("tabular_models", [])
        assert "tcn" in metrics.metadata.get("sequence_models", [])

        # Test prediction with both data types
        output = ensemble.predict(
            synthetic_data["X_val_2d"],
            X_seq=synthetic_data["X_val_3d"],
        )
        assert output.class_probabilities.shape == (len(synthetic_data["y_val"]), 3)
        assert output.metadata.get("is_heterogeneous") is True


if __name__ == "__main__":
    # Run quick validation tests
    print("=== Testing Heterogeneous Stacking Validation ===\n")

    test_val = TestHeterogeneousValidator()
    test_val.test_homogeneous_tabular_valid()
    print("✓ Homogeneous tabular validation")

    test_val.test_homogeneous_sequence_valid()
    print("✓ Homogeneous sequence validation")

    test_val.test_heterogeneous_invalid_for_voting()
    print("✓ Heterogeneous invalid for voting")

    test_val.test_heterogeneous_valid_for_stacking()
    print("✓ Heterogeneous VALID for stacking")

    test_val.test_is_heterogeneous_detection()
    print("✓ Heterogeneous detection")

    test_val.test_classify_base_models()
    print("✓ Base model classification")

    print("\n=== Testing Ensemble Creation ===\n")

    test_create = TestHeterogeneousStackingCreation()
    test_create.test_create_heterogeneous_stacking()
    print("✓ Heterogeneous stacking creation")

    test_create.test_requires_sequences_property()
    print("✓ requires_sequences property")

    print("\n=== All Validation Tests Passed ===")
