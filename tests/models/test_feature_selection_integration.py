"""
Tests for Feature Selection Integration with Model Training.

Tests:
- FeatureSelectionConfig validation and model family defaults
- FeatureSelectionManager operations
- PersistedFeatureSelection serialization
- Integration with Trainer pipeline
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.feature_selection import (
    FeatureSelectionConfig,
    FeatureSelectionManager,
    ModelFamilyDefaults,
    PersistedFeatureSelection,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_feature_data():
    """Generate sample feature data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 30

    # Create DataFrame with named columns
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names,
    )
    y = pd.Series(np.random.choice([-1, 0, 1], size=n_samples))
    weights = pd.Series(np.random.uniform(0.5, 1.5, size=n_samples))

    return {
        "X": X,
        "y": y,
        "weights": weights,
        "feature_names": feature_names,
    }


@pytest.fixture
def persisted_result(sample_feature_data):
    """Create a sample PersistedFeatureSelection."""
    selected = sample_feature_data["feature_names"][:10]
    return PersistedFeatureSelection(
        selected_features=selected,
        feature_indices={f: i for i, f in enumerate(selected)},
        selection_method="mda",
        n_features_original=30,
        n_features_selected=10,
        stability_scores={f: 0.8 for f in selected},
        importance_scores={f: 0.1 for f in selected},
        metadata={"test": True},
    )


# =============================================================================
# MODEL FAMILY DEFAULTS TESTS
# =============================================================================

class TestModelFamilyDefaults:
    """Tests for ModelFamilyDefaults configuration."""

    def test_get_defaults_boosting(self):
        """Test boosting family defaults."""
        defaults = ModelFamilyDefaults.get_defaults("boosting")
        assert defaults["enabled"] is True
        assert defaults["n_features"] == 50
        assert defaults["method"] == "mda"

    def test_get_defaults_classical(self):
        """Test classical family defaults."""
        defaults = ModelFamilyDefaults.get_defaults("classical")
        assert defaults["enabled"] is True
        assert defaults["n_features"] == 40

    def test_get_defaults_neural(self):
        """Test neural family defaults - should be disabled."""
        defaults = ModelFamilyDefaults.get_defaults("neural")
        assert defaults["enabled"] is False

    def test_get_defaults_sequence_alias(self):
        """Test sequence alias maps to neural defaults."""
        defaults = ModelFamilyDefaults.get_defaults("sequence")
        assert defaults["enabled"] is False

    def test_get_defaults_unknown_family(self):
        """Test unknown family raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model family"):
            ModelFamilyDefaults.get_defaults("unknown_family")

    def test_is_enabled_by_default(self):
        """Test is_enabled_by_default for different families."""
        assert ModelFamilyDefaults.is_enabled_by_default("boosting") is True
        assert ModelFamilyDefaults.is_enabled_by_default("classical") is True
        assert ModelFamilyDefaults.is_enabled_by_default("neural") is False


# =============================================================================
# FEATURE SELECTION CONFIG TESTS
# =============================================================================

class TestFeatureSelectionConfig:
    """Tests for FeatureSelectionConfig."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = FeatureSelectionConfig(
            enabled=True,
            n_features=50,
            method="mda",
        )
        assert config.enabled is True
        assert config.n_features == 50
        assert config.method == "mda"

    def test_invalid_method(self):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            FeatureSelectionConfig(method="invalid")

    def test_invalid_n_features(self):
        """Test negative n_features raises ValueError."""
        with pytest.raises(ValueError, match="n_features must be >= 0"):
            FeatureSelectionConfig(n_features=-5)

    def test_invalid_min_frequency(self):
        """Test invalid min_feature_frequency raises ValueError."""
        with pytest.raises(ValueError, match="min_feature_frequency must be in"):
            FeatureSelectionConfig(min_feature_frequency=1.5)

    def test_from_model_family(self):
        """Test creating config from model family."""
        config = FeatureSelectionConfig.from_model_family("boosting")
        assert config.enabled is True
        assert config.n_features == 50
        assert config.model_family == "boosting"

    def test_from_model_family_with_override(self):
        """Test creating config with overrides."""
        config = FeatureSelectionConfig.from_model_family(
            "boosting",
            override={"n_features": 25},
        )
        assert config.n_features == 25

    def test_disabled_config(self):
        """Test creating disabled config."""
        config = FeatureSelectionConfig.disabled()
        assert config.enabled is False
        assert config.n_features == 0

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        config = FeatureSelectionConfig(
            enabled=True,
            n_features=50,
            method="mda",
        )
        data = config.to_dict()
        loaded = FeatureSelectionConfig.from_dict(data)

        assert loaded.enabled == config.enabled
        assert loaded.n_features == config.n_features
        assert loaded.method == config.method


# =============================================================================
# PERSISTED FEATURE SELECTION TESTS
# =============================================================================

class TestPersistedFeatureSelection:
    """Tests for PersistedFeatureSelection."""

    def test_creation(self, persisted_result):
        """Test creating PersistedFeatureSelection."""
        assert persisted_result.n_features_selected == 10
        assert persisted_result.n_features_original == 30
        assert len(persisted_result.selected_features) == 10

    def test_reduction_ratio(self, persisted_result):
        """Test reduction ratio calculation."""
        expected = 1 - (10 / 30)
        assert abs(persisted_result.reduction_ratio - expected) < 0.01

    def test_is_empty(self):
        """Test is_empty property."""
        empty = PersistedFeatureSelection(
            selected_features=[],
            feature_indices={},
            selection_method="mda",
            n_features_original=30,
            n_features_selected=0,
        )
        assert empty.is_empty is True

    def test_get_column_indices(self, sample_feature_data, persisted_result):
        """Test get_column_indices."""
        all_features = sample_feature_data["feature_names"]
        indices = persisted_result.get_column_indices(all_features)

        assert len(indices) == 10
        assert all(isinstance(i, int) for i in indices)

    def test_get_feature_mask(self, sample_feature_data, persisted_result):
        """Test get_feature_mask."""
        all_features = sample_feature_data["feature_names"]
        mask = persisted_result.get_feature_mask(all_features)

        assert len(mask) == len(all_features)
        assert sum(mask) == 10

    def test_save_load(self, persisted_result, tmp_path):
        """Test save and load operations."""
        path = tmp_path / "feature_selection.json"

        persisted_result.save(path)
        assert path.exists()

        loaded = PersistedFeatureSelection.load(path)

        assert loaded.selected_features == persisted_result.selected_features
        assert loaded.n_features_selected == persisted_result.n_features_selected
        assert loaded.selection_method == persisted_result.selection_method

    def test_to_dict_from_dict(self, persisted_result):
        """Test serialization round-trip."""
        data = persisted_result.to_dict()
        loaded = PersistedFeatureSelection.from_dict(data)

        assert loaded.selected_features == persisted_result.selected_features
        assert loaded.n_features_original == persisted_result.n_features_original

    def test_passthrough(self, sample_feature_data):
        """Test creating passthrough (no selection)."""
        features = sample_feature_data["feature_names"]
        passthrough = PersistedFeatureSelection.passthrough(features)

        assert passthrough.n_features_selected == len(features)
        assert passthrough.reduction_ratio == 0.0
        assert passthrough.selection_method == "passthrough"

    def test_load_nonexistent_raises(self, tmp_path):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PersistedFeatureSelection.load(tmp_path / "nonexistent.json")


# =============================================================================
# FEATURE SELECTION MANAGER TESTS
# =============================================================================

class TestFeatureSelectionManager:
    """Tests for FeatureSelectionManager."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = FeatureSelectionConfig(n_features=25, method="mdi")
        manager = FeatureSelectionManager(config=config)

        assert manager.config.n_features == 25
        assert manager.config.method == "mdi"

    def test_init_from_model_family(self):
        """Test initialization from model family."""
        manager = FeatureSelectionManager.from_model_family("boosting")

        assert manager.config.enabled is True
        assert manager.is_enabled is True

    def test_disabled_manager(self):
        """Test creating disabled manager."""
        manager = FeatureSelectionManager.disabled()

        assert manager.is_enabled is False

    def test_is_fitted_before_selection(self):
        """Test is_fitted returns False before selection."""
        manager = FeatureSelectionManager.from_model_family("boosting")
        assert manager.is_fitted is False

    def test_select_features_single_fold(self, sample_feature_data):
        """Test single-fold feature selection."""
        manager = FeatureSelectionManager(n_features=10, method="mdi")

        result = manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
            sample_weights=sample_feature_data["weights"],
        )

        assert result.n_features_selected == 10
        assert manager.is_fitted is True

    def test_select_features_disabled(self, sample_feature_data):
        """Test selection when disabled returns passthrough."""
        manager = FeatureSelectionManager.disabled()

        result = manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
        )

        assert result.n_features_selected == 30
        assert result.selection_method == "passthrough"

    def test_apply_selection(self, sample_feature_data):
        """Test applying feature selection."""
        manager = FeatureSelectionManager(n_features=10, method="mdi")
        manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
        )

        X_selected = manager.apply_selection(sample_feature_data["X"])

        assert X_selected.shape == (500, 10)

    def test_apply_selection_df(self, sample_feature_data):
        """Test applying selection and returning DataFrame."""
        manager = FeatureSelectionManager(n_features=10, method="mdi")
        manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
        )

        X_selected = manager.apply_selection_df(sample_feature_data["X"])

        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape == (500, 10)

    def test_apply_selection_not_fitted_raises(self, sample_feature_data):
        """Test applying selection before fitting raises RuntimeError."""
        manager = FeatureSelectionManager(n_features=10)

        with pytest.raises(RuntimeError, match="not been run"):
            manager.apply_selection(sample_feature_data["X"])

    def test_save_load(self, sample_feature_data, tmp_path):
        """Test save and load operations."""
        manager = FeatureSelectionManager(n_features=10, method="mdi")
        manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
        )

        path = tmp_path / "feature_selection.json"
        manager.save(path)

        # Load in new manager
        loaded_manager = FeatureSelectionManager.load_from_path(path)

        assert loaded_manager.n_features_selected == 10
        assert loaded_manager.is_fitted is True

    def test_get_feature_report(self, sample_feature_data):
        """Test get_feature_report."""
        manager = FeatureSelectionManager(n_features=10, method="mdi")
        manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
        )

        report = manager.get_feature_report()

        assert report["status"] == "complete"
        assert report["n_features_selected"] == 10
        assert report["n_features_original"] == 30
        assert "top_10_features" in report

    def test_get_feature_report_not_run(self):
        """Test get_feature_report before running selection."""
        manager = FeatureSelectionManager(n_features=10)
        report = manager.get_feature_report()

        assert report["status"] == "not_run"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFeatureSelectionTrainerIntegration:
    """Tests for feature selection integration with Trainer."""

    def test_trainer_config_has_feature_selection_settings(self):
        """Test TrainerConfig includes feature selection settings."""
        from src.models.config import TrainerConfig

        config = TrainerConfig(
            model_name="xgboost",
            use_feature_selection=True,
            feature_selection_n_features=30,
            feature_selection_method="mda",
        )

        assert config.use_feature_selection is True
        assert config.feature_selection_n_features == 30
        assert config.feature_selection_method == "mda"

    def test_trainer_config_feature_selection_in_dict(self):
        """Test feature selection settings are serialized."""
        from src.models.config import TrainerConfig

        config = TrainerConfig(
            model_name="xgboost",
            use_feature_selection=True,
            feature_selection_n_features=40,
        )

        config_dict = config.to_dict()

        assert "use_feature_selection" in config_dict
        assert "feature_selection_n_features" in config_dict
        assert config_dict["use_feature_selection"] is True
        assert config_dict["feature_selection_n_features"] == 40


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_n_features_greater_than_total(self, sample_feature_data):
        """Test selecting more features than available."""
        manager = FeatureSelectionManager(n_features=100, method="mdi")

        result = manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
        )

        # Should return passthrough when n_features >= total
        assert result.n_features_selected == 30
        assert result.selection_method == "passthrough"

    def test_numpy_array_input_requires_names(self, sample_feature_data):
        """Test that numpy array input requires feature names."""
        manager = FeatureSelectionManager(n_features=10, method="mdi")
        manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
        )

        # Apply to numpy array without names should raise
        with pytest.raises(ValueError, match="feature_names must be provided"):
            manager.apply_selection(sample_feature_data["X"].values)

    def test_numpy_array_input_with_names(self, sample_feature_data):
        """Test applying selection to numpy array with feature names."""
        manager = FeatureSelectionManager(n_features=10, method="mdi")
        manager.select_features_single_fold(
            X_train=sample_feature_data["X"],
            y_train=sample_feature_data["y"],
        )

        X_selected = manager.apply_selection(
            sample_feature_data["X"].values,
            feature_names=sample_feature_data["feature_names"],
        )

        assert X_selected.shape == (500, 10)

    def test_validation_inconsistent_n_features(self):
        """Test validation catches inconsistent n_features."""
        with pytest.raises(ValueError, match="n_features_selected"):
            PersistedFeatureSelection(
                selected_features=["a", "b", "c"],
                feature_indices={"a": 0, "b": 1, "c": 2},
                selection_method="mda",
                n_features_original=10,
                n_features_selected=5,  # Mismatch!
            )
