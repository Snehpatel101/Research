"""
Tests for PurgedFeatureSelector.

Tests cover integration with PurgedKFold for proper temporal handling
with purge and embargo constraints.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.feature_selection import (
    PurgedFeatureSelector,
    FeatureSelectionResult,
    create_purged_selector,
)


# =============================================================================
# MOCK CV CLASS FOR TESTING
# =============================================================================

class MockPurgedKFold:
    """Mock PurgedKFold for testing without importing the real class."""

    def __init__(self, n_splits: int = 5, purge_bars: int = 60, embargo_bars: int = 1440):
        self.config = MagicMock()
        self.config.n_splits = n_splits
        self.config.purge_bars = purge_bars
        self.config.embargo_bars = embargo_bars
        self.n_splits = n_splits

    def split(self, X, y=None):
        """Generate time-series splits with purge."""
        n_samples = len(X)
        fold_size = n_samples // self.n_splits

        for fold_idx in range(self.n_splits):
            # Simple time-series split (train before test)
            test_start = fold_idx * fold_size
            test_end = min((fold_idx + 1) * fold_size, n_samples)

            # Train indices (everything before test, minus purge)
            train_end = max(0, test_start - self.config.purge_bars)
            if train_end < 100:
                train_end = test_start  # Fallback for small datasets

            train_idx = np.arange(train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) >= 100 and len(test_idx) >= 50:
                yield train_idx, test_idx


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_cv():
    """Create mock PurgedKFold."""
    return MockPurgedKFold(n_splits=3, purge_bars=20, embargo_bars=100)


@pytest.fixture
def time_series_data():
    """Create time series data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Make first 3 features predictive
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }


@pytest.fixture
def regime_data():
    """Create data with regimes."""
    np.random.seed(42)
    n_samples = 1000

    X = np.random.randn(n_samples, 10)
    feature_names = [f"feature_{i}" for i in range(10)]

    # Create regimes
    regimes = np.zeros(n_samples, dtype=int)
    regimes[n_samples // 2:] = 1

    # Label based on different features per regime
    y = np.zeros(n_samples, dtype=int)
    y[regimes == 0] = (X[regimes == 0, 0] > 0).astype(int)
    y[regimes == 1] = (X[regimes == 1, 1] > 0).astype(int)

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "regimes": regimes,
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestPurgedFeatureSelectorConfig:
    """Tests for selector configuration."""

    def test_basic_initialization(self, mock_cv):
        """Test basic initialization."""
        selector = PurgedFeatureSelector(cv=mock_cv)
        assert selector.cv is mock_cv
        assert selector.min_stability_score == 0.5
        assert selector.correlation_threshold == 0.85

    def test_custom_parameters(self, mock_cv):
        """Test custom parameter initialization."""
        selector = PurgedFeatureSelector(
            cv=mock_cv,
            min_stability_score=0.7,
            correlation_threshold=0.9,
            use_regime_conditioning=True,
        )
        assert selector.min_stability_score == 0.7
        assert selector.correlation_threshold == 0.9
        assert selector.use_regime_conditioning is True


# =============================================================================
# SELECTION TESTS
# =============================================================================

class TestPurgedSelection:
    """Tests for feature selection with PurgedKFold."""

    def test_basic_selection(self, mock_cv, time_series_data):
        """Test basic feature selection."""
        selector = PurgedFeatureSelector(
            cv=mock_cv,
            min_stability_score=0.0,  # Accept all features for testing
        )

        result = selector.select_features(
            time_series_data["X"],
            time_series_data["y"],
            time_series_data["feature_names"],
        )

        assert isinstance(result, FeatureSelectionResult)
        assert result.n_original == 20
        assert result.n_selected > 0
        assert len(result.selected_features) == result.n_selected

    def test_selection_metadata_includes_purge_embargo(self, mock_cv, time_series_data):
        """Test that selection metadata includes purge/embargo info."""
        selector = PurgedFeatureSelector(cv=mock_cv, min_stability_score=0.0)

        result = selector.select_features(
            time_series_data["X"],
            time_series_data["y"],
            time_series_data["feature_names"],
        )

        assert "purge_bars" in result.selection_metadata
        assert "embargo_bars" in result.selection_metadata
        assert result.selection_metadata["purge_bars"] == 20
        assert result.selection_metadata["embargo_bars"] == 100

    def test_selection_with_regimes(self, mock_cv, regime_data):
        """Test selection with regime conditioning."""
        selector = PurgedFeatureSelector(
            cv=mock_cv,
            min_stability_score=0.0,
            use_regime_conditioning=True,
        )

        result = selector.select_features(
            regime_data["X"],
            regime_data["y"],
            regime_data["feature_names"],
            regimes=regime_data["regimes"],
        )

        assert result.regime_importances is not None

    def test_selection_with_sample_weights(self, mock_cv, time_series_data):
        """Test selection with sample weights."""
        selector = PurgedFeatureSelector(cv=mock_cv, min_stability_score=0.0)

        weights = np.random.rand(len(time_series_data["y"]))

        result = selector.select_features(
            time_series_data["X"],
            time_series_data["y"],
            time_series_data["feature_names"],
            sample_weights=weights,
        )

        assert result.n_selected > 0


# =============================================================================
# CV INTEGRATION TESTS
# =============================================================================

class TestCVIntegration:
    """Tests for integration with PurgedKFold."""

    def test_uses_cv_splits(self, mock_cv, time_series_data):
        """Test that selector uses CV splits."""
        selector = PurgedFeatureSelector(cv=mock_cv, min_stability_score=0.0)

        # Track how many times split() is called
        original_split = mock_cv.split
        call_count = [0]

        def counting_split(*args, **kwargs):
            call_count[0] += 1
            return original_split(*args, **kwargs)

        mock_cv.split = counting_split

        selector.select_features(
            time_series_data["X"],
            time_series_data["y"],
            time_series_data["feature_names"],
        )

        assert call_count[0] > 0

    def test_respects_n_splits(self, time_series_data):
        """Test that number of folds matches CV configuration."""
        cv_3_splits = MockPurgedKFold(n_splits=3)
        selector = PurgedFeatureSelector(cv=cv_3_splits, min_stability_score=0.0)

        result = selector.select_features(
            time_series_data["X"],
            time_series_data["y"],
            time_series_data["feature_names"],
        )

        assert result.selection_metadata["n_splits"] == 3


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_feature_name_mismatch(self, mock_cv, time_series_data):
        """Test error on feature name mismatch."""
        selector = PurgedFeatureSelector(cv=mock_cv)

        with pytest.raises(ValueError, match="feature_names length"):
            selector.select_features(
                time_series_data["X"],
                time_series_data["y"],
                ["wrong", "number"],
            )


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunction:
    """Tests for create_purged_selector factory."""

    def test_create_default(self, mock_cv):
        """Test default factory creation."""
        selector = create_purged_selector(cv=mock_cv)
        assert isinstance(selector, PurgedFeatureSelector)

    def test_create_custom(self, mock_cv):
        """Test custom factory creation."""
        selector = create_purged_selector(
            cv=mock_cv,
            min_stability=0.7,
            correlation_threshold=0.9,
            use_regimes=True,
        )
        assert selector.min_stability_score == 0.7
        assert selector.correlation_threshold == 0.9
        assert selector.use_regime_conditioning is True


# =============================================================================
# INTEGRATION WITH REAL PURGEDKFOLD (if available)
# =============================================================================

class TestRealPurgedKFoldIntegration:
    """Integration tests with real PurgedKFold (skipped if not available)."""

    @pytest.fixture
    def real_cv(self):
        """Try to create real PurgedKFold."""
        try:
            from src.cross_validation import PurgedKFold, PurgedKFoldConfig
            config = PurgedKFoldConfig(n_splits=3, purge_bars=20, embargo_bars=50)
            return PurgedKFold(config)
        except ImportError:
            pytest.skip("PurgedKFold not available")

    def test_with_real_purged_kfold(self, real_cv, time_series_data):
        """Test with real PurgedKFold."""
        selector = PurgedFeatureSelector(cv=real_cv, min_stability_score=0.0)

        result = selector.select_features(
            time_series_data["X"],
            time_series_data["y"],
            time_series_data["feature_names"],
        )

        assert isinstance(result, FeatureSelectionResult)
        assert result.n_selected > 0
