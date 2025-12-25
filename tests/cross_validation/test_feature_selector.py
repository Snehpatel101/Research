"""
Tests for Walk-Forward Feature Selector.

Tests:
- MDI (Mean Decrease in Impurity) importance computation
- MDA (Mean Decrease in Accuracy) importance computation
- Hybrid importance (MDI + MDA)
- Walk-forward selection across CV folds
- Stable feature identification
- Clustered feature importance
"""
import numpy as np
import pandas as pd
import pytest

from src.cross_validation.feature_selector import (
    WalkForwardFeatureSelector,
    FeatureSelectorConfig,
    FeatureSelectionResult,
    CVIntegratedFeatureSelector,
)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestFeatureSelectorConfig:
    """Tests for FeatureSelectorConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creates successfully."""
        config = FeatureSelectorConfig(
            n_features_to_select=50,
            selection_method="mda",
            n_estimators=100,
            min_feature_frequency=0.6,
        )
        assert config.n_features_to_select == 50
        assert config.selection_method == "mda"
        assert config.n_estimators == 100
        assert config.min_feature_frequency == 0.6

    def test_invalid_n_features_zero(self):
        """Test that n_features_to_select <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_features_to_select must be > 0"):
            FeatureSelectorConfig(n_features_to_select=0)

    def test_invalid_selection_method(self):
        """Test that invalid selection_method raises ValueError."""
        with pytest.raises(ValueError, match="selection_method must be"):
            FeatureSelectorConfig(selection_method="invalid")

    def test_invalid_min_frequency_zero(self):
        """Test that min_feature_frequency <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="min_feature_frequency must be in"):
            FeatureSelectorConfig(min_feature_frequency=0)

    def test_invalid_min_frequency_above_one(self):
        """Test that min_feature_frequency > 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_feature_frequency must be in"):
            FeatureSelectorConfig(min_feature_frequency=1.5)


# =============================================================================
# MDI IMPORTANCE TESTS
# =============================================================================

class TestMDIImportance:
    """Tests for MDI (Mean Decrease in Impurity) importance."""

    def test_mdi_returns_series(self, small_time_series_data):
        """Test that MDI importance returns pandas Series."""
        selector = WalkForwardFeatureSelector(selection_method="mdi")
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        importance = selector._mdi_importance(X, y)

        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]

    def test_mdi_values_nonnegative(self, small_time_series_data):
        """Test that MDI importance values are non-negative."""
        selector = WalkForwardFeatureSelector(selection_method="mdi")
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        importance = selector._mdi_importance(X, y)

        assert (importance >= 0).all()

    def test_mdi_values_sum_to_one(self, small_time_series_data):
        """Test that MDI importance values sum to approximately 1."""
        selector = WalkForwardFeatureSelector(selection_method="mdi")
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        importance = selector._mdi_importance(X, y)

        assert np.isclose(importance.sum(), 1.0, atol=0.01)

    def test_mdi_reproducible(self, small_time_series_data):
        """Test that MDI importance is reproducible with same seed."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        selector1 = WalkForwardFeatureSelector(selection_method="mdi", random_state=42)
        selector2 = WalkForwardFeatureSelector(selection_method="mdi", random_state=42)

        importance1 = selector1._mdi_importance(X, y)
        importance2 = selector2._mdi_importance(X, y)

        pd.testing.assert_series_equal(importance1, importance2)


# =============================================================================
# MDA IMPORTANCE TESTS
# =============================================================================

class TestMDAImportance:
    """Tests for MDA (Mean Decrease in Accuracy) importance."""

    def test_mda_returns_series(self, small_time_series_data):
        """Test that MDA importance returns pandas Series."""
        selector = WalkForwardFeatureSelector(selection_method="mda")
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        importance = selector._mda_importance(X, y)

        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]

    def test_mda_can_be_negative(self, small_time_series_data):
        """Test that MDA importance can have negative values (noise features)."""
        selector = WalkForwardFeatureSelector(selection_method="mda")
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        importance = selector._mda_importance(X, y)

        # MDA can have negative values for noise features
        # Just verify it runs and returns valid output
        assert not importance.isna().any()

    def test_mda_reproducible(self, small_time_series_data):
        """Test that MDA importance is reproducible with same seed."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        selector1 = WalkForwardFeatureSelector(selection_method="mda", random_state=42)
        selector2 = WalkForwardFeatureSelector(selection_method="mda", random_state=42)

        importance1 = selector1._mda_importance(X, y)
        importance2 = selector2._mda_importance(X, y)

        pd.testing.assert_series_equal(importance1, importance2)


# =============================================================================
# HYBRID IMPORTANCE TESTS
# =============================================================================

class TestHybridImportance:
    """Tests for hybrid (MDI + MDA) importance."""

    def test_hybrid_returns_series(self, small_time_series_data):
        """Test that hybrid importance returns pandas Series."""
        selector = WalkForwardFeatureSelector(selection_method="hybrid")
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        importance = selector._compute_importance(X, y)

        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]

    def test_hybrid_combines_rankings(self, small_time_series_data):
        """Test that hybrid importance combines MDI and MDA rankings."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        selector = WalkForwardFeatureSelector(selection_method="hybrid", random_state=42)

        # Get individual importances
        mdi = selector._mdi_importance(X, y)
        mda = selector._mda_importance(X, y)

        # Get hybrid
        hybrid = selector._compute_importance(X, y)

        # Hybrid should be average of ranks
        expected_hybrid = (mdi.rank() + mda.rank()) / 2

        pd.testing.assert_series_equal(hybrid, expected_hybrid)


# =============================================================================
# WALK-FORWARD SELECTION TESTS
# =============================================================================

class TestWalkForwardSelection:
    """Tests for walk-forward feature selection."""

    def test_select_features_returns_result(self, small_time_series_data, cv_splits):
        """Test that selection returns FeatureSelectionResult."""
        selector = WalkForwardFeatureSelector(n_features_to_select=5)
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        result = selector.select_features_walkforward(X, y, cv_splits)

        assert isinstance(result, FeatureSelectionResult)
        assert hasattr(result, "stable_features")
        assert hasattr(result, "feature_counts")
        assert hasattr(result, "per_fold_selections")

    def test_selects_correct_number_per_fold(self, small_time_series_data, cv_splits):
        """Test that correct number of features selected per fold."""
        n_features = 5
        selector = WalkForwardFeatureSelector(n_features_to_select=n_features)
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        result = selector.select_features_walkforward(X, y, cv_splits)

        for fold_selection in result.per_fold_selections:
            assert len(fold_selection) == n_features

    def test_stable_features_meet_threshold(self, small_time_series_data, cv_splits):
        """Test that stable features appear in >= min_frequency folds."""
        min_freq = 0.6
        selector = WalkForwardFeatureSelector(
            n_features_to_select=5,
            min_feature_frequency=min_freq,
        )
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        result = selector.select_features_walkforward(X, y, cv_splits)

        n_folds = len(cv_splits)
        min_count = int(n_folds * min_freq)

        for feature in result.stable_features:
            assert result.feature_counts[feature] >= min_count

    def test_feature_counts_are_correct(self, small_time_series_data, cv_splits):
        """Test that feature counts match actual selections."""
        selector = WalkForwardFeatureSelector(n_features_to_select=5)
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        result = selector.select_features_walkforward(X, y, cv_splits)

        # Verify counts
        for feature, count in result.feature_counts.items():
            actual_count = sum(feature in s for s in result.per_fold_selections)
            assert count == actual_count

    def test_importance_history_recorded(self, small_time_series_data, cv_splits):
        """Test that importance history is recorded for each fold."""
        selector = WalkForwardFeatureSelector(n_features_to_select=5)
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        result = selector.select_features_walkforward(X, y, cv_splits)

        assert len(result.importance_history) == len(cv_splits)

        for hist in result.importance_history:
            assert "fold" in hist
            assert "importance" in hist
            assert "top_feature" in hist

    def test_n_folds_attribute(self, small_time_series_data, cv_splits):
        """Test that n_folds attribute is set correctly."""
        selector = WalkForwardFeatureSelector(n_features_to_select=5)
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        result = selector.select_features_walkforward(X, y, cv_splits)

        assert result.n_folds == len(cv_splits)


# =============================================================================
# STABLE FEATURE IDENTIFICATION TESTS
# =============================================================================

class TestStableFeatureIdentification:
    """Tests for identifying stable features across folds."""

    def test_higher_threshold_fewer_features(self, small_time_series_data, cv_splits):
        """Test that higher min_frequency threshold yields fewer stable features."""
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        selector_low = WalkForwardFeatureSelector(
            n_features_to_select=5,
            min_feature_frequency=0.3,
        )
        selector_high = WalkForwardFeatureSelector(
            n_features_to_select=5,
            min_feature_frequency=0.9,
        )

        result_low = selector_low.select_features_walkforward(X, y, cv_splits)
        result_high = selector_high.select_features_walkforward(X, y, cv_splits)

        assert len(result_high.stable_features) <= len(result_low.stable_features)

    def test_stability_scores(self, small_time_series_data, cv_splits):
        """Test get_stability_scores method."""
        selector = WalkForwardFeatureSelector(n_features_to_select=5)
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        result = selector.select_features_walkforward(X, y, cv_splits)
        stability_scores = result.get_stability_scores()

        assert isinstance(stability_scores, dict)

        for feature, score in stability_scores.items():
            assert 0 <= score <= 1
            expected_score = result.feature_counts[feature] / result.n_folds
            assert np.isclose(score, expected_score)


# =============================================================================
# CLUSTERED IMPORTANCE TESTS
# =============================================================================

class TestClusteredImportance:
    """Tests for clustered MDA importance."""

    def test_clustered_mda_returns_series(self, correlated_features_data):
        """Test that clustered MDA returns pandas Series."""
        selector = WalkForwardFeatureSelector(
            selection_method="mda",
            use_clustered_importance=True,
            max_clusters=5,
        )
        X = correlated_features_data["X"]
        y = correlated_features_data["y"]

        importance = selector._clustered_mda_importance(X, y)

        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]

    def test_clustered_importance_nonnegative(self, correlated_features_data):
        """Test that clustered importance values are non-negative."""
        selector = WalkForwardFeatureSelector(
            selection_method="mda",
            use_clustered_importance=True,
            max_clusters=5,
        )
        X = correlated_features_data["X"]
        y = correlated_features_data["y"]

        importance = selector._clustered_mda_importance(X, y)

        assert (importance >= 0).all()

    def test_clustered_distributes_within_cluster(self, correlated_features_data):
        """Test that importance is distributed equally within clusters."""
        selector = WalkForwardFeatureSelector(
            selection_method="mda",
            use_clustered_importance=True,
            max_clusters=3,  # Should create ~3 clusters given 3 feature groups
        )
        X = correlated_features_data["X"]
        y = correlated_features_data["y"]

        importance = selector._clustered_mda_importance(X, y)

        # Features in same correlated group should have similar importance
        high_corr_importance = [importance[f] for f in correlated_features_data["high_corr_group"]]

        # All values in a correlated group should be close
        assert np.std(high_corr_importance) < np.mean(high_corr_importance) * 0.5


# =============================================================================
# CV INTEGRATED FEATURE SELECTOR TESTS
# =============================================================================

class TestCVIntegratedFeatureSelector:
    """Tests for CVIntegratedFeatureSelector."""

    def test_select_single_fold(self, small_time_series_data):
        """Test select_single_fold returns list of features."""
        selector = CVIntegratedFeatureSelector(n_features=5)
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]

        features = selector.select_single_fold(X, y)

        assert isinstance(features, list)
        assert len(features) == 5
        assert all(f in X.columns for f in features)

    def test_select_single_fold_with_weights(self, small_time_series_data):
        """Test select_single_fold works with sample weights."""
        selector = CVIntegratedFeatureSelector(n_features=5)
        X = small_time_series_data["X"]
        y = small_time_series_data["y"]
        weights = small_time_series_data["weights"]

        features = selector.select_single_fold(X, y, sample_weights=weights)

        assert isinstance(features, list)
        assert len(features) == 5


# =============================================================================
# FEATURE SELECTION RESULT TESTS
# =============================================================================

class TestFeatureSelectionResult:
    """Tests for FeatureSelectionResult dataclass."""

    def test_n_folds_auto_computed(self):
        """Test that n_folds is auto-computed if not provided."""
        result = FeatureSelectionResult(
            stable_features=["a", "b"],
            feature_counts={"a": 3, "b": 2},
            per_fold_selections=[{"a", "b"}, {"a"}, {"a", "b"}],
            importance_history=[],
        )

        assert result.n_folds == 3

    def test_get_stability_scores_empty(self):
        """Test get_stability_scores with empty result."""
        result = FeatureSelectionResult(
            stable_features=[],
            feature_counts={},
            per_fold_selections=[],
            importance_history=[],
            n_folds=0,
        )

        scores = result.get_stability_scores()
        assert scores == {}
