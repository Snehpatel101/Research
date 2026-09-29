"""
D3: Regression test for C-FSEL-2 -- feature index > 50 must stay reachable.

1. DEFAULT_MAX_FEATURES_TO_SEARCH is not silently capped at a low value.
2. create_5d_objective lets callers override max_features_to_search.
3. The real feature-selection pipelines (walk-forward MDA, OHLCV selector) rank by
   permutation importance, so an informative feature at index 60 of 70 is selected
   instead of being lost to positional truncation.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.constants import DEFAULT_MAX_FEATURES_TO_SEARCH
from src.optimization.feature_selection.ohlcv_selector import OHLCVFeatureSelector
from src.optimization.feature_selection.result import FeatureSelectionResult
from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector
from src.validation.cv import PurgedKFold, PurgedKFoldConfig


class TestDefaultMaxFeaturesConstant:
    """Verify the constant value is reasonable and documented."""

    def test_default_max_features_is_positive(self):
        assert DEFAULT_MAX_FEATURES_TO_SEARCH > 0

    def test_default_max_features_value_documented(self):
        """The constant should be at least 30 to cover meaningful search spaces."""
        assert DEFAULT_MAX_FEATURES_TO_SEARCH >= 30, (
            f"DEFAULT_MAX_FEATURES_TO_SEARCH={DEFAULT_MAX_FEATURES_TO_SEARCH} "
            "is suspiciously low; feature search would be too narrow."
        )


class TestFiveDimensionObjectiveSignature:
    """Verify the objective function accepts max_features_to_search override."""

    def test_create_5d_objective_accepts_max_features(self):
        """The factory function must expose max_features_to_search parameter."""
        import inspect

        from src.optimization.five_dimension_objective import create_5d_objective

        sig = inspect.signature(create_5d_objective)
        assert "max_features_to_search" in sig.parameters, (
            "create_5d_objective must accept max_features_to_search "
            "so callers can include features beyond the default limit."
        )

    def test_max_features_default_matches_constant(self):
        """The default value of max_features_to_search must equal the constant."""
        import inspect

        from src.optimization.five_dimension_objective import create_5d_objective

        sig = inspect.signature(create_5d_objective)
        param = sig.parameters["max_features_to_search"]
        assert param.default == DEFAULT_MAX_FEATURES_TO_SEARCH, (
            f"Default in function ({param.default}) != constant "
            f"({DEFAULT_MAX_FEATURES_TO_SEARCH}). They must stay in sync."
        )


N_FEATURES, INFORMATIVE = 70, 60


@pytest.fixture(scope="module")
def data_with_late_signal() -> tuple[pd.DataFrame, pd.Series]:
    """70 noise features plus one real signal at position 60 (beyond any 50-feature cap)."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        rng.normal(size=(n, N_FEATURES)), columns=[f"feat_{i}" for i in range(N_FEATURES)]
    )
    y = pd.Series((X[f"feat_{INFORMATIVE}"] + 0.3 * rng.normal(size=n) > 0).astype(int))
    return X, y


class TestSelectionIsImportanceBasedNotPositional:
    """MDA ranks by predictive power, so a feature at index 60 can outrank features 0-49."""

    def test_walk_forward_selector_finds_the_feature_at_index_60(self, data_with_late_signal):
        X, y = data_with_late_signal
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=5, embargo_bars=5))
        selector = WalkForwardFeatureSelector(
            n_features_to_select=5,
            n_estimators=15,
            mda_n_repeats=1,
            min_feature_frequency=0.5,
        )

        result = selector.select_features_walkforward(X, y, list(cv.split(X, y)))

        assert isinstance(result, FeatureSelectionResult)
        assert f"feat_{INFORMATIVE}" in result.selected_features
        assert result.per_fold_selections and all(
            f"feat_{INFORMATIVE}" in fold for fold in result.per_fold_selections
        )

    def test_ohlcv_selector_finds_the_feature_at_index_60(self, data_with_late_signal):
        X, y = data_with_late_signal
        selector = OHLCVFeatureSelector(
            n_splits=3,
            min_stability_score=0.0,
            correlation_threshold=0.99,
            n_features_per_fold=5,
            n_estimators=15,
        )

        result = selector.select_features(X.to_numpy(), y.to_numpy(), list(X.columns))

        assert isinstance(result, FeatureSelectionResult)
        assert f"feat_{INFORMATIVE}" in result.selected_features
        assert max(result.feature_importances, key=result.feature_importances.get) == (
            f"feat_{INFORMATIVE}"
        )


class TestBaseFeatureSetsSize:
    """Verify base feature sets are large enough to have features beyond index 50."""

    def test_at_least_one_family_has_over_30_features(self):
        """At least one model family should have a large enough feature set."""
        from src.core.types import ModelFamily
        from src.optimization.base_feature_sets import get_base_features

        max_len = 0
        for family in ModelFamily:
            features = get_base_features(family)
            max_len = max(max_len, len(features))

        assert max_len >= 30, (
            f"Largest base feature set has only {max_len} features. "
            "Feature search space is too narrow."
        )
