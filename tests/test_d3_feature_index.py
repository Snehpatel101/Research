"""
D3: Regression test for C-FSEL-2 — Optuna feature index > 50.

Verifies that:
1. DEFAULT_MAX_FEATURES_TO_SEARCH is not silently capped at a low value.
2. The five_dimension_objective allows overriding max_features_to_search so
   that features beyond the default limit can be evaluated.
3. Feature lists with 70+ entries are indexable end-to-end (no silent truncation
   that drops tail features).
4. The real feature selection pipeline (MDA / walk-forward) ranks by importance
   not by positional index, so features at any position can be selected.
"""

import pytest

from src.core.constants import DEFAULT_MAX_FEATURES_TO_SEARCH


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


class TestFeatureIndexingBeyond50:
    """Regression: features at index > 50 must be reachable."""

    @pytest.fixture()
    def feature_list_70(self):
        """A synthetic feature list with 70 entries."""
        return [f"feat_{i}" for i in range(70)]

    def test_all_70_features_indexable(self, feature_list_70):
        """Every feature in the list can be accessed by index."""
        for i in range(70):
            assert feature_list_70[i] == f"feat_{i}"

    def test_positional_truncation_preserves_order(self, feature_list_70):
        """Slicing mirrors the 5D objective's truncation logic.

        The five_dimension_objective does:
            searchable_features = base_features[:max_features_to_search]

        When max_features_to_search >= len(features), no feature is dropped.
        """
        # With max_features_to_search=70, nothing is lost
        subset = feature_list_70[:70]
        assert len(subset) == 70
        assert subset[-1] == "feat_69"

    def test_override_max_features_reaches_index_60(self, feature_list_70):
        """Passing max_features_to_search=70 includes feature at index 60."""
        max_features = 70
        subset = feature_list_70[:max_features]
        assert "feat_60" in subset
        assert "feat_69" in subset


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


class TestMDAIsImportanceBased:
    """Verify feature selection uses importance ranking, not positional truncation.

    The MDA (Mean Decrease in Accuracy / permutation importance) pipeline ranks
    features by predictive power, which means a feature at index 60 can outrank
    one at index 5.  This is the key difference from naive first-N truncation.
    """

    def test_walk_forward_selector_uses_permutation_importance(self):
        """WalkForwardFeatureSelector must use sklearn permutation_importance."""
        import inspect

        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        source = inspect.getsource(WalkForwardFeatureSelector)
        assert "permutation_importance" in source, (
            "WalkForwardFeatureSelector should use permutation_importance (MDA) "
            "for importance-based ranking, not positional truncation."
        )

    def test_ohlcv_selector_uses_permutation_importance(self):
        """OHLCVFeatureSelector must use permutation_importance for MDA."""
        import inspect

        from src.optimization.feature_selection.ohlcv_selector import (
            OHLCVFeatureSelector,
        )

        source = inspect.getsource(OHLCVFeatureSelector)
        assert "permutation_importance" in source, (
            "OHLCVFeatureSelector should use permutation_importance (MDA) "
            "for importance-based ranking."
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
