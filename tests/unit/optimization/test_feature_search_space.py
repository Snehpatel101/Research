"""
Feature selection ranks by importance, not position: an informative feature at
index 60 of 70 must be selected by the walk-forward MDA selector instead of
being lost to positional truncation.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.feature_selection.result import FeatureSelectionResult
from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector
from src.validation.cv import PurgedKFold, PurgedKFoldConfig

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
