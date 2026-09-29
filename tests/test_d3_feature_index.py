"""
D3: Regression test for C-FSEL-2 — features beyond index 50 are reachable.

The old bug truncated the candidate list positionally (first-N features), so a
predictive feature late in the column order could never be selected. The live
selectors rank by permutation importance (MDA); these tests plant the only
predictive feature at index 60 of 70 and require it to win.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.feature_selection.ohlcv_selector import OHLCVFeatureSelector
from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector

N_FEATURES = 70
SIGNAL_INDEX = 60
SIGNAL = f"feat_{SIGNAL_INDEX}"


@pytest.fixture(scope="module")
def signal_at_index_60() -> tuple[pd.DataFrame, pd.Series]:
    """70 noise features; only feat_60 carries the label."""
    rng = np.random.RandomState(7)
    n = 400
    y = pd.Series(rng.choice([-1, 0, 1], n))
    X = pd.DataFrame(rng.randn(n, N_FEATURES), columns=[f"feat_{i}" for i in range(N_FEATURES)])
    X[SIGNAL] = y + rng.randn(n) * 0.1
    return X, y


class TestMDAReachesTailFeatures:
    def test_walk_forward_selector_picks_feature_60(self, signal_at_index_60):
        X, y = signal_at_index_60
        n = len(X)
        cv_splits = [
            (np.arange(0, n // 2), np.arange(n // 2, 3 * n // 4)),
            (np.arange(0, 3 * n // 4), np.arange(3 * n // 4, n)),
        ]
        selector = WalkForwardFeatureSelector(
            n_features_to_select=5, n_estimators=20, mda_n_repeats=2, min_feature_frequency=1.0
        )

        result = selector.select_features_walkforward(X, y, cv_splits)

        assert SIGNAL in result.selected_features
        for fold in result.importance_history:
            assert fold["n_features_evaluated"] == N_FEATURES, "no candidate may be truncated"
            assert fold["top_feature"] == SIGNAL

    def test_ohlcv_selector_picks_feature_60(self, signal_at_index_60):
        X, y = signal_at_index_60
        selector = OHLCVFeatureSelector(n_splits=2, n_estimators=20, n_features_per_fold=5)

        result = selector.select_features(X.values, y.values, list(X.columns))

        assert SIGNAL in result.selected_features
        assert max(result.feature_importances, key=result.feature_importances.get) == SIGNAL
