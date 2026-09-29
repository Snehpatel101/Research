"""MDA permutation importance is scored on holdout data."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

# =============================================================================
# MDA Holdout Scoring (Fix #6) and Timestamp Alignment (Fix #7)
# =============================================================================


class TestMDAHoldoutScoring:
    """Verify MDA permutation importance uses holdout data."""

    def _make_data(self, n_train=200, n_test=80, n_features=10, seed=42):
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(
            rng.randn(n_train + n_test, n_features),
            columns=[f"f{i}" for i in range(n_features)],
        )
        y = pd.Series((X["f0"] > 0).astype(int))
        return X.iloc[:n_train], y.iloc[:n_train], X.iloc[n_train:], y.iloc[n_train:]

    def test_mda_uses_holdout_when_provided(self):
        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        X_train, y_train, X_test, y_test = self._make_data()
        selector = WalkForwardFeatureSelector(selection_method="mda", n_estimators=20)

        captured = {}
        import sklearn.inspection as si

        original_pi = si.permutation_importance

        def spy_pi(estimator, X, y, **kwargs):
            captured["X_shape"] = X.shape
            captured["y_len"] = len(y)
            return original_pi(estimator, X, y, **kwargs)

        with patch(
            "src.optimization.feature_selection.walk_forward.permutation_importance",
            side_effect=spy_pi,
        ):
            selector._mda_importance(X_train, y_train, X_test=X_test, y_test=y_test)

        assert captured["X_shape"][0] == 80

    def test_mda_warns_without_holdout(self, caplog):
        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        X_train, y_train, _, _ = self._make_data()
        selector = WalkForwardFeatureSelector(selection_method="mda", n_estimators=20)

        with caplog.at_level("WARNING"):
            result = selector._mda_importance(X_train, y_train)

        assert len(result) == 10
        assert "no holdout set provided" in caplog.text

    def test_walkforward_passes_test_split(self):
        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.randint(0, 2, n))

        cv_splits = [
            (np.arange(0, 200), np.arange(200, 260)),
            (np.arange(0, 240), np.arange(240, 300)),
        ]

        selector = WalkForwardFeatureSelector(
            selection_method="mda", n_estimators=10, n_features_to_select=3
        )

        calls = []
        original = selector._compute_importance

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        with patch.object(selector, "_compute_importance", side_effect=spy):
            selector.select_features_walkforward(X, y, cv_splits)

        assert len(calls) == 2
        for _i, call_kwargs in enumerate(calls):
            assert "X_test" in call_kwargs
            assert "y_test" in call_kwargs
