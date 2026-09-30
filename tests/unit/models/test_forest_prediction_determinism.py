"""Forest predictions are bit-identical from call to call.

scikit-learn forests with ``n_jobs != 1`` add per-tree probabilities in
thread-completion order, so ``predict_proba`` changes in the last bits on every
call. In MDA feature ranking that turned the importance of a feature the forest
never uses into random +/-1e-17 noise, so near-tied features swapped ranks and
two identical runs could select different features (seen as a flaky
``test_identical_metrics_across_runs`` under load).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.core.reproducibility import sequential_prediction
from src.models.classical.random_forest import RandomForestModel
from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector


def _data(n: int = 4000, n_features: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1]
    y = np.digitize(signal + rng.normal(scale=1.0, size=n), [-0.5, 0.5]) - 1
    return X, y


def test_sequential_prediction_sets_single_thread() -> None:
    rf = RandomForestClassifier(n_estimators=5, n_jobs=-1, random_state=0)
    assert sequential_prediction(rf) is rf
    assert rf.n_jobs == 1


class TestRandomForestModel:
    def test_predictions_bit_identical_and_match_sequential(self) -> None:
        X, y = _data()
        model = RandomForestModel(config={"n_estimators": 40, "n_jobs": -1, "random_state": 3})
        model.fit(X[:3000], y[:3000], X[3000:], y[3000:])

        first = model.predict_proba(X)
        for _ in range(5):
            assert np.array_equal(model.predict_proba(X), first)
        # Row-parallel prediction equals a plain single-threaded pass
        forest = model._model
        assert forest is not None and forest.n_jobs == 1
        assert np.array_equal(first, forest.predict_proba(X))
        assert np.array_equal(model.predict(X).class_probabilities, first)


class TestClusteredMDA:
    @pytest.fixture
    def frames(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X, y = _data(n=3000)
        cols = [f"f{i}" for i in range(X.shape[1])]
        frame = pd.DataFrame(X, columns=cols)
        frame["constant"] = 1.0  # never split on: its importance must be exactly 0
        labels = pd.Series(y)
        return frame.iloc[:2000], labels.iloc[:2000], frame.iloc[2000:], labels.iloc[2000:]

    def test_importances_bit_identical(self, frames) -> None:
        X, y, X_test, y_test = frames
        selector = WalkForwardFeatureSelector(
            n_estimators=20, max_clusters=X.shape[1], random_state=5
        )
        first = selector._clustered_mda_importance(X, y, X_test=X_test, y_test=y_test)
        for _ in range(3):
            again = selector._clustered_mda_importance(X, y, X_test=X_test, y_test=y_test)
            assert again.equals(first)

    def test_unused_feature_scores_exactly_zero(self, frames) -> None:
        X, y, X_test, y_test = frames
        selector = WalkForwardFeatureSelector(n_estimators=20, mda_n_repeats=2, random_state=5)
        clustered = selector._clustered_mda_importance(X, y, X_test=X_test, y_test=y_test)
        plain = selector._mda_importance(X, y, X_test=X_test, y_test=y_test)
        assert clustered["constant"] == 0.0
        assert plain["constant"] == 0.0
