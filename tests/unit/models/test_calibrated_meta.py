"""
CalibratedMetaLearner calibrates with time-ordered folds.

sklearn's default (stratified, shuffled) K-fold would train the calibrator on bars
that come AFTER the bars it is calibrated on. Every calibration fold must train on
the past and score on the future.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

import src.models.ensemble.calibrated_meta as calibrated_meta
from src.models.ensemble.calibrated_meta import CalibratedMetaLearner


@pytest.fixture
def calibration_splits(monkeypatch: pytest.MonkeyPatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """Capture the (train, calibrate) index pairs CalibratedClassifierCV iterates."""
    captured: list[tuple[np.ndarray, np.ndarray]] = []

    class _Spy(CalibratedClassifierCV):
        def fit(self, X, y, **kwargs: Any):  # noqa: ANN001, ANN401
            captured.extend(self.cv.split(X, y))  # type: ignore[union-attr]
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(calibrated_meta, "CalibratedClassifierCV", _Spy)
    return captured


def _fit(config: dict[str, Any]) -> CalibratedMetaLearner:
    rng = np.random.default_rng(0)
    n, k = 400, 6
    X_train, X_val = rng.normal(size=(n, k)), rng.normal(size=(100, k))
    y_train, y_val = rng.integers(-1, 2, size=n), rng.integers(-1, 2, size=100)
    learner = CalibratedMetaLearner(config=config)
    learner.fit(X_train, y_train, X_val, y_val)
    return learner


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_every_calibration_fold_trains_strictly_before_it_calibrates(
    calibration_splits, method: str
) -> None:
    _fit({"cv": 4, "method": method})

    assert len(calibration_splits) == 4
    for train_idx, calib_idx in calibration_splits:
        assert train_idx.max() < calib_idx.min(), "calibration fold overlaps or precedes its train"


def test_calibrated_probabilities_are_valid_after_fit() -> None:
    learner = _fit({"cv": 3})
    proba = learner.predict(np.random.default_rng(1).normal(size=(20, 6))).class_probabilities

    assert proba.shape == (20, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
