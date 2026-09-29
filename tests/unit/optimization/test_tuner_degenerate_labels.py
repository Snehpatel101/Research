"""
D4: degenerate label trials never look like good trials in the live tuner.

Phase 92 made failed/degenerate Optuna trials score -inf instead of 0.0 so they
could not pollute best-value selection. That fix lived in the deleted 5-D
objective; these tests pin the same property on the live path,
``TimeSeriesOptunaTuner`` (used by ``HyperparameterTuningService``).

A fold whose train or validation labels hold a single class has no
hyperparameter signal — a constant predictor scores a perfect F1 on it.
The tuner must score such trials at the worst possible value, never fit a
model on them, and hand back empty params so callers keep model defaults.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.validation.cv.cv_tuner as cv_tuner
from src.validation.cv import PurgedKFold, PurgedKFoldConfig, TimeSeriesOptunaTuner


class _StubModel:
    """Predicts the most frequent training class; records every fit."""

    fits: list[np.ndarray] = []

    def __init__(self) -> None:
        self._majority = 0

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        _StubModel.fits.append(np.asarray(y_train))
        values, counts = np.unique(y_train, return_counts=True)
        self._majority = values[np.argmax(counts)]

    def predict(self, X):
        return SimpleNamespace(class_predictions=np.full(len(X), self._majority))


class _StubRegistry:
    @staticmethod
    def create(model_name, config=None):
        return _StubModel()


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    """Replace real model construction so the tests stay fast and observable."""
    monkeypatch.setattr(cv_tuner, "ModelRegistry", _StubRegistry)
    _StubModel.fits = []


def _features(n: int) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(rng.randn(n, 4), columns=[f"f{i}" for i in range(4)])


def _tuner(direction: str = "maximize") -> TimeSeriesOptunaTuner:
    cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, embargo_bars=2))
    return TimeSeriesOptunaTuner(
        "xgboost", cv, n_trials=3, metric="f1_weighted", direction=direction
    )


class TestDegenerateLabelsLiveTuner:
    def test_single_class_labels_score_neg_inf_without_fitting(self):
        n = 240
        result = _tuner().tune(_features(n), pd.Series(np.ones(n, dtype=int)))

        assert math.isinf(result["best_value"]) and result["best_value"] < 0
        assert result["best_params"] == {}, "degenerate trial params must not be applied"
        assert result["skipped"] is True
        assert _StubModel.fits == [], "no model may be fit on single-class folds"

    def test_one_degenerate_fold_rejects_the_trial(self):
        """Only the first third is single-class — the fold it forms still poisons."""
        n = 240
        rng = np.random.RandomState(1)
        y = pd.Series(np.r_[np.zeros(n // 3, dtype=int), rng.choice([-1, 0, 1], n - n // 3)])

        result = _tuner().tune(_features(n), y)

        assert result["best_value"] == float("-inf")
        assert result["best_params"] == {}
        assert _StubModel.fits == []

    def test_minimize_direction_scores_pos_inf(self):
        """The worst value depends on direction: +inf when minimizing."""
        n = 240
        result = _tuner(direction="minimize").tune(_features(n), pd.Series(np.ones(n, dtype=int)))

        assert result["best_value"] == float("inf")
        assert result["best_params"] == {}

    def test_healthy_labels_are_tuned_normally(self):
        """Guard is not over-eager: multi-class folds fit models and score finitely."""
        n = 240
        rng = np.random.RandomState(2)
        y = pd.Series(rng.choice([-1, 0, 1], n))

        result = _tuner().tune(_features(n), y)

        assert np.isfinite(result["best_value"])
        assert result["best_params"], "healthy trials must return sampled params"
        assert "skipped" not in result
        assert len(_StubModel.fits) > 0

    def test_3d_single_class_labels_rejected(self):
        """The guard covers the ndarray (3D/4D) path too."""
        n = 240
        X = np.random.RandomState(3).randn(n, 5, 4).astype(np.float32)

        result = _tuner().tune(X, np.full(n, -1), data_rank=3)

        assert result["best_value"] == float("-inf")
        assert _StubModel.fits == []


class TestDegenerateLabelsTuningService:
    def test_service_keeps_defaults_for_degenerate_labels(self):
        """HyperparameterTuningService (the path factory training uses) returns no
        params and a -inf score instead of a spurious perfect score."""
        from src.data.adapters import PreparedData
        from src.models.training.services.hyperparameter_tuning import (
            HyperparameterTuningService,
            TuningRequest,
        )

        n = 240
        prepared = PreparedData(
            X_train=np.random.RandomState(4).randn(n, 4).astype(np.float32),
            y_train=np.ones(n, dtype=np.int64),
            X_val=np.zeros((10, 4), dtype=np.float32),
            y_val=np.ones(10, dtype=np.int64),
            model_name="xgboost",
            adapter_type="tabular",
            data_rank=2,
            feature_names=[f"f{i}" for i in range(4)],
        )
        request = TuningRequest(
            model_name="xgboost",
            horizon=5,
            prepared_data=prepared,
            n_splits=3,
            n_trials=2,
            embargo_bars=2,
        )

        result = HyperparameterTuningService().optimize(request)

        assert result.best_params == {}
        assert result.best_score == float("-inf")
        assert _StubModel.fits == []
