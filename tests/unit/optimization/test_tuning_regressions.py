"""
Hyperparameter-tuning regressions: ATR parity, temporal subsampling, embargo wiring.

- The Optuna objective's ATR is Wilder's EMA (alpha = 1/period), like labeling and backtest.
- Above ``max_samples`` the tuner subsamples on a fixed stride (temporal order kept,
  embargo scaled) instead of randomly.
- ``HyperparameterTuningService`` builds its CV from the pipeline embargo, falling back
  to a floor derived from the horizon only when none is given.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.models.training.services.hyperparameter_tuning as hp_tuning
import src.validation.cv.cv_tuner as cv_tuner
from src.optimization.five_dimension_objective import _compute_atr
from src.validation.cv import PurgedKFold, PurgedKFoldConfig, TimeSeriesOptunaTuner
from tests.helpers import tiny_prepared_data

# ---------------------------------------------------------------------------
# Objective ATR
# ---------------------------------------------------------------------------


def _ewm_atr(tr: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros(len(tr))
    out[0] = tr[0]
    for i in range(1, len(tr)):
        out[i] = alpha * tr[i] + (1 - alpha) * out[i - 1]
    return out


def test_objective_atr_is_wilders_ema_not_standard_ema() -> None:
    rng = np.random.default_rng(42)
    n, period = 100, 14
    close = 100.0 + np.cumsum(rng.normal(size=n) * 0.5)
    high = close + np.abs(rng.normal(size=n) * 0.3)
    low = close - np.abs(rng.normal(size=n) * 0.3)
    df = pd.DataFrame({"high": high, "low": low, "close": close})

    atr = _compute_atr(df, period=period)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    np.testing.assert_allclose(atr, _ewm_atr(tr, 1.0 / period), rtol=1e-10)
    # The standard EMA (alpha = 2 / (period + 1)) would be a materially different series
    assert not np.allclose(atr, _ewm_atr(tr, 2.0 / (period + 1)), rtol=1e-6)


# ---------------------------------------------------------------------------
# Strided subsampling
# ---------------------------------------------------------------------------


class _RowRecorder:
    """Fold model that records which original rows (feature 0 = row id) it sees."""

    seen: list[np.ndarray] = []

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):  # noqa: ANN001
        _RowRecorder.seen += [np.asarray(X_train)[:, 0], np.asarray(X_val)[:, 0]]

    def predict(self, X):  # noqa: ANN001
        _RowRecorder.seen.append(np.asarray(X)[:, 0])
        return SimpleNamespace(class_predictions=np.zeros(len(X), dtype=int))


@pytest.fixture
def row_recorder(monkeypatch: pytest.MonkeyPatch) -> type[_RowRecorder]:
    _RowRecorder.seen = []
    registry = SimpleNamespace(create=lambda name, config: _RowRecorder())
    monkeypatch.setattr(cv_tuner, "ModelRegistry", registry)
    return _RowRecorder


def _tune_big_frame(n: int, max_samples: int, embargo: int) -> tuple[TimeSeriesOptunaTuner, Any]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"row": np.arange(n, dtype=float), "noise": rng.normal(size=n)})
    y = pd.Series(rng.integers(-1, 2, size=n))
    cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=0, embargo_bars=embargo))
    tuner = TimeSeriesOptunaTuner("xgboost", cv, n_trials=1, max_samples=max_samples)
    tuner.tune(X, y)
    return tuner, cv


class TestStridedSubsampling:
    def test_rows_above_cap_are_sampled_on_a_fixed_stride(self, row_recorder) -> None:
        n, max_samples = 1000, 250  # stride 4
        _tune_big_frame(n, max_samples, embargo=8)

        rows = np.unique(np.concatenate(row_recorder.seen))
        assert len(rows) <= max_samples
        assert np.all(rows % 4 == 0), "random subsampling would not stay on the stride grid"
        assert np.all(np.diff(rows) == 4), "no gaps: the subsample spans the series uniformly"
        assert rows.max() >= n - 4, "the tail of the series must not be dropped"

    def test_rows_below_cap_are_all_used(self, row_recorder) -> None:
        _tune_big_frame(300, max_samples=1000, embargo=8)

        rows = np.unique(np.concatenate(row_recorder.seen))
        np.testing.assert_array_equal(rows, np.arange(300))

    def test_embargo_shrinks_with_the_stride(self, row_recorder) -> None:
        """One subsampled row spans `stride` bars, so the embargo is divided by it."""
        _, cv = _tune_big_frame(1000, max_samples=250, embargo=8)
        assert cv.config.embargo_bars == 2


# ---------------------------------------------------------------------------
# Embargo wiring in HyperparameterTuningService
# ---------------------------------------------------------------------------


class _CvCapturingTuner:
    captured: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        _CvCapturingTuner.captured = dict(kwargs)

    def tune(self, X, y, **kwargs):  # noqa: ANN001, ANN003
        return {"best_params": {}, "best_value": 0.5}


def _embargo_used(monkeypatch: pytest.MonkeyPatch, horizon: int, embargo_bars: int | None) -> int:
    monkeypatch.setattr(hp_tuning, "TimeSeriesOptunaTuner", _CvCapturingTuner)
    request = hp_tuning.TuningRequest(
        model_name="xgboost",
        horizon=horizon,
        prepared_data=tiny_prepared_data(),
        n_splits=2,
        n_trials=1,
        embargo_bars=embargo_bars,
    )
    hp_tuning.HyperparameterTuningService().optimize(request)
    return _CvCapturingTuner.captured["cv"].config.embargo_bars


class TestTuningEmbargo:
    def test_pipeline_embargo_wins_over_horizon_default(self, monkeypatch) -> None:
        assert _embargo_used(monkeypatch, horizon=10, embargo_bars=42) == 42

    def test_zero_pipeline_embargo_is_respected(self, monkeypatch) -> None:
        """An explicit 0 must not be treated as 'missing' and replaced by the fallback."""
        assert _embargo_used(monkeypatch, horizon=10, embargo_bars=0) == 0

    def test_fallback_is_at_least_sixty_bars(self, monkeypatch) -> None:
        # labels can resolve up to the barrier table's max_bars, so short horizons get a floor
        assert _embargo_used(monkeypatch, horizon=10, embargo_bars=None) == 60

    def test_fallback_scales_with_long_horizons(self, monkeypatch) -> None:
        assert _embargo_used(monkeypatch, horizon=50, embargo_bars=None) == 100
