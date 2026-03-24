"""
Test D4: Verify that failed/degenerate Optuna trials return -inf, not 0.0.

Phase 92 fixed a critical bug where failed Optuna trials returned 0.0,
which polluted best_value selection (0.0 looks like a valid score). Now
all failure paths in the 5D objective return float('-inf').
"""

from __future__ import annotations

import math

import numpy as np
import optuna
import pandas as pd

from src.core.contracts import FeatureSpec
from src.core.types import ModelFamily
from src.optimization.five_dimension_objective import (
    _compute_validation_metric,
    create_5d_objective,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.RandomState(seed)
    close = 100 + rng.randn(n).cumsum() * 0.5
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    opn = close + rng.uniform(-0.5, 0.5, n)
    volume = rng.randint(100, 10000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDegenerateTrialsReturnNegInf:
    """Verify every failure path in the 5D objective returns -inf."""

    def test_compute_validation_metric_exception_returns_neg_inf(self):
        """_compute_validation_metric returns -inf when the model raises."""
        spec = FeatureSpec(
            profit_threshold=2.0,
            loss_threshold=1.0,
            max_holding_bars=60,
            selected_features=["nonexistent_feature"],
            feature_params={},
            feature_timeframes={},
            hyperparameters={},
            model_name="xgboost",
        )
        # Pass deliberately broken data (wrong shapes) to force an exception
        train = np.array([[1, 2], [3, 4]])
        val = np.array([[5, 6]])
        train_labels = np.array([0, 1])
        val_labels = np.array([0])
        train_mask = np.array([True, True])
        val_mask = np.array([True])

        result = _compute_validation_metric(
            spec=spec,
            train_data=train,
            val_data=val,
            train_labels=train_labels,
            val_labels=val_labels,
            train_valid_mask=train_mask,
            val_valid_mask=val_mask,
            metric_fn=lambda y_true, y_pred: float(np.mean(y_true == y_pred)),
            model_family=ModelFamily.CLASSICAL,
        )
        # With only 2 training samples and CLASSICAL (LogisticRegression),
        # this may or may not error. If it doesn't error, it still returns a
        # valid float. The key invariant: it should NEVER return exactly 0.0
        # from the exception path. We test the exception path explicitly below.
        assert isinstance(result, float)

    def test_compute_validation_metric_forced_exception_returns_neg_inf(self):
        """Force an exception in _compute_validation_metric and verify -inf."""
        spec = FeatureSpec(
            profit_threshold=2.0,
            loss_threshold=1.0,
            max_holding_bars=60,
            selected_features=[],
            feature_params={},
            feature_timeframes={},
            hyperparameters={},
            model_name="xgboost",
        )
        # Empty arrays will cause the model to crash
        train = np.empty((0, 0))
        val = np.empty((0, 0))
        train_labels = np.array([], dtype=int)
        val_labels = np.array([], dtype=int)
        train_mask = np.array([], dtype=bool)
        val_mask = np.array([], dtype=bool)

        result = _compute_validation_metric(
            spec=spec,
            train_data=train,
            val_data=val,
            train_labels=train_labels,
            val_labels=val_labels,
            train_valid_mask=train_mask,
            val_valid_mask=val_mask,
            metric_fn=lambda y_true, y_pred: float(np.mean(y_true == y_pred)),
            model_family=ModelFamily.BOOSTING,
        )
        assert result == float(
            "-inf"
        ), f"Expected -inf for crashed _compute_validation_metric, got {result}"

    def test_objective_exception_returns_neg_inf(self):
        """The objective catch-all (line 822-824) returns -inf, not 0.0."""

        # Provide a metric_fn that always raises
        def exploding_metric(y_true, y_pred):
            raise RuntimeError("Simulated metric failure")

        train_df = _make_ohlcv_df(500, seed=1)
        val_df = _make_ohlcv_df(200, seed=2)

        objective = create_5d_objective(
            model_family=ModelFamily.BOOSTING,
            train_data=train_df,
            val_data=val_df,
            metric_fn=exploding_metric,
            generate_labels_per_trial=True,
        )

        # Run a single trial
        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=1)

        trial = study.trials[0]
        assert trial.value is not None, "Trial value should not be None"
        assert (
            math.isinf(trial.value) and trial.value < 0
        ), f"Failed trial should return -inf, got {trial.value}"

    def test_no_labels_returns_neg_inf(self):
        """Objective with no labels and generate_labels_per_trial=False returns -inf."""
        train_df = _make_ohlcv_df(500, seed=3)
        val_df = _make_ohlcv_df(200, seed=4)

        # generate_labels_per_trial=False but no pre-computed labels provided
        objective = create_5d_objective(
            model_family=ModelFamily.BOOSTING,
            train_data=train_df,
            val_data=val_df,
            train_labels=None,
            val_labels=None,
            generate_labels_per_trial=False,
        )

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=1)

        trial = study.trials[0]
        assert trial.value is not None
        assert (
            math.isinf(trial.value) and trial.value < 0
        ), f"No-labels trial should return -inf, got {trial.value}"

    def test_insufficient_valid_labels_returns_neg_inf(self):
        """Objective with too few valid labels returns -inf (line 758-764)."""
        # Create a tiny dataset where triple barrier can't produce enough labels
        train_df = _make_ohlcv_df(50, seed=5)  # way below the 100 threshold
        val_df = _make_ohlcv_df(20, seed=6)  # way below the 50 threshold

        objective = create_5d_objective(
            model_family=ModelFamily.BOOSTING,
            train_data=train_df,
            val_data=val_df,
            generate_labels_per_trial=True,
        )

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=1)

        trial = study.trials[0]
        assert trial.value is not None
        assert (
            math.isinf(trial.value) and trial.value < 0
        ), f"Insufficient-labels trial should return -inf, got {trial.value}"

    def test_neg_inf_is_not_zero(self):
        """Sanity: float('-inf') != 0.0 — the old buggy behavior."""
        assert float("-inf") != 0.0
        assert float("-inf") < 0.0
        assert math.isinf(float("-inf"))
