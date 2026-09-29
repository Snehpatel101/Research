"""Shared test helpers (plain functions/constants; fixtures live in conftest.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.adapters.preparation import PreparedData
from src.inference.backtesting.backtest import BacktestConfig, Backtester
from src.inference.backtesting.position_sizing import BasePositionSizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_intraday_ohlcv(
    n_rows: int = 2500,
    seed: int = 11,
    freq: str = "5min",
    index_name: str | None = "datetime",
) -> pd.DataFrame:
    """Synthetic intraday OHLCV: random-walk close, high/low bracket open and close.

    Shared by the factory / end-to-end tests so they all run on the same kind of
    bars. Same ``(n_rows, seed)`` always yields the same frame.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-02 09:30", periods=n_rows, freq=freq)
    close = 5000.0 + np.cumsum(rng.normal(0, 2.0, n_rows))
    open_ = np.roll(close, 1) + rng.normal(0, 0.5, n_rows)
    open_[0] = close[0]
    eps = np.abs(rng.normal(0, 0.5, n_rows)) + 0.25
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + eps,
            "low": np.minimum(open_, close) - eps,
            "close": close,
            "volume": rng.randint(100, 5000, n_rows).astype(float),
        },
        index=idx,
    )
    df.index.name = index_name
    return df


def tiny_prepared_data(n_train: int = 120, n_val: int = 40, n_features: int = 4) -> PreparedData:
    """Build a tiny 2D PreparedData for xgboost-style tabular training."""
    rng = np.random.RandomState(42)
    return PreparedData(
        X_train=rng.normal(size=(n_train, n_features)).astype(np.float32),
        y_train=rng.choice([-1, 0, 1], size=n_train).astype(np.int64),
        X_val=rng.normal(size=(n_val, n_features)).astype(np.float32),
        y_val=rng.choice([-1, 0, 1], size=n_val).astype(np.int64),
        model_name="xgboost",
        adapter_type="tabular",
        data_rank=2,
        feature_names=[f"f{i}" for i in range(n_features)],
    )


def make_minimal_backtester(
    position_sizer: BasePositionSizer | None = None, **config_overrides: Any
) -> Backtester:
    """A two-bar Backtester for unit-testing single methods (never ``run()``)."""
    defaults: dict[str, Any] = {
        "enable_market_hours_filter": False,
        "slippage_ticks": 0.0,
        "tick_size": 0.25,
        "commission_per_contract": 0.0,
    }
    defaults.update(config_overrides)
    ts = pd.date_range("2024-01-01", periods=2, freq="h")
    prices = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
        }
    )
    preds = pd.DataFrame({"timestamp": ts, "prediction": [0, 0], "confidence": [1.0, 1.0]})
    return Backtester(
        predictions=preds,
        prices=prices,
        config=BacktestConfig(**defaults),
        position_sizer=position_sizer,
    )


def make_signal_features(
    n_samples: int = 500,
    n_features: int = 10,
    seed: int = 42,
    informative: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Gaussian features ``feat_0..`` whose binary label is the sign of the first ``informative``."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    signal = X.iloc[:, :informative].sum(axis=1)
    y = pd.Series((signal > 0).astype(int), name="label")
    return X, y
