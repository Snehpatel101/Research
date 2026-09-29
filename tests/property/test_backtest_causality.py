"""Property 5: the backtest simulator has no lookahead.

Appending future bars (and predictions) to a series must not change the equity curve
of the bars that were already there. The last bar of the shorter run is excluded: a run
always force-closes an open position on its final bar, which is a property of the run's
end, not of the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.inference.backtesting.backtest import BacktestConfig, Backtester
from tests.property.strategies import build_ohlcv


def _scenario(
    n_total: int, seed: int, vol: float, signal_prob: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices (with a ``timestamp`` column) and random long/short/flat predictions."""
    bars = build_ohlcv(n_total, seed=seed, vol=vol)
    prices = bars.reset_index().rename(columns={"datetime": "timestamp"})
    rng = np.random.default_rng(seed + 1)
    active = rng.random(n_total) < signal_prob
    preds = pd.DataFrame(
        {
            "timestamp": prices["timestamp"],
            "prediction": np.where(active, rng.choice([-1, 1], n_total), 0),
            "confidence": rng.uniform(0.5, 1.0, n_total),
        }
    )
    return prices, preds


def _equity(
    prices: pd.DataFrame, preds: pd.DataFrame, config: BacktestConfig, n: int
) -> np.ndarray:
    """Equity curve (one value per bar) of a backtest over the first n bars."""
    result = Backtester(preds.iloc[:n], prices.iloc[:n], config).run()
    return np.asarray(result.equity_curve.equity_values, dtype=float)


@settings(max_examples=15, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n_total=st.integers(80, 200),
    cut_frac=st.floats(0.3, 0.9),
    vol=st.sampled_from([5e-4, 2e-3, 8e-3]),
    signal_prob=st.floats(0.05, 0.6),
    barriers=st.sampled_from([(0.0, 0.0), (1.5, 1.5), (2.0, 1.0)]),
    barrier_cost=st.sampled_from([0.0, 0.3]),
    max_holding=st.sampled_from([0, 5, 20]),
    execution_model=st.sampled_from(["market_on_open", "market_on_close"]),
    market_hours=st.booleans(),
)
def test_equity_up_to_t_ignores_appended_future_bars(
    seed: int,
    n_total: int,
    cut_frac: float,
    vol: float,
    signal_prob: float,
    barriers: tuple[float, float],
    barrier_cost: float,
    max_holding: int,
    execution_model: str,
    market_hours: bool,
) -> None:
    """Equity at every bar before the cut is identical with or without the future."""
    prices, preds = _scenario(n_total, seed, vol, signal_prob)
    config = BacktestConfig(
        barrier_k_up=barriers[0],
        barrier_k_down=barriers[1],
        max_holding_period=max_holding,
        execution_model=execution_model,
        enable_market_hours_filter=market_hours,
        # The factory passes the labeling run's cost explicitly (see the xfail test below
        # for the derive-from-prices default)
        barrier_cost_in_atr=barrier_cost,
    )
    cut = max(30, int(cut_frac * n_total))

    short = _equity(prices, preds, config, cut)
    full = _equity(prices, preds, config, n_total)

    assert len(short) == cut
    np.testing.assert_allclose(
        full[: cut - 1],
        short[: cut - 1],
        rtol=1e-9,
        atol=1e-6,
        err_msg="equity before the cut changed when future bars were appended (lookahead)",
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Backtester derives barrier_cost_in_atr (barrier_cost_in_atr=None) from the MEDIAN ATR "
        "of the WHOLE price series, so appending bars shifts the stop/take-profit distances of "
        "earlier trades. Mild global-calibration lookahead in the default; the factory passes "
        "the labeling run's cost explicitly and is unaffected."
    ),
)
def test_derived_barrier_cost_ignores_appended_future_bars() -> None:
    """Default (derived) barrier cost: equity before the cut must not depend on later bars."""
    prices, preds = _scenario(n_total=80, seed=0, vol=5e-4, signal_prob=0.5)
    config = BacktestConfig(barrier_k_up=1.5, barrier_k_down=1.5, enable_market_hours_filter=False)
    short = _equity(prices, preds, config, 40)
    full = _equity(prices, preds, config, 80)
    np.testing.assert_allclose(full[:39], short[:39], rtol=1e-9, atol=1e-6)
