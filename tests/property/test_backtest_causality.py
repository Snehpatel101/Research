"""Property 5: the backtest simulator has no lookahead.

Replacing every bar (and prediction) from bar ``cut`` on with different data must not
change anything known at the close of bar ``cut - 1``: the equity curve through
``cut - 1`` and every order decided by then (direction, fill bar, barrier distance).
Both runs cover the same bars, so a bar that reads the next bar's price or prediction
sees different data in the two runs instead of running into the end of the series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.inference.backtesting.backtest import BacktestConfig, Backtester
from src.inference.backtesting.equity_curve import Trade
from tests.property.strategies import budget, build_ohlcv


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


def _perturb_from(
    prices: pd.DataFrame, preds: pd.DataFrame, cut: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same bars before ``cut``; from ``cut`` on a wilder walk and flipped predictions."""
    future = build_ohlcv(
        len(prices) - cut,
        seed=seed,
        vol=8e-3,
        start_price=float(prices["close"].iloc[cut - 1]),
    )
    prices, preds = prices.copy(), preds.copy()
    for column in ("open", "high", "low", "close", "volume"):
        prices.loc[cut:, column] = future[column].to_numpy()
    preds.loc[cut:, "prediction"] = -preds.loc[cut:, "prediction"]
    return prices, preds


def _decided_orders(
    trades: list[Trade], prices: pd.DataFrame, delay: int, cut: int
) -> list[tuple[int, int, int, float | None]]:
    """(signal bar, fill bar, direction, stop distance) of every trade signalled before cut.

    The fill price of an order decided at ``cut - 1`` may legitimately be a later bar's
    price, so fills are compared through the stop distance (ATR and cost at the signal
    bar) rather than the price itself.
    """
    bar_of = {ts: i for i, ts in enumerate(prices["timestamp"])}
    orders = []
    for trade in trades:
        fill_bar = bar_of[trade.entry_time]
        signal_bar = fill_bar - delay
        if signal_bar > cut - 1:
            continue
        stop = trade.stop_loss_price
        distance = None if stop is None else abs(trade.entry_price - stop)
        orders.append((signal_bar, fill_bar, trade.direction, distance))
    return orders


def _assert_ignores_bars_from(
    prices: pd.DataFrame, preds: pd.DataFrame, config: BacktestConfig, cut: int, seed: int
) -> None:
    other_prices, other_preds = _perturb_from(prices, preds, cut, seed)
    base = Backtester(preds, prices, config).run()
    other = Backtester(other_preds, other_prices, config).run()

    np.testing.assert_allclose(
        np.asarray(other.equity_curve.equity_values, dtype=float)[:cut],
        np.asarray(base.equity_curve.equity_values, dtype=float)[:cut],
        rtol=1e-9,
        atol=1e-6,
        err_msg="equity through cut - 1 changed when later bars changed (lookahead)",
    )
    delay = config.resolved_signal_delay
    base_orders = _decided_orders(base.trades, prices, delay, cut)
    other_orders = _decided_orders(other.trades, prices, delay, cut)
    assert [o[:3] for o in other_orders] == [
        o[:3] for o in base_orders
    ], "orders decided by cut - 1 changed when later bars changed (lookahead)"
    np.testing.assert_allclose(
        np.array([np.nan if o[3] is None else o[3] for o in other_orders], dtype=float),
        np.array([np.nan if o[3] is None else o[3] for o in base_orders], dtype=float),
        rtol=1e-9,
        err_msg="barrier distance of an order decided by cut - 1 depends on later bars",
    )


@settings(budget(15), deadline=None)
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
def test_decisions_up_to_t_ignore_later_bars(
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
    """Equity and orders through bar t are identical whatever the bars after t are."""
    prices, preds = _scenario(n_total, seed, vol, signal_prob)
    config = BacktestConfig(
        barrier_k_up=barriers[0],
        barrier_k_down=barriers[1],
        max_holding_period=max_holding,
        execution_model=execution_model,
        enable_market_hours_filter=market_hours,
        # Explicit cost, as the factory passes the labeling run's value (the test below
        # covers the derived default)
        barrier_cost_in_atr=barrier_cost,
    )
    cut = max(30, int(cut_frac * n_total))

    _assert_ignores_bars_from(prices, preds, config, cut, seed + 99)


@settings(budget(15), deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n_total=st.integers(80, 200),
    cut_frac=st.floats(0.3, 0.9),
    vol=st.sampled_from([5e-4, 2e-3, 8e-3]),
    signal_prob=st.floats(0.1, 0.6),
    execution_model=st.sampled_from(["market_on_open", "market_on_close"]),
)
def test_derived_barrier_cost_ignores_later_bars(
    seed: int, n_total: int, cut_frac: float, vol: float, signal_prob: float, execution_model: str
) -> None:
    """Default (derived) barrier cost is causal: an expanding median ATR up to the signal bar."""
    prices, preds = _scenario(n_total, seed, vol, signal_prob)
    config = BacktestConfig(
        barrier_k_up=1.5,
        barrier_k_down=1.5,
        execution_model=execution_model,
        enable_market_hours_filter=False,
    )
    cut = max(30, int(cut_frac * n_total))

    _assert_ignores_bars_from(prices, preds, config, cut, seed + 99)


def test_derived_barrier_cost_is_expanding_median() -> None:
    """The per-bar cost equals price cost / median of the valid ATR values so far."""
    from src.data.labeling.triple_barrier import expanding_cost_in_atr

    atr = np.array([np.nan, 2.0, 4.0, np.nan, 6.0, 0.0, 10.0])
    out = expanding_cost_in_atr(1.0, atr)

    np.testing.assert_allclose(
        out, [0.0, 1 / 2.0, 1 / 3.0, 1 / 3.0, 1 / 4.0, 1 / 4.0, 1 / 5.0], rtol=1e-12
    )
