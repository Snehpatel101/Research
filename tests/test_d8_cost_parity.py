"""
Test D8: Transaction cost parity between labeling and backtesting.

Verifies that the cost parameters used during triple-barrier labeling
(barriers_config.py) are consistent with the cost parameters used during
backtesting (costs.py TransactionCosts). A mismatch here means the model
trains on labels that assume one cost regime but gets evaluated on another,
causing H-BT-2 (cost gap between train and eval).
"""

import pytest

from src.config.symbol import SymbolConfig
from src.data.pipeline.config.barriers_config import (
    SLIPPAGE_TICKS,
    TICK_VALUES,
    TRANSACTION_COSTS,
    get_tick_value,
    get_total_trade_cost,
)
from src.inference.backtesting.costs import TransactionCosts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backtest_costs(symbol: str) -> TransactionCosts:
    """Get TransactionCosts for a symbol using the factory method."""
    factories = {
        "MES": TransactionCosts.for_mes,
        "MGC": TransactionCosts.for_mgc,
        "MNQ": TransactionCosts.for_mnq,
    }
    return factories[symbol]()


def _symbol_config(symbol: str) -> SymbolConfig:
    """Get SymbolConfig for a symbol."""
    return SymbolConfig.from_symbol(symbol)


# ---------------------------------------------------------------------------
# Tests — tick value parity
# ---------------------------------------------------------------------------

SYMBOLS = ["MES", "MGC", "MNQ"]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_tick_value_parity(symbol: str) -> None:
    """Tick value must agree across barriers_config, costs.py, and SymbolConfig."""
    labeling_tv = get_tick_value(symbol)
    backtest_tv = _backtest_costs(symbol).tick_value
    config_tv = _symbol_config(symbol).tick_value

    assert labeling_tv == pytest.approx(backtest_tv), (
        f"{symbol}: labeling tick_value ({labeling_tv}) != " f"backtest tick_value ({backtest_tv})"
    )
    assert labeling_tv == pytest.approx(config_tv), (
        f"{symbol}: labeling tick_value ({labeling_tv}) != "
        f"SymbolConfig tick_value ({config_tv})"
    )


# ---------------------------------------------------------------------------
# Tests — tick size parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_tick_size_parity(symbol: str) -> None:
    """Tick size must agree between costs.py and SymbolConfig."""
    backtest_ts = _backtest_costs(symbol).tick_size
    config_ts = _symbol_config(symbol).tick_size

    assert backtest_ts == pytest.approx(config_ts), (
        f"{symbol}: backtest tick_size ({backtest_ts}) != " f"SymbolConfig tick_size ({config_ts})"
    )


# ---------------------------------------------------------------------------
# Tests — fixed commission parity (dollar amount)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_fixed_commission_parity(symbol: str) -> None:
    """
    Fixed commission (commission + exchange_fee + nfa_fee) in dollars must
    match barriers_config TRANSACTION_COSTS (in ticks) * tick_value.
    """
    bt = _backtest_costs(symbol)
    backtest_fixed_dollars = bt.total_fixed_cost_per_contract

    labeling_commission_ticks = TRANSACTION_COSTS[symbol]
    labeling_tick_value = TICK_VALUES[symbol]
    labeling_fixed_dollars = labeling_commission_ticks * labeling_tick_value

    assert labeling_fixed_dollars == pytest.approx(backtest_fixed_dollars, abs=0.02), (
        f"{symbol}: labeling fixed cost ${labeling_fixed_dollars:.4f} "
        f"({labeling_commission_ticks} ticks * ${labeling_tick_value}/tick) != "
        f"backtest fixed cost ${backtest_fixed_dollars:.4f} "
        f"(commission=${bt.commission_per_contract} + "
        f"exchange=${bt.exchange_fee} + nfa=${bt.nfa_fee})"
    )


# ---------------------------------------------------------------------------
# Tests — slippage parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_slippage_ticks_parity(symbol: str) -> None:
    """
    Low-vol slippage ticks in labeling must match backtest slippage_ticks.

    The backtest TransactionCosts.slippage_ticks is a per-fill value.
    The labeling SLIPPAGE_TICKS[symbol]["low_vol"] is also per-fill.
    """
    bt = _backtest_costs(symbol)
    labeling_slippage = SLIPPAGE_TICKS[symbol]["low_vol"]

    assert labeling_slippage == pytest.approx(bt.slippage_ticks), (
        f"{symbol}: labeling slippage ({labeling_slippage} ticks/fill) != "
        f"backtest slippage ({bt.slippage_ticks} ticks/fill)"
    )


# ---------------------------------------------------------------------------
# Tests — total round-trip cost parity (dollars)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_total_round_trip_cost_dollars(symbol: str) -> None:
    """
    Total round-trip cost in dollars must be consistent.

    Labeling: (TRANSACTION_COSTS[ticks] + 2 * SLIPPAGE_TICKS[low_vol]) * tick_value
    Backtest: total_fixed_cost + 2 * slippage_ticks * tick_value
    """
    bt = _backtest_costs(symbol)
    backtest_total = bt.total_cost_per_contract

    labeling_total_ticks = get_total_trade_cost(symbol, regime="low_vol")
    labeling_tick_value = get_tick_value(symbol)
    labeling_total_dollars = labeling_total_ticks * labeling_tick_value

    assert labeling_total_dollars == pytest.approx(backtest_total, abs=0.05), (
        f"{symbol}: labeling total ${labeling_total_dollars:.4f} "
        f"({labeling_total_ticks:.2f} ticks * ${labeling_tick_value}/tick) != "
        f"backtest total ${backtest_total:.4f}"
    )


# ---------------------------------------------------------------------------
# Tests — all supported symbols are present in both configs
# ---------------------------------------------------------------------------


def test_all_symbols_present_in_labeling() -> None:
    """Every symbol with a backtest factory must have labeling cost entries."""
    for symbol in SYMBOLS:
        assert (
            symbol in TRANSACTION_COSTS
        ), f"{symbol} missing from barriers_config.TRANSACTION_COSTS"
        assert symbol in SLIPPAGE_TICKS, f"{symbol} missing from barriers_config.SLIPPAGE_TICKS"
        assert symbol in TICK_VALUES, f"{symbol} missing from barriers_config.TICK_VALUES"


def test_all_symbols_present_in_symbol_config() -> None:
    """Every symbol with costs must have a SymbolConfig preset."""
    for symbol in SYMBOLS:
        cfg = SymbolConfig.from_symbol(symbol)
        assert cfg.symbol == symbol


# ---------------------------------------------------------------------------
# Tests — cost ratio sanity check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_cost_ratio_within_bounds(symbol: str) -> None:
    """
    The ratio of labeling total cost to backtest total cost must be ~1.0.

    A ratio > 1.5 or < 0.67 indicates a significant cost gap (H-BT-2).
    """
    bt = _backtest_costs(symbol)
    backtest_total = bt.total_cost_per_contract

    labeling_total_ticks = get_total_trade_cost(symbol, regime="low_vol")
    labeling_total_dollars = labeling_total_ticks * get_tick_value(symbol)

    if backtest_total > 0:
        ratio = labeling_total_dollars / backtest_total
        assert 0.67 < ratio < 1.5, (
            f"{symbol}: cost ratio {ratio:.3f} out of bounds [0.67, 1.5] — "
            f"labeling=${labeling_total_dollars:.4f} vs backtest=${backtest_total:.4f}"
        )
