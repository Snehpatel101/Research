"""
Regression tests for label/backtest barrier parity.

MLFactory._resolve_barrier_params() is the single source of truth for
triple-barrier parameters, shared by labeling (_run_data_pipeline) and the
backtester (_run_evaluation). These tests pin down:

1. Default LabelingConfig (all None) resolves to the per-symbol/per-horizon
   BARRIER_PARAMS table (same values get_barrier_params returns).
2. Explicit LabelingConfig overrides win per-field, and the source string
   reports the override.
3. Labeling and backtest wire the exact same values (helper is deterministic;
   TripleBarrierConfig built the way the factory builds it carries symbol and
   horizon=max_bars from the table, not the prediction horizon).
4. Cost parity: TripleBarrierLabeler._calculate_cost_in_atr uses per-symbol
   costs from barriers_config (MGC vs MES differ by the expected ratio).
5. PipelineConfig/LabelingConfig barrier fields default to None ("auto") and
   LabelingConfig.validate() accepts None but rejects non-positive values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.data import LabelingConfig
from src.config.experiment import ExperimentConfig
from src.core.config import PipelineConfig
from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler
from src.data.pipeline.config.barriers_config import (
    BARRIER_PARAMS,
    BARRIER_PARAMS_DEFAULT,
    TICK_VALUES,
    TRANSACTION_COSTS,
    get_barrier_params,
    get_tick_value,
    get_total_trade_cost,
)
from src.factory import MLFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factory(tmp_path, symbol: str, horizons: list[int]) -> MLFactory:
    """Build an MLFactory with a default (None-auto) LabelingConfig."""
    cfg = ExperimentConfig()
    cfg.data.symbol = symbol
    cfg.training.horizons = list(horizons)
    cfg.training.optuna.n_trials = 0
    cfg.output_dir = tmp_path / f"run_{symbol.lower()}"
    return MLFactory(cfg, verbose=0, enable_checkpoints=False)


def _synthetic_ohlcv_with_atr(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """Tiny synthetic OHLCV at 5min freq with a pre-computed atr_14 column."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:30", periods=n_bars, freq="5min")

    close = 2000.0 + np.cumsum(rng.normal(0, 1.0, n_bars))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    eps = np.abs(rng.normal(0, 0.25, n_bars)) + 0.05
    high = np.maximum(open_, close) + eps
    low = np.minimum(open_, close) - eps
    volume = rng.integers(100, 1000, n_bars).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    # Simple ATR(14): rolling mean of true range (positive by construction)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=1).mean()
    return df


# ---------------------------------------------------------------------------
# 1. Default LabelingConfig -> BARRIER_PARAMS table values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "horizon"),
    [("MES", 5), ("MGC", 5), ("MNQ", 20)],
)
def test_resolve_barrier_params_defaults_match_table(tmp_path, symbol, horizon):
    """None-auto LabelingConfig must resolve to get_barrier_params table values."""
    factory = _make_factory(tmp_path, symbol, [horizon])

    # Preconditions: config really is None-auto
    labeling = factory.config.data.labeling
    assert labeling.upper_mult is None
    assert labeling.lower_mult is None
    assert labeling.max_holding_bars is None

    k_up, k_down, max_bars, source = factory._resolve_barrier_params(horizon)
    table = get_barrier_params(symbol, horizon)

    assert k_up == pytest.approx(float(table["k_up"]))
    assert k_down == pytest.approx(float(table["k_down"]))
    assert max_bars == int(table["max_bars"])
    assert source == "barriers table"


def test_table_routing_symbol_specific_vs_default():
    """MES/MGC have symbol-specific entries; MNQ falls back to defaults."""
    assert get_barrier_params("MES", 5) is BARRIER_PARAMS["MES"][5]
    assert get_barrier_params("MGC", 5) is BARRIER_PARAMS["MGC"][5]
    # MNQ has no BARRIER_PARAMS entry -> BARRIER_PARAMS_DEFAULT
    assert "MNQ" not in BARRIER_PARAMS
    assert get_barrier_params("MNQ", 20) is BARRIER_PARAMS_DEFAULT[20]


# ---------------------------------------------------------------------------
# 2. Explicit override wins per-field, source mentions override
# ---------------------------------------------------------------------------


def test_explicit_upper_mult_override_is_per_field(tmp_path):
    """upper_mult=2.5 overrides k_up only; k_down/max_bars stay table-resolved."""
    factory = _make_factory(tmp_path, "MES", [5])
    factory.config.data.labeling.upper_mult = 2.5

    k_up, k_down, max_bars, source = factory._resolve_barrier_params(5)
    table = get_barrier_params("MES", 5)

    assert k_up == pytest.approx(2.5)
    assert k_down == pytest.approx(float(table["k_down"]))
    assert max_bars == int(table["max_bars"])
    assert "override" in source.lower()


# ---------------------------------------------------------------------------
# 3. Labeling and backtest play the same game
# ---------------------------------------------------------------------------


def test_labeling_and_backtest_use_same_resolved_params(tmp_path):
    """Both sides call the same helper — it must be deterministic, and the
    TripleBarrierConfig built the way the factory builds it must carry
    symbol='MES' and horizon=max_bars from the table (not the prediction
    horizon)."""
    factory = _make_factory(tmp_path, "MES", [5, 10])

    # Deterministic: repeated calls return the exact same tuple
    first = factory._resolve_barrier_params(5)
    second = factory._resolve_barrier_params(5)
    assert first == second

    # Backtest side resolves the FIRST horizon — identical to labeling side
    first_horizon = factory.config.training.horizons[0]
    assert factory._resolve_barrier_params(first_horizon) == first

    # Build TripleBarrierConfig exactly like _run_data_pipeline does
    k_up, k_down, max_bars, _src = first
    labeling = factory.config.data.labeling
    label_config = TripleBarrierConfig(
        horizon=max_bars,
        upper_mult=k_up,
        lower_mult=k_down,
        atr_period=labeling.atr_period,
        atr_column=f"atr_{labeling.atr_period}",
        symbol=factory.config.data.symbol.upper(),
    )

    table = get_barrier_params("MES", 5)
    assert label_config.symbol == "MES"
    # Time barrier is max_bars from the table (12), NOT the prediction horizon (5)
    assert label_config.horizon == int(table["max_bars"])
    assert label_config.horizon != 5
    assert label_config.upper_mult == pytest.approx(float(table["k_up"]))
    assert label_config.lower_mult == pytest.approx(float(table["k_down"]))


# ---------------------------------------------------------------------------
# 4. Cost parity: labeling cost calc uses per-symbol costs
# ---------------------------------------------------------------------------


def test_cost_in_atr_uses_per_symbol_costs():
    """_calculate_cost_in_atr for MGC vs MES must differ by the exact ratio of
    (total_trade_cost * tick_value) from barriers_config."""
    # Table sanity (values documented in barriers_config.py)
    assert TRANSACTION_COSTS["MES"] == pytest.approx(2.43)
    assert TRANSACTION_COSTS["MGC"] == pytest.approx(3.04)
    assert TICK_VALUES["MES"] == pytest.approx(1.25)
    assert TICK_VALUES["MGC"] == pytest.approx(1.00)

    df = _synthetic_ohlcv_with_atr(n_bars=300)
    atr = df["atr_14"].to_numpy()
    median_atr = float(np.median(atr[~np.isnan(atr) & (atr > 0)]))

    costs = {}
    for symbol in ("MES", "MGC"):
        config = TripleBarrierConfig(
            upper_mult=1.5,
            lower_mult=1.0,
            horizon=12,
            atr_column="atr_14",
            symbol=symbol,
        )
        labeler = TripleBarrierLabeler(config)
        costs[symbol] = labeler._calculate_cost_in_atr(atr)

    # Each symbol: cost_in_atr = total_trade_cost(ticks) * tick_value / median_atr
    for symbol in ("MES", "MGC"):
        expected = get_total_trade_cost(symbol, "low_vol") * get_tick_value(symbol) / median_atr
        assert costs[symbol] == pytest.approx(expected, rel=1e-9), symbol

    # Symbols must actually differ, by the exact price-cost ratio
    assert costs["MGC"] != pytest.approx(costs["MES"])
    expected_ratio = (get_total_trade_cost("MGC", "low_vol") * get_tick_value("MGC")) / (
        get_total_trade_cost("MES", "low_vol") * get_tick_value("MES")
    )
    assert costs["MGC"] / costs["MES"] == pytest.approx(expected_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. None-auto defaults on LabelingConfig and PipelineConfig
# ---------------------------------------------------------------------------


def test_labeling_config_none_auto_defaults_and_validation():
    """Defaults are None (= auto from BARRIER_PARAMS); validate() accepts None
    and rejects non-positive explicit values."""
    cfg = LabelingConfig()
    assert cfg.upper_mult is None
    assert cfg.lower_mult is None
    assert cfg.max_holding_bars is None
    assert cfg.validate() == []

    # Negative / zero explicit values are rejected per-field
    bad = LabelingConfig(upper_mult=-1.0)
    assert any("upper_mult" in issue for issue in bad.validate())

    bad = LabelingConfig(lower_mult=-2.0)
    assert any("lower_mult" in issue for issue in bad.validate())

    bad = LabelingConfig(max_holding_bars=-5)
    assert any("max_holding_bars" in issue for issue in bad.validate())

    bad = LabelingConfig(upper_mult=0.0)
    assert any("upper_mult" in issue for issue in bad.validate())

    # Valid explicit values pass
    good = LabelingConfig(upper_mult=2.0, lower_mult=1.5, max_holding_bars=20)
    assert good.validate() == []


def test_pipeline_config_none_auto_defaults(tmp_path):
    """PipelineConfig barrier overrides also default to None (= auto)."""
    cfg = PipelineConfig(
        symbol="MES",
        data_path=tmp_path / "data.parquet",
        output_dir=tmp_path / "out",
    )
    assert cfg.upper_mult is None
    assert cfg.lower_mult is None
    assert cfg.max_holding_bars is None
