"""AFML ch. 10 probability bet sizing: the formula, the sizer, and the backtester wiring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.inference.backtesting import (
    BacktestConfig,
    Backtester,
    PositionSizingMethod,
    ProbabilityBetSizer,
    afml_bet_size,
    create_position_sizer,
)

# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_size_is_zero_at_one_over_k_and_below(n_classes: int) -> None:
    assert afml_bet_size(1.0 / n_classes, n_classes) == 0.0
    assert afml_bet_size(0.5 / n_classes, n_classes) == 0.0
    assert afml_bet_size(0.0, n_classes) == 0.0


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_size_is_strictly_increasing_in_p_and_bounded(n_classes: int) -> None:
    ps = np.linspace(1.0 / n_classes + 1e-4, 0.98, 400)
    sizes = np.array([afml_bet_size(p, n_classes) for p in ps])
    assert np.all(np.diff(sizes) > 0)
    assert sizes.min() > 0.0
    assert sizes.max() < 1.0
    # towards p -> 1 the size saturates at 1 (never above)
    tail = np.array([afml_bet_size(p, n_classes) for p in np.linspace(0.98, 1.0, 50)])
    assert np.all(np.diff(tail) >= 0) and tail.max() == 1.0
    assert afml_bet_size(1.0, n_classes) == 1.0


def test_size_matches_the_closed_form() -> None:
    p, k = 0.7, 3
    z = (p - 1 / k) / math.sqrt(p * (1 - p))
    phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert afml_bet_size(p, k) == pytest.approx(2 * phi - 1)
    # binary: p = 0.5 -> 0, p = 0.9 -> known value
    assert afml_bet_size(0.5, 2) == 0.0
    assert afml_bet_size(0.9, 2) == pytest.approx(math.erf((0.4 / math.sqrt(0.09)) / math.sqrt(2)))


def test_step_size_discretizes_and_max_size_caps() -> None:
    sizes = {afml_bet_size(p, 3, step_size=0.25) for p in np.linspace(0.34, 0.999, 300)}
    assert sizes <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert {0.25, 0.5, 0.75, 1.0} <= sizes
    assert afml_bet_size(0.99, 3, max_size=0.5) == 0.5
    # the cap applies after discretization and never exceeds max_size
    assert (
        max(afml_bet_size(p, 3, step_size=0.3, max_size=0.6) for p in np.linspace(0.4, 1, 50))
        <= 0.6
    )


def test_non_finite_probability_means_no_bet() -> None:
    assert afml_bet_size(float("nan"), 3) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_classes": 1},
        {"step_size": -0.1},
        {"step_size": 1.5},
        {"max_size": 0.0},
        {"max_size": 1.5},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        afml_bet_size(0.7, **kwargs)


# ---------------------------------------------------------------------------
# The sizer
# ---------------------------------------------------------------------------


def test_contracts_are_monotone_bounded_and_zero_without_edge() -> None:
    sizer = ProbabilityBetSizer(max_contracts=10, n_classes=3)
    ps = np.linspace(0.0, 1.0, 201)
    contracts = np.array([sizer.calculate_position_size(100_000.0, probability=p) for p in ps])
    assert np.all(np.diff(contracts) >= 0)
    assert contracts.min() == 0
    assert contracts.max() == 10
    assert sizer.calculate_position_size(100_000.0, probability=1.0 / 3.0) == 0
    assert sizer.calculate_position_size(100_000.0, probability=0.34) == 0  # < half a contract


def test_contracts_scale_with_max_contracts() -> None:
    p = 0.9
    fraction = afml_bet_size(p, 3)
    for max_contracts in (1, 4, 20):
        sizer = ProbabilityBetSizer(max_contracts=max_contracts)
        assert sizer.calculate_position_size(1e5, probability=p) == int(
            math.floor(fraction * max_contracts + 0.5)
        )


def test_sizer_ignores_equity_and_extra_kwargs() -> None:
    sizer = ProbabilityBetSizer(max_contracts=6)
    a = sizer.calculate_position_size(1e3, probability=0.8, current_price=4000.0, win_rate=0.9)
    b = sizer.calculate_position_size(5e6, probability=0.8)
    assert a == b


def test_sizer_validates_construction() -> None:
    with pytest.raises(ValueError, match="max_contracts"):
        ProbabilityBetSizer(max_contracts=0)
    with pytest.raises(ValueError, match="n_classes"):
        ProbabilityBetSizer(n_classes=1)


def test_factory_function_builds_the_sizer_from_bet_kwargs() -> None:
    sizer = create_position_sizer(
        "probability", bet_max_contracts=7, bet_n_classes=2, bet_step_size=0.2
    )
    assert isinstance(sizer, ProbabilityBetSizer)
    assert (sizer.max_contracts, sizer.n_classes, sizer.step_size) == (7, 2, 0.2)
    assert PositionSizingMethod("probability") is PositionSizingMethod.PROBABILITY


# ---------------------------------------------------------------------------
# Backtester wiring
# ---------------------------------------------------------------------------


def _bars_and_signals(confidences: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A drifting price series with one long signal every 12 bars at the given confidences."""
    n = 12 * len(confidences) + 20
    ts = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = 5000.0 + np.arange(n) * 0.25
    prices = pd.DataFrame(
        {"timestamp": ts, "open": close, "high": close + 1.0, "low": close - 1.0, "close": close}
    )
    rows = [(ts[5 + 12 * i], 1, c) for i, c in enumerate(confidences)]
    # The backtester only replays bars up to the last prediction: a neutral
    # closing row keeps the last real signal's bars in range
    rows.append((ts[-1], 0, 0.5))
    signals = pd.DataFrame(rows, columns=["timestamp", "prediction", "confidence"])
    return prices, signals


def _run(confidences: list[float], **config: object) -> list:
    prices, signals = _bars_and_signals(confidences)
    cfg = BacktestConfig(
        position_sizing="probability",
        enable_market_hours_filter=False,
        max_holding_period=6,
        min_holding_period=1,
        **config,  # type: ignore[arg-type]
    )
    result = Backtester(predictions=signals, prices=prices, config=cfg).run()
    return list(result.trades)


def test_backtester_sizes_each_trade_from_its_predicted_probability() -> None:
    confidences = [0.36, 0.5, 0.7, 0.9, 0.99]
    trades = _run(confidences, bet_max_contracts=10)
    sizer = ProbabilityBetSizer(max_contracts=10)
    expected = [sizer.calculate_position_size(1e5, probability=c) for c in confidences]
    taken = [(t.confidence, t.contracts) for t in trades]
    assert [c for c, _ in taken] == [c for c, e in zip(confidences, expected, strict=True) if e > 0]
    assert [n for _, n in taken] == [e for e in expected if e > 0]
    assert len({n for _, n in taken}) > 1  # sizes actually vary with the probability


def test_backtester_takes_no_position_without_an_edge() -> None:
    assert _run([0.30, 1 / 3, 0.34]) == []
    # ... and the outcome count K moves the no-edge point: p=0.4 has an edge among 3
    # outcomes but none among 2
    assert _run([0.4], bet_max_contracts=10, bet_n_classes=3) != []
    assert _run([0.4], bet_max_contracts=10, bet_n_classes=2) == []


def test_bet_parameters_reach_the_sizer() -> None:
    config = BacktestConfig(bet_max_contracts=8, bet_n_classes=2, bet_step_size=0.5)
    prices, signals = _bars_and_signals([0.9])
    sizer = Backtester(
        predictions=signals,
        prices=prices,
        config=BacktestConfig(
            position_sizing="probability",
            bet_max_contracts=config.bet_max_contracts,
            bet_n_classes=config.bet_n_classes,
            bet_step_size=config.bet_step_size,
            enable_market_hours_filter=False,
        ),
    ).position_sizer
    assert isinstance(sizer, ProbabilityBetSizer)
    assert (sizer.max_contracts, sizer.n_classes, sizer.step_size) == (8, 2, 0.5)


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------


def test_experiment_config_accepts_probability_and_validates_sizing() -> None:
    cfg = ExperimentConfig()
    cfg.evaluation.position_sizing = "probability"
    assert cfg.validate() == []
    cfg.evaluation.position_sizing = "yolo"
    assert any("position_sizing" in issue for issue in cfg.validate())
    cfg.evaluation.position_sizing = "probability"
    cfg.evaluation.bet_max_contracts = 0
    cfg.evaluation.bet_step_size = 2.0
    issues = cfg.validate()
    assert any("bet_max_contracts" in i for i in issues)
    assert any("bet_step_size" in i for i in issues)
