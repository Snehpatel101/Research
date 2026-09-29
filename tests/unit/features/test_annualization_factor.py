"""
Volatility annualization is derived from the bar timeframe, not hardcoded.

factor = sqrt(bars_per_day * 252): a 5-minute regular session has 78 bars/day.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.pipeline.stages.features.constants import (
    ANNUALIZATION_FACTOR,
    get_annualization_factor,
    get_bars_per_day,
)


def test_five_minute_factor_uses_78_bars_per_day() -> None:
    assert get_bars_per_day("5min") == pytest.approx(78.0)
    assert get_annualization_factor("5min") == pytest.approx(np.sqrt(252 * 78), abs=1e-10)


def test_factor_varies_with_timeframe() -> None:
    five, one, fifteen = (get_annualization_factor(tf) for tf in ("5min", "1min", "15min"))

    assert one == pytest.approx(np.sqrt(252 * 390), abs=1e-10)
    assert one > five > fifteen


def test_extended_session_gives_a_larger_factor() -> None:
    assert get_annualization_factor("5min", extended_hours=True) > get_annualization_factor("5min")


def test_module_default_is_the_five_minute_regular_session_factor() -> None:
    assert pytest.approx(get_annualization_factor("5min"), abs=1e-10) == ANNUALIZATION_FACTOR


def test_unknown_timeframe_is_rejected_not_defaulted() -> None:
    with pytest.raises(ValueError):
        get_annualization_factor("13min")
