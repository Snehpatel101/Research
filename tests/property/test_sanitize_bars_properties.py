"""Property 8: sanitize_bars turns arbitrary messy bars into clean, consistent OHLCV.

Whatever the input (shuffled, duplicated, NaT / NaN / inf / negative / zero / junk values,
tz-aware or naive timestamps), a frame that comes out is sorted, has unique naive
timestamps, positive finite prices with high/low bracketing open/close, and finite
non-negative volume; if nothing valid is left it raises ValueError instead.
"""

from __future__ import annotations

from datetime import UTC, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.pipeline.stages.clean.sanitize import PRICE_COLUMNS, sanitize_bars

BASE = pd.Timestamp("2024-01-08 09:30")
MESSY_PRICE = st.one_of(
    st.floats(0.01, 1e6),
    st.floats(0.01, 1e6),
    st.floats(0.01, 1e6),
    st.sampled_from([np.nan, np.inf, -np.inf, 0.0, -1.0, -1e-9]),
    st.just("junk"),
)
MESSY_VOLUME = st.one_of(
    st.floats(0.0, 1e7),
    st.sampled_from([np.nan, np.inf, -np.inf, -5.0]),
)


@st.composite
def messy_bars(draw: st.DrawFn) -> pd.DataFrame:
    """Raw bars with duplicated / shuffled / missing timestamps and garbage cell values."""
    n = draw(st.integers(1, 40))
    minutes = draw(st.lists(st.integers(0, 30), min_size=n, max_size=n))  # small range: dups
    stamps: list[Any] = [BASE + pd.Timedelta(minutes=m) for m in minutes]
    for i in draw(st.lists(st.integers(0, n - 1), max_size=2)):
        stamps[i] = pd.NaT
    # fixed offsets: no dependency on the system tz database
    tz = draw(
        st.sampled_from([None, UTC, timezone(timedelta(hours=-5)), timezone(timedelta(hours=9))])
    )
    index = pd.DatetimeIndex(stamps)
    if tz is not None:
        index = index.tz_localize(tz)
    frame = {
        column: draw(st.lists(MESSY_PRICE, min_size=n, max_size=n)) for column in PRICE_COLUMNS
    }
    frame["volume"] = draw(st.lists(MESSY_VOLUME, min_size=n, max_size=n))
    return pd.DataFrame(frame, index=index)


@settings(max_examples=150, deadline=None)
@given(raw=messy_bars())
def test_sanitized_bars_are_sorted_unique_positive_and_ohlc_consistent(raw: pd.DataFrame) -> None:
    """Every frame that survives sanitizing satisfies the OHLCV invariants."""
    try:
        clean, report = sanitize_bars(raw)
    except ValueError:
        return  # no valid bar left: an explicit error is the contract

    assert isinstance(clean.index, pd.DatetimeIndex)
    assert clean.index.tz is None
    assert clean.index.name == "datetime"
    assert not clean.index.hasnans
    assert clean.index.is_unique
    assert clean.index.is_monotonic_increasing

    prices = clean[PRICE_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(prices).all()
    assert (prices > 0).all()
    assert (clean["high"] >= clean[["open", "close", "low"]].max(axis=1)).all()
    assert (clean["low"] <= clean[["open", "close", "high"]].min(axis=1)).all()

    volume = clean["volume"].to_numpy(dtype=float)
    assert np.isfinite(volume).all()
    assert (volume >= 0).all()

    assert report.rows_in == len(raw)
    assert report.rows_out == len(clean) <= len(raw)


@settings(max_examples=100, deadline=None)
@given(raw=messy_bars())
def test_sanitize_is_idempotent(raw: pd.DataFrame) -> None:
    """Sanitizing an already-sanitized frame changes nothing (train and serve agree)."""
    try:
        once, _ = sanitize_bars(raw)
    except ValueError:
        return

    twice, report = sanitize_bars(once)

    pd.testing.assert_frame_equal(once, twice)
    assert not report.changed


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(2, 30),
    seed=st.integers(0, 2**32 - 1),
)
def test_sanitize_shuffle_invariant_for_unique_timestamps(n: int, seed: int) -> None:
    """With unique timestamps the row order of the input does not matter."""
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1, n).cumsum() ** 2 + 1
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1, 100, n).astype(float),
        },
        index=pd.date_range(BASE, periods=n, freq="5min"),
    )
    shuffled = frame.iloc[rng.permutation(n)]

    clean_sorted, _ = sanitize_bars(frame)
    clean_shuffled, _ = sanitize_bars(shuffled)

    pd.testing.assert_frame_equal(clean_sorted, clean_shuffled)


def test_sanitize_all_invalid_raises() -> None:
    """A frame with no valid bar is an error, not an empty frame."""
    frame = pd.DataFrame(
        {"open": [-1.0], "high": [np.nan], "low": [0.0], "close": [-2.0], "volume": [1.0]},
        index=pd.DatetimeIndex([BASE]),
    )
    with pytest.raises(ValueError, match="No valid bars"):
        sanitize_bars(frame)
