"""Shared hypothesis strategies and builders for the property-based tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st

BAR_FREQ = "5min"


def build_ohlcv(
    n: int,
    seed: int,
    vol: float = 0.001,
    flat_fraction: float = 0.0,
    start: str = "2024-01-02 09:30",
    start_price: float = 5000.0,
    tz: str | None = None,
) -> pd.DataFrame:
    """Synthetic 5-minute OHLCV with a ``datetime`` index (valid bars: low <= o,c <= high).

    ``flat_fraction`` of the bars repeat the previous close with zero range and
    tiny volume (illiquid stretches stress division-by-range features).
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]]) * np.exp(rng.normal(0.0, vol / 4, n))
    span = np.abs(rng.normal(0.0, vol, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.integers(50, 5000, n).astype(float)
    if flat_fraction > 0:
        flat = rng.random(n) < flat_fraction
        flat[0] = False
        for i in np.flatnonzero(flat):
            close[i] = close[i - 1]
            open_[i] = high[i] = low[i] = close[i]
            volume[i] = 1.0
    idx = pd.date_range(start, periods=n, freq=BAR_FREQ, tz=tz, name="datetime")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


@st.composite
def ohlcv_frames(draw: st.DrawFn, min_n: int = 150, max_n: int = 400) -> pd.DataFrame:
    """Valid OHLCV bars with drawn length, volatility, seed and illiquid stretches."""
    return build_ohlcv(
        n=draw(st.integers(min_n, max_n)),
        seed=draw(st.integers(0, 2**32 - 1)),
        vol=draw(st.sampled_from([1e-4, 5e-4, 2e-3, 1e-2])),
        flat_fraction=draw(st.sampled_from([0.0, 0.0, 0.1, 0.4])),
    )
