"""Features must not depend on where the input history starts.

Training computes features over the full history; a deployed bundle serves a
window of recent bars. OBV as a running sum from the first bar and wavelet
coefficients z-scored with an expanding mean/std gave the same bar different
values in the two, so the served inputs left the training distribution (the docs
meta-labeling example traded 891 of 917 bars at P(win) ~0.9 for a filter that kept
about half of its bets in training). OBV now resets every session and the wavelet
z-scores use a trailing window, so values agree once the window covers them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline.stages.features.volume import add_obv, add_volume_features
from src.data.pipeline.stages.features.wavelets import (
    DEFAULT_WINDOW,
    NORMALIZE_WINDOW,
    PYWT_AVAILABLE,
    add_wavelet_features,
)
from tests.helpers import make_intraday_ohlcv


def _bars(n_rows: int = 1000) -> pd.DataFrame:
    """5-min bars with a ``datetime`` column (the feature engine's layout)."""
    return make_intraday_ohlcv(n_rows, seed=3).reset_index()


def _first_bar_of_second_session(df: pd.DataFrame) -> int:
    dates = df["datetime"].dt.date.to_numpy()
    return int(np.flatnonzero(dates != dates[0])[0])


def test_obv_is_independent_of_the_input_start() -> None:
    df = _bars()
    start = 150  # mid-session: the truncated input's first session is partial
    full = add_volume_features(df.copy(), {}).iloc[start:].reset_index(drop=True)
    window = add_volume_features(df.iloc[start:].reset_index(drop=True), {})

    # From the first complete session on (+1 bar: OBV is lagged one bar)
    first = _first_bar_of_second_session(window) + 1
    np.testing.assert_allclose(window["obv"].iloc[first:], full["obv"].iloc[first:])
    np.testing.assert_allclose(
        window["obv_sma_20"].iloc[first + 20 :], full["obv_sma_20"].iloc[first + 20 :]
    )


def test_obv_restarts_at_each_session() -> None:
    df = _bars()
    obv = add_obv(df.copy(), {})["obv"]
    signed_volume = np.sign(df["close"].diff()) * df["volume"]
    dates = df["datetime"].dt.date
    session_starts = np.flatnonzero((dates != dates.shift()).to_numpy())
    assert len(session_starts) > 2
    for s in session_starts[1:]:
        # obv[t] is the session's running sum through bar t-1
        assert obv.iloc[s + 1] == signed_volume.iloc[s]
    assert add_volume_features(df.copy(), {})["obv"].equals(obv)


@pytest.mark.skipif(not PYWT_AVAILABLE, reason="PyWavelets not installed")
def test_wavelet_coefficients_are_independent_of_the_input_start() -> None:
    df = _bars(900).set_index("datetime")
    start = 200
    kwargs = {"include_energy": False, "include_volatility": False, "include_trend": False}
    full = add_wavelet_features(df.copy(), {}, **kwargs).iloc[start:]
    window = add_wavelet_features(df.iloc[start:].copy(), {}, **kwargs)
    columns = [c for c in window.columns if c.startswith("wavelet_")]
    assert any(c.endswith("_approx") for c in columns)

    # Once the DWT window and the z-score window are both inside the input (+1 lag)
    settled = DEFAULT_WINDOW + NORMALIZE_WINDOW
    pd.testing.assert_frame_equal(window[columns].iloc[settled:], full[columns].iloc[settled:])
