"""Features must not depend on where the input history starts.

Training computes features over the full history; a deployed bundle serves a
window of recent bars. OBV as a running sum from the first bar and wavelet
coefficients z-scored with an expanding mean/std gave the same bar different
values in the two, so the served inputs left the training distribution (the docs
meta-labeling example traded 891 of 917 bars at P(win) ~0.9 for a filter that kept
about half of its bets in training). OBV now resets every session and the wavelet
z-scores use a trailing window, so values agree once the window covers them.

Training and serving then apply one warmup rule (``warmup_mask``): a row is kept
only once ``FeatureEngineer.warmup_bars()`` bars precede it and its session starts
inside the input. Every kept row of a truncated input must equal the full-history
row, for any start.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.constants import OHLCV_COLUMNS
from src.data.pipeline.stages.features.engineer import FeatureEngineer, warmup_mask
from src.data.pipeline.stages.features.regime import (
    VOLATILITY_REGIME_WINDOW,
    add_trend_regime,
    add_volatility_regime,
)
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


# ---------------------------------------------------------------------------
# The warmup rule
# ---------------------------------------------------------------------------

# An EWM's start still weighs EWM_SETTLE_TOLERANCE (1e-3) after the warmup, so
# kept rows agree to well under that fraction of each feature's spread
FEATURE_TOLERANCE = 1e-3


def _assert_kept_rows_match(engineer: FeatureEngineer, raw: pd.DataFrame, start: int) -> int:
    """Features of ``raw[start:]`` on its kept rows equal those of ``raw``; returns kept rows."""
    full, _ = engineer.compute_features(raw)
    window, _ = engineer.compute_features(raw.iloc[start:].reset_index(drop=True))
    columns = [c for c in full.columns if c not in ("datetime", *OHLCV_COLUMNS)]

    keep = engineer.warmup_mask(window["datetime"])
    served = window.loc[keep, columns].to_numpy(float)
    reference = full[columns].iloc[start:].to_numpy(float)[keep]
    assert not np.isnan(served).any(), "a kept row still has warmup NaNs"

    spread = np.nanstd(full[columns].to_numpy(float), axis=0)
    tolerance = FEATURE_TOLERANCE * spread + 1e-9 * np.abs(reference)
    off = np.abs(served - reference) > tolerance
    bad = sorted({columns[j] for j in np.nonzero(off)[1]})
    assert not off.any(), f"start {start}: differ from full history: {bad}"
    return int(keep.sum())


def _rth_bars(n_days: int, freq: str, seed: int) -> pd.DataFrame:
    """Gapped sessions: 09:30-16:00 on weekdays only (overnight and weekend gaps)."""
    days = pd.bdate_range("2024-01-02", periods=n_days)
    index = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(day + pd.Timedelta("9h30min"), day + pd.Timedelta("16h"), freq=freq)[
                    :-1
                ].to_numpy()
                for day in days
            ]
        ),
        name="datetime",
    )
    bars = make_intraday_ohlcv(len(index), seed=seed, freq=freq)
    bars.index = index
    return bars.reset_index()


@pytest.mark.parametrize(
    ("timeframe", "mtf"),
    [("5min", False), ("5min", True), ("1min", False)],
    ids=["5min", "5min_mtf", "1min"],
)
def test_every_kept_row_matches_full_history(timeframe: str, mtf: bool) -> None:
    engineer = FeatureEngineer(timeframe=timeframe, enable_mtf=mtf)
    warmup = engineer.warmup_bars()
    start = 437  # mid-session
    raw = make_intraday_ohlcv(start + warmup + 400, seed=5, freq=timeframe).reset_index()
    assert _assert_kept_rows_match(engineer, raw, start) > 300

    # The full-history warmup also covers every feature's NaN warmup
    full, _ = engineer.compute_features(raw)
    columns = [c for c in full.columns if c not in ("datetime", *OHLCV_COLUMNS)]
    first_complete = int(full[columns].notna().all(axis=1).to_numpy().argmax())
    assert first_complete <= warmup


@pytest.mark.parametrize("timeframe", ["5min", "1min"])
def test_windows_starting_around_midnight_match_full_history(timeframe: str) -> None:
    """24/7 bars, windows starting at random times incl. just after midnight.

    With wavelets off the warmup is shorter than a day, so the session rule binds:
    the OBV moving average must not average OBV from the partial first date.
    """
    engineer = FeatureEngineer(timeframe=timeframe, enable_mtf=False, enable_wavelets=False)
    per_day = pd.Timedelta("1D") // pd.Timedelta(timeframe)
    raw = make_intraday_ohlcv(3 * per_day + engineer.warmup_bars(), seed=11, freq=timeframe)
    raw = raw.reset_index()
    first_midnight = int(np.flatnonzero(raw["datetime"].dt.hour.diff().lt(0).to_numpy())[0])
    rng = np.random.default_rng(0)
    starts = [first_midnight + 1, first_midnight + 7, *rng.integers(1, per_day, size=3)]
    for start in starts:
        assert _assert_kept_rows_match(engineer, raw, int(start)) > 100


@pytest.mark.parametrize("timeframe", ["5min", "1min"])
def test_gapped_sessions_match_full_history_for_any_start(timeframe: str) -> None:
    """RTH-style sessions with overnight and weekend gaps, several start offsets."""
    engineer = FeatureEngineer(timeframe=timeframe, enable_mtf=False, enable_wavelets=False)
    per_day = pd.Timedelta("6h30min") // pd.Timedelta(timeframe)
    n_days = engineer.warmup_bars() // per_day + 6
    raw = _rth_bars(n_days, timeframe, seed=13)
    rng = np.random.default_rng(1)
    for start in [1, per_day - 3, per_day + 1, *rng.integers(1, 2 * per_day, size=2)]:
        assert _assert_kept_rows_match(engineer, raw, int(start)) > 100


def test_warmup_mask_needs_warmup_bars_and_a_covered_session() -> None:
    times = pd.Series(pd.date_range("2024-01-02 20:00", periods=12, freq="1h"))
    # Bars 0-3 are on the first (partial) date; a row is kept once the bar
    # session_lookback rows back is past it
    np.testing.assert_array_equal(warmup_mask(times, 2, session_lookback=1), np.arange(12) >= 5)
    np.testing.assert_array_equal(warmup_mask(times, 2, session_lookback=3), np.arange(12) >= 7)
    np.testing.assert_array_equal(warmup_mask(times, 9, session_lookback=3), np.arange(12) >= 9)
    np.testing.assert_array_equal(warmup_mask(times, 2, session_lookback=0), np.arange(12) >= 2)


def test_session_lookback_covers_the_obv_moving_average() -> None:
    engineer = FeatureEngineer(timeframe="5min", enable_mtf=False)
    assert engineer.session_lookback_bars() == max(engineer.period_config["volume_sma"]) + 1
    assert FeatureEngineer(enable_volume_features=False).session_lookback_bars() == 0


def test_warmup_covers_mtf_timeframes_and_mtf_min_rows() -> None:
    base = FeatureEngineer(timeframe="5min", enable_mtf=False)
    hourly = FeatureEngineer(timeframe="5min", enable_mtf=True, mtf_timeframes=["60min"])
    quarter = FeatureEngineer(timeframe="5min", enable_mtf=True, mtf_timeframes=["15min"])
    assert hourly.warmup_bars() > quarter.warmup_bars() > base.warmup_bars()
    # Below mtf_min_rows MTF features are not computed at all
    assert quarter.warmup_bars() >= quarter.mtf_min_rows


def test_warmup_tolerance_is_a_spec_field() -> None:
    loose = FeatureEngineer(timeframe="5min", ewm_settle_tolerance=1e-2)
    strict = FeatureEngineer(timeframe="5min", ewm_settle_tolerance=1e-4)
    assert (
        loose.warmup_bars() < FeatureEngineer(timeframe="5min").warmup_bars() < strict.warmup_bars()
    )
    assert FeatureEngineer.from_spec(loose.to_spec()).warmup_bars() == loose.warmup_bars()


def test_regime_features_are_nan_until_warm() -> None:
    df = add_volume_features(_bars(400), {})
    df["hvol_20"] = df["close"].pct_change().rolling(20).std()
    df["sma_50"] = df["close"].rolling(50).mean().shift(1)
    df["sma_200"] = df["close"].rolling(200).mean().shift(1)
    vol = add_volatility_regime(df.copy(), {})["volatility_regime"]
    trend = add_trend_regime(df.copy(), {})["trend_regime"]
    first_hvol = int(df["hvol_20"].notna().to_numpy().argmax())
    assert vol.iloc[: first_hvol + VOLATILITY_REGIME_WINDOW - 1].isna().all()
    assert vol.iloc[first_hvol + VOLATILITY_REGIME_WINDOW - 1 :].notna().all()
    assert trend.iloc[:200].isna().all() and trend.iloc[200:].notna().all()
