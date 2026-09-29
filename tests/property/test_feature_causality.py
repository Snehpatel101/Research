"""Property 1: features are causal (no lookahead).

Truncation invariance: feature values at bars <= t must not change when the bars
after t are replaced by arbitrary (even wild) data. A feature that fails this uses
the future, which is data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.data.pipeline.stages.features.engineer import FeatureEngineer
from tests.property.strategies import build_ohlcv

WAVELET_WINDOW = 64  # wavelets are skipped below this many rows
RTOL = 1e-6
ATOL = 1e-9

# deadline=None: numba warm-up + ~1 s of feature computation per example
_SLOW_OK = {"deadline": None, "suppress_health_check": [HealthCheck.too_slow]}


def _perturb_future(df: pd.DataFrame, t: int, magnitude: float, seed: int) -> pd.DataFrame:
    """Replace every bar after position t with a different, wilder random walk."""
    future = build_ohlcv(
        n=len(df) - t - 1,
        seed=seed,
        vol=0.001 * magnitude,
        start_price=float(df["close"].iloc[t]) * (1.0 + 0.01 * magnitude),
    )
    out = df.copy()
    for column in ("open", "high", "low", "close", "volume"):
        scale = magnitude if column == "volume" else 1.0
        out.iloc[t + 1 :, out.columns.get_loc(column)] = future[column].to_numpy() * scale
    return out


def _first_difference(a: pd.DataFrame, b: pd.DataFrame, upto: int) -> dict[str, int]:
    """Feature -> first bar position <= upto where the two frames differ."""
    assert list(a.columns) == list(b.columns)
    bad: dict[str, int] = {}
    for column in a.columns:
        x = a[column].iloc[: upto + 1].to_numpy()
        y = b[column].iloc[: upto + 1].to_numpy()
        if x.dtype.kind not in "fiub" or y.dtype.kind not in "fiub":
            same = x == y
        else:
            same = np.isclose(
                x.astype(float), y.astype(float), rtol=RTOL, atol=ATOL, equal_nan=True
            )
        if not np.all(same):
            bad[column] = int(np.flatnonzero(~same)[0])
    return bad


def _causality_violations(
    engineer: FeatureEngineer, df: pd.DataFrame, t: int, magnitude: float, seed: int
) -> dict[str, int]:
    frame = df.reset_index()
    perturbed = _perturb_future(df, t, magnitude, seed).reset_index()
    original, _ = engineer.compute_features(frame)
    changed, _ = engineer.compute_features(perturbed)
    return _first_difference(original, changed, t)


@settings(max_examples=12, **_SLOW_OK)
@given(
    seed=st.integers(0, 2**32 - 1),
    vol=st.sampled_from([1e-4, 5e-4, 2e-3, 1e-2]),
    flat_fraction=st.sampled_from([0.0, 0.1, 0.4]),
    t_frac=st.floats(0.2, 0.9),
    magnitude=st.floats(0.5, 50.0),
    future_seed=st.integers(0, 2**32 - 1),
)
def test_features_at_or_before_t_ignore_future_bars(
    seed: int, vol: float, flat_fraction: float, t_frac: float, magnitude: float, future_seed: int
) -> None:
    """Perturbing bars after t leaves every feature value at bars <= t unchanged."""
    df = build_ohlcv(400, seed=seed, vol=vol, flat_fraction=flat_fraction)
    t = max(WAVELET_WINDOW, int(t_frac * len(df)))
    engineer = FeatureEngineer(enable_mtf=False)

    violations = _causality_violations(engineer, df, t, magnitude, future_seed)

    assert not violations, (
        f"lookahead: {len(violations)} feature(s) at bars <= {t} depend on later bars; "
        f"feature -> first differing bar: {dict(list(violations.items())[:10])}"
    )


@settings(max_examples=5, **_SLOW_OK)
@given(
    seed=st.integers(0, 2**32 - 1),
    t_frac=st.floats(0.4, 0.9),
    magnitude=st.floats(0.5, 20.0),
    future_seed=st.integers(0, 2**32 - 1),
)
def test_mtf_features_at_or_before_t_ignore_future_bars(
    seed: int, t_frac: float, magnitude: float, future_seed: int
) -> None:
    """Multi-timeframe features (15min / 60min resampled) are causal too."""
    df = build_ohlcv(1200, seed=seed, vol=5e-4)
    t = int(t_frac * len(df))
    engineer = FeatureEngineer(enable_mtf=True, mtf_min_rows=200)

    violations = _causality_violations(engineer, df, t, magnitude, future_seed)

    assert not violations, (
        f"MTF lookahead: {len(violations)} feature(s) at bars <= {t} depend on later "
        f"bars; feature -> first differing bar: {dict(list(violations.items())[:10])}"
    )
