"""Shared hypothesis strategies, builders and settings profiles for the property-based tests."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, Phase, settings
from hypothesis import strategies as st

BAR_FREQ = "5min"

PROFILE_ENV = "HYPOTHESIS_PROFILE"
DEFAULT_PROFILE = "dev"
# Example budget of the dev profile (hypothesis' default); per-test budgets are written
# against it and scaled to the active profile by ``budget``.
DEV_MAX_EXAMPLES = 100


def _register_profiles() -> None:
    # dev: hypothesis defaults (random exploration, shrinking on failure).
    settings.register_profile(DEFAULT_PROFILE, max_examples=DEV_MAX_EXAMPLES)
    # ci: reproducible (derandomized) and bounded (half the dev budget); no deadline
    # because the first example pays numba JIT compilation.
    settings.register_profile(
        "ci",
        derandomize=True,
        max_examples=DEV_MAX_EXAMPLES // 2,
        deadline=None,
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )


_register_profiles()
PROFILES = ("dev", "ci")


def active_profile_name() -> str:
    """``HYPOTHESIS_PROFILE`` when it names a registered profile, else ``dev``."""
    name = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    return name if name in PROFILES else DEFAULT_PROFILE


def load_hypothesis_profile() -> str:
    """Load the profile ``HYPOTHESIS_PROFILE`` selects; an unknown name warns and uses dev."""
    requested = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    name = active_profile_name()
    if name != requested:
        warnings.warn(
            f"Unknown {PROFILE_ENV}={requested!r} (expected one of {PROFILES}); "
            f"using {DEFAULT_PROFILE!r}",
            stacklevel=2,
        )
    settings.load_profile(name)
    return name


def budget(dev_examples: int, *, heavy: bool = False) -> settings:
    """Per-test settings: ``dev_examples`` under dev, scaled to the active profile's budget.

    ``heavy`` marks tests whose examples cost seconds (full feature engineering): under
    ci they skip the shrink phase, since shrinking re-runs the expensive example many
    times; the printed blob reproduces the failure locally, where dev shrinks it.
    """
    name = active_profile_name()
    active = settings.get_profile(name)
    examples = max(1, round(dev_examples * active.max_examples / DEV_MAX_EXAMPLES))
    phases = active.phases
    if heavy and name == "ci":
        phases = tuple(phase for phase in phases if phase is not Phase.shrink)
    return settings(active, max_examples=examples, phases=phases)


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
