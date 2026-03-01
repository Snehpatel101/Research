"""
Liquidity estimation features from OHLCV data.

PHASE_19A ML Pipeline Enhancement: LIQUIDITY features.

These features estimate market liquidity, bid-ask spreads, and
liquidity regime changes from OHLCV data without requiring
order book information.
"""

import functools
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.features.compute._helpers import rolling_std, sma

# Numba JIT acceleration for rolling.apply() functions
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# liquidity uses min_periods=1 (different from the default min_periods=window)
_sma = functools.partial(sma, min_periods=1)
_rolling_std = functools.partial(rolling_std, min_periods=1)


# =============================================================================
# SPREAD ESTIMATION FEATURES
# =============================================================================


def compute_spread_estimate(df: pd.DataFrame) -> pd.Series:
    """
    Estimate bid-ask spread from OHLCV using high-low range.

    This method provides a simple spread estimator adapted
    for intraday bars using the high-low range as a proxy.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns

    Returns:
        Series with estimated spread as percentage of price
    """
    # Simple high-low spread estimator
    hl_ratio = (df["high"] - df["low"]) / df["close"]

    # Smooth to reduce noise
    spread = hl_ratio.rolling(window=5, min_periods=1).mean()

    return spread


def compute_spread_estimate_10(df: pd.DataFrame) -> pd.Series:
    """
    10-period smoothed spread estimate.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with smoothed spread estimate
    """
    hl_ratio = (df["high"] - df["low"]) / df["close"]
    return hl_ratio.rolling(window=10, min_periods=1).mean()


def compute_spread_estimate_20(df: pd.DataFrame) -> pd.Series:
    """
    20-period smoothed spread estimate.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with smoothed spread estimate
    """
    hl_ratio = (df["high"] - df["low"]) / df["close"]
    return hl_ratio.rolling(window=20, min_periods=1).mean()


# =============================================================================
# LIQUIDITY REGIME FEATURES
# =============================================================================


def _percentile_rank(x: np.ndarray) -> float:
    """Calculate percentile rank of the last value (raw ndarray version)."""
    if len(x) < 2:
        return 0.5
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if denom == 0:
        return np.nan
    return (x[-1] - x_min) / denom


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _percentile_rank_numba(x: np.ndarray) -> float:
        """Numba-accelerated percentile rank of last value in window."""
        n = len(x)
        if n < 2:
            return 0.5
        last = x[-1]
        count_below = 0
        for i in range(n - 1):
            if x[i] < last:
                count_below += 1
        return count_below / (n - 1)

    _percentile_rank = _percentile_rank_numba

    @njit(cache=True)
    def _rolling_percentile_rank_numba(
        values: np.ndarray, window: int, min_periods: int
    ) -> np.ndarray:
        """Fully vectorized rolling percentile rank (replaces pandas rolling.apply)."""
        n = len(values)
        result = np.full(n, np.nan)
        for i in range(min_periods - 1, n):
            start = max(0, i - window + 1)
            # Count valid (non-NaN) values in window
            count = 0
            last = values[i]
            if np.isnan(last):
                continue
            count_below = 0
            for j in range(start, i + 1):
                v = values[j]
                if not np.isnan(v):
                    count += 1
                    if j < i and v < last:
                        count_below += 1
            if count >= min_periods and count >= 2:
                result[i] = count_below / (count - 1)
        return result


def _rolling_percentile_rank_pandas(
    values: np.ndarray, window: int, min_periods: int
) -> np.ndarray:
    """Fallback rolling percentile rank when numba is not available."""
    s = pd.Series(values)
    return (
        s.rolling(window=window, min_periods=min_periods).apply(_percentile_rank, raw=True).values
    )


if not NUMBA_AVAILABLE:
    _rolling_percentile_rank_numba = _rolling_percentile_rank_pandas  # type: ignore[assignment]


def compute_liquidity_regime_10(df: pd.DataFrame) -> pd.Series:
    """
    10-period liquidity regime classification.

    Classifies liquidity into regimes: -1 (low), 0 (normal), 1 (high).
    Uses spread percentile within rolling window.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with regime labels
    """
    spread = compute_spread_estimate(df)

    # Rolling percentile rank (vectorized numba loop)
    pct_rank = pd.Series(_rolling_percentile_rank_numba(spread.values, 10, 5), index=spread.index)

    # Classify: low spread = high liquidity (1), high spread = low liquidity (-1)
    regime = pd.Series(0, index=df.index)
    regime[pct_rank < 0.33] = 1  # High liquidity (tight spreads)
    regime[pct_rank > 0.67] = -1  # Low liquidity (wide spreads)

    return regime


def compute_liquidity_regime_20(df: pd.DataFrame) -> pd.Series:
    """
    20-period liquidity regime classification.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with regime labels
    """
    spread = compute_spread_estimate(df)

    pct_rank = pd.Series(_rolling_percentile_rank_numba(spread.values, 20, 5), index=spread.index)

    regime = pd.Series(0, index=df.index)
    regime[pct_rank < 0.33] = 1
    regime[pct_rank > 0.67] = -1

    return regime


def compute_liquidity_regime_60(df: pd.DataFrame) -> pd.Series:
    """
    60-period liquidity regime classification.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with regime labels
    """
    spread = compute_spread_estimate(df)

    pct_rank = pd.Series(_rolling_percentile_rank_numba(spread.values, 60, 10), index=spread.index)

    regime = pd.Series(0, index=df.index)
    regime[pct_rank < 0.33] = 1
    regime[pct_rank > 0.67] = -1

    return regime


# =============================================================================
# SLIPPAGE ESTIMATION FEATURES
# =============================================================================


def compute_slippage_estimate(df: pd.DataFrame) -> pd.Series:
    """
    Estimate expected slippage based on spread and volume.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with estimated slippage as fraction of price
    """
    spread = compute_spread_estimate(df)
    volume_impact = 0.1

    # Lower volume = higher slippage potential
    vol_ma = df["volume"].rolling(window=20, min_periods=1).mean()
    vol_ratio = vol_ma / (df["volume"] + 1)  # Current vs average

    # Slippage = half spread + volume impact
    slippage = (spread / 2) + (volume_impact * (vol_ratio - 1).clip(lower=0))

    return slippage.clip(lower=0)


# =============================================================================
# VOLUME PROFILE FEATURES
# =============================================================================


def compute_volume_ratio_20(df: pd.DataFrame) -> pd.Series:
    """
    Volume relative to 20-period average.

    Values > 1 indicate above-average volume (higher liquidity).

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with volume ratio
    """
    vol = df["volume"]
    vol_ma = vol.rolling(window=20, min_periods=1).mean()
    return vol / (vol_ma + 1)


def compute_volume_trend(df: pd.DataFrame) -> pd.Series:
    """
    Volume trend: percentage change in volume moving average.

    Positive values indicate increasing liquidity.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with volume trend
    """
    vol = df["volume"]
    vol_ma = vol.rolling(window=20, min_periods=1).mean()
    return vol_ma.pct_change(periods=10).fillna(0)


def compute_volume_cv(df: pd.DataFrame) -> pd.Series:
    """
    Volume coefficient of variation (volatility of volume).

    High CV indicates unstable liquidity.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with volume CV
    """
    vol = df["volume"]
    vol_ma = vol.rolling(window=20, min_periods=1).mean()
    vol_std = vol.rolling(window=20, min_periods=1).std()
    return vol_std / (vol_ma + 1)


# =============================================================================
# LIQUIDITY SHOCK FEATURES
# =============================================================================


def compute_liquidity_shock(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity shock indicator: sudden spread widening.

    Values > 2 indicate significant liquidity deterioration.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with liquidity shock indicator
    """
    spread = compute_spread_estimate(df)
    spread_ma = spread.rolling(window=20, min_periods=5).mean()
    spread_std = spread.rolling(window=20, min_periods=5).std()

    # Z-score of current spread
    spread_std = spread_std.replace(0, np.nan)
    shock = (spread - spread_ma) / spread_std

    return shock.fillna(0)


def compute_illiquidity_index(df: pd.DataFrame) -> pd.Series:
    """
    Composite illiquidity index combining spread and volume.

    Higher values indicate worse liquidity conditions.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Series with illiquidity index
    """
    spread = compute_spread_estimate(df)
    vol_ratio = compute_volume_ratio_20(df)

    # High spread + low volume = high illiquidity
    illiquidity = spread / (vol_ratio + 0.01)

    return illiquidity


# =============================================================================
# AGGREGATE FEATURE COMPUTATION
# =============================================================================


def compute_liquidity_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Compute all liquidity features for a DataFrame.

    Args:
        df: DataFrame with OHLCV columns
        windows: List of windows for rolling features

    Returns:
        DataFrame with all liquidity features
    """
    if windows is None:
        windows = [10, 20, 60]

    features = pd.DataFrame(index=df.index)

    # Spread estimate
    features["spread_estimate"] = compute_spread_estimate(df)

    # Liquidity regime at different windows
    features["liquidity_regime_10"] = compute_liquidity_regime_10(df)
    features["liquidity_regime_20"] = compute_liquidity_regime_20(df)
    features["liquidity_regime_60"] = compute_liquidity_regime_60(df)

    # Slippage estimate
    features["slippage_estimate"] = compute_slippage_estimate(df)

    # Volume profile
    features["volume_ratio_liq"] = compute_volume_ratio_20(df)
    features["volume_trend"] = compute_volume_trend(df)
    features["volume_cv"] = compute_volume_cv(df)

    return features


# =============================================================================
# FEATURE MAP
# =============================================================================

LIQUIDITY_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Spread Estimation
    "spread_estimate": compute_spread_estimate,
    "spread_estimate_10": compute_spread_estimate_10,
    "spread_estimate_20": compute_spread_estimate_20,
    # Liquidity Regime
    "liquidity_regime_10": compute_liquidity_regime_10,
    "liquidity_regime_20": compute_liquidity_regime_20,
    "liquidity_regime_60": compute_liquidity_regime_60,
    # Slippage
    "slippage_estimate": compute_slippage_estimate,
    # Volume Profile
    "volume_ratio_liq": compute_volume_ratio_20,
    "volume_trend": compute_volume_trend,
    "volume_cv": compute_volume_cv,
    # Liquidity Shocks
    "liquidity_shock": compute_liquidity_shock,
    "illiquidity_index": compute_illiquidity_index,
}

# Feature family metadata
FEATURE_FAMILY = "liquidity"
FEATURE_COUNT = 12

__all__ = [
    "LIQUIDITY_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # Spread Estimation
    "compute_spread_estimate",
    "compute_spread_estimate_10",
    "compute_spread_estimate_20",
    # Liquidity Regime
    "compute_liquidity_regime_10",
    "compute_liquidity_regime_20",
    "compute_liquidity_regime_60",
    # Slippage
    "compute_slippage_estimate",
    # Volume Profile
    "compute_volume_ratio_20",
    "compute_volume_trend",
    "compute_volume_cv",
    # Liquidity Shocks
    "compute_liquidity_shock",
    "compute_illiquidity_index",
    # Aggregate
    "compute_liquidity_features",
]
