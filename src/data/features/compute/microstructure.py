"""
Microstructure feature computation - Amihud, Roll, Kyle, spreads, imbalance.

PHASE_1 Unified Features: 15 MICROSTRUCTURE features.

These features measure market quality, liquidity, and microstructure effects.

Phase 24: Added caching to avoid redundant computation for Amihud, Roll spread,
relative spread, and volume imbalance base calculations.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

# =============================================================================
# MODULE-LEVEL CACHES (Phase 24: Avoid redundant computation)
# =============================================================================

# Caches keyed by id(df)
_amihud_cache: dict[int, pd.Series] = {}
_roll_spread_cache: dict[int, pd.Series] = {}
_rel_spread_cache: dict[int, pd.Series] = {}
_volume_imbalance_cache: dict[int, pd.Series] = {}


def clear_microstructure_cache() -> None:
    """Clear all microstructure computation caches. Call between different DataFrames."""
    _amihud_cache.clear()
    _roll_spread_cache.clear()
    _rel_spread_cache.clear()
    _volume_imbalance_cache.clear()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def _rolling_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling covariance."""
    return x.rolling(window=window, min_periods=window).cov(y)


def _rolling_var(series: pd.Series, window: int) -> pd.Series:
    """Rolling variance."""
    return series.rolling(window=window, min_periods=window).var()


def _log_returns(close: pd.Series) -> pd.Series:
    """Calculate log returns."""
    return np.log(close / close.shift(1))


# =============================================================================
# AMIHUD ILLIQUIDITY FEATURES
# =============================================================================


def _compute_micro_amihud_base(df: pd.DataFrame) -> pd.Series:
    """
    Amihud illiquidity measure (internal computation).

    Amihud = |Return| / Dollar Volume
    Higher values indicate lower liquidity (price moves more per dollar traded).
    """
    returns = df["close"].pct_change().abs()
    dollar_volume = df["close"] * df["volume"]

    # Avoid division by zero
    dollar_volume = dollar_volume.replace(0, np.nan)

    return returns / dollar_volume


def _get_amihud_cached(df: pd.DataFrame) -> pd.Series:
    """Get cached Amihud or compute if not cached. Phase 24."""
    cache_key = id(df)
    if cache_key not in _amihud_cache:
        _amihud_cache[cache_key] = _compute_micro_amihud_base(df)
    return _amihud_cache[cache_key]


def compute_micro_amihud(df: pd.DataFrame) -> pd.Series:
    """
    Amihud illiquidity measure.

    Amihud = |Return| / Dollar Volume
    Higher values indicate lower liquidity (price moves more per dollar traded).
    """
    return _get_amihud_cached(df)


def compute_micro_amihud_10(df: pd.DataFrame) -> pd.Series:
    """10-period average Amihud illiquidity."""
    amihud = _get_amihud_cached(df)
    return _sma(amihud, window=10)


def compute_micro_amihud_20(df: pd.DataFrame) -> pd.Series:
    """20-period average Amihud illiquidity."""
    amihud = _get_amihud_cached(df)
    return _sma(amihud, window=20)


# =============================================================================
# ROLL SPREAD FEATURES
# =============================================================================


def _compute_micro_roll_spread_base(df: pd.DataFrame) -> pd.Series:
    """
    Roll spread estimator (internal computation).

    Roll Spread = 2 * sqrt(-Cov(Delta_P_t, Delta_P_t-1))
    Estimates effective bid-ask spread from price autocorrelation.
    """
    price_diff = df["close"].diff()
    cov = _rolling_cov(price_diff, price_diff.shift(1), window=20)

    # Roll spread only valid for negative covariance
    # For positive covariance, spread is undefined (return 0)
    roll_spread = 2 * np.sqrt((-cov).clip(lower=0))

    return roll_spread


def _get_roll_spread_cached(df: pd.DataFrame) -> pd.Series:
    """Get cached Roll spread or compute if not cached. Phase 24."""
    cache_key = id(df)
    if cache_key not in _roll_spread_cache:
        _roll_spread_cache[cache_key] = _compute_micro_roll_spread_base(df)
    return _roll_spread_cache[cache_key]


def compute_micro_roll_spread(df: pd.DataFrame) -> pd.Series:
    """
    Roll spread estimator.

    Roll Spread = 2 * sqrt(-Cov(Delta_P_t, Delta_P_t-1))
    Estimates effective bid-ask spread from price autocorrelation.
    """
    return _get_roll_spread_cached(df)


def compute_micro_roll_spread_pct(df: pd.DataFrame) -> pd.Series:
    """Roll spread as percentage of price."""
    roll_spread = _get_roll_spread_cached(df)
    # Avoid division by zero
    close = df["close"].replace(0, np.nan)
    return (roll_spread / close) * 100


# =============================================================================
# KYLE'S LAMBDA FEATURES
# =============================================================================


def compute_micro_kyle_lambda(df: pd.DataFrame) -> pd.Series:
    """
    Kyle's Lambda - price impact coefficient.

    Lambda = Cov(Delta_P, signed_volume) / Var(signed_volume)
    Measures how much price moves per unit of signed order flow.
    """
    price_diff = df["close"].diff()

    # Signed volume: positive for up bars, negative for down bars
    direction = np.sign(price_diff)
    signed_volume = direction * df["volume"]

    # Compute lambda using rolling regression
    cov = _rolling_cov(price_diff, signed_volume, window=20)
    var = _rolling_var(signed_volume, window=20)

    # Avoid division by zero
    var = var.replace(0, np.nan)

    return cov / var


# =============================================================================
# CORWIN-SCHULTZ SPREAD
# =============================================================================


def compute_micro_cs_spread(df: pd.DataFrame) -> pd.Series:
    """
    Corwin-Schultz high-low spread estimator.

    Uses the relationship between daily high-low range and volatility
    to estimate the bid-ask spread.
    """
    # Calculate beta (sum of squared log high-low ratios)
    log_hl = np.log(df["high"] / df["low"])
    log_hl_sq = log_hl**2

    # Two-period sum
    beta = (log_hl_sq + log_hl_sq.shift(1)).rolling(window=20, min_periods=20).mean()

    # Calculate gamma (squared log of two-period high-low range)
    high_2 = df["high"].rolling(window=2).max()
    low_2 = df["low"].rolling(window=2).min()
    gamma = (np.log(high_2 / low_2)) ** 2

    gamma = gamma.rolling(window=20, min_periods=20).mean()

    # Alpha parameter
    sqrt_2 = np.sqrt(2)
    denom = 3 - 2 * sqrt_2
    alpha_sq = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha_sq = alpha_sq.clip(lower=0)

    # Spread estimate
    spread = 2 * (np.exp(np.sqrt(alpha_sq)) - 1) / (1 + np.exp(np.sqrt(alpha_sq)))

    return spread


# =============================================================================
# RELATIVE SPREAD FEATURES
# =============================================================================


def _compute_micro_rel_spread_base(df: pd.DataFrame) -> pd.Series:
    """
    Relative spread proxy using high-low range (internal computation).

    Approximates bid-ask spread as a fraction of the range.
    """
    hl_range = df["high"] - df["low"]
    mid_price = (df["high"] + df["low"]) / 2

    # Avoid division by zero
    mid_price = mid_price.replace(0, np.nan)

    return hl_range / mid_price


def _get_rel_spread_cached(df: pd.DataFrame) -> pd.Series:
    """Get cached relative spread or compute if not cached. Phase 24."""
    cache_key = id(df)
    if cache_key not in _rel_spread_cache:
        _rel_spread_cache[cache_key] = _compute_micro_rel_spread_base(df)
    return _rel_spread_cache[cache_key]


def compute_micro_rel_spread(df: pd.DataFrame) -> pd.Series:
    """
    Relative spread proxy using high-low range.

    Approximates bid-ask spread as a fraction of the range.
    """
    return _get_rel_spread_cached(df)


def compute_micro_rel_spread_10(df: pd.DataFrame) -> pd.Series:
    """10-period average relative spread."""
    rel_spread = _get_rel_spread_cached(df)
    return _sma(rel_spread, window=10)


# =============================================================================
# VOLUME IMBALANCE FEATURES
# =============================================================================


def _compute_micro_volume_imbalance_base(df: pd.DataFrame) -> pd.Series:
    """
    Volume imbalance indicator (internal computation).

    Positive when buying pressure (price up with volume),
    negative when selling pressure (price down with volume).
    """
    price_direction = np.sign(df["close"].diff())
    return price_direction * df["volume"]


def _get_volume_imbalance_cached(df: pd.DataFrame) -> pd.Series:
    """Get cached volume imbalance or compute if not cached. Phase 24."""
    cache_key = id(df)
    if cache_key not in _volume_imbalance_cache:
        _volume_imbalance_cache[cache_key] = _compute_micro_volume_imbalance_base(df)
    return _volume_imbalance_cache[cache_key]


def compute_micro_volume_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Volume imbalance indicator.

    Positive when buying pressure (price up with volume),
    negative when selling pressure (price down with volume).
    """
    return _get_volume_imbalance_cached(df)


def compute_micro_cum_imbalance_20(df: pd.DataFrame) -> pd.Series:
    """
    20-period cumulative volume imbalance.

    Measures net buying/selling pressure over the window.
    """
    imbalance = _get_volume_imbalance_cached(df)
    return imbalance.rolling(window=20, min_periods=20).sum()


# =============================================================================
# TRADE INTENSITY FEATURES
# =============================================================================


def compute_micro_trade_intensity_20(df: pd.DataFrame) -> pd.Series:
    """
    20-period trade intensity (volume acceleration).

    Measures how much current volume deviates from recent average.
    """
    vol_sma = _sma(df["volume"], window=20)

    # Avoid division by zero
    vol_sma = vol_sma.replace(0, np.nan)

    # Intensity as ratio above/below average
    return df["volume"] / vol_sma


def compute_micro_trade_intensity_50(df: pd.DataFrame) -> pd.Series:
    """
    50-period trade intensity.

    Longer-term volume comparison.
    """
    vol_sma = _sma(df["volume"], window=50)
    vol_sma = vol_sma.replace(0, np.nan)
    return df["volume"] / vol_sma


# =============================================================================
# PRICE EFFICIENCY FEATURES
# =============================================================================


def compute_micro_efficiency_10(df: pd.DataFrame) -> pd.Series:
    """
    10-period price efficiency ratio.

    Efficiency = |Net Price Change| / Sum(|Individual Changes|)
    Ranges from 0 (mean-reverting) to 1 (trending).
    """
    price_diff = df["close"].diff().abs()
    net_change = df["close"].diff(10).abs()

    sum_changes = price_diff.rolling(window=10, min_periods=10).sum()

    # Avoid division by zero
    sum_changes = sum_changes.replace(0, np.nan)

    return net_change / sum_changes


def compute_micro_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Realized volatility ratio (short-term / long-term).

    High ratio indicates volatility spike, useful for regime detection.
    """
    log_ret = _log_returns(df["close"])

    # Short-term volatility (5-period)
    short_vol = log_ret.rolling(window=5, min_periods=5).std()

    # Long-term volatility (20-period)
    long_vol = log_ret.rolling(window=20, min_periods=20).std()

    # Avoid division by zero
    long_vol = long_vol.replace(0, np.nan)

    return short_vol / long_vol


# =============================================================================
# FEATURE MAP
# =============================================================================

MICROSTRUCTURE_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Amihud
    "micro_amihud": compute_micro_amihud,
    "micro_amihud_10": compute_micro_amihud_10,
    "micro_amihud_20": compute_micro_amihud_20,
    # Roll Spread
    "micro_roll_spread": compute_micro_roll_spread,
    "micro_roll_spread_pct": compute_micro_roll_spread_pct,
    # Kyle's Lambda
    "micro_kyle_lambda": compute_micro_kyle_lambda,
    # Corwin-Schultz
    "micro_cs_spread": compute_micro_cs_spread,
    # Relative Spread
    "micro_rel_spread": compute_micro_rel_spread,
    "micro_rel_spread_10": compute_micro_rel_spread_10,
    # Volume Imbalance
    "micro_volume_imbalance": compute_micro_volume_imbalance,
    "micro_cum_imbalance_20": compute_micro_cum_imbalance_20,
    # Trade Intensity
    "micro_trade_intensity_20": compute_micro_trade_intensity_20,
    "micro_trade_intensity_50": compute_micro_trade_intensity_50,
    # Efficiency
    "micro_efficiency_10": compute_micro_efficiency_10,
    "micro_vol_ratio": compute_micro_vol_ratio,
}

# Feature family metadata
FEATURE_FAMILY = "microstructure"
FEATURE_COUNT = 15

__all__ = [
    "MICROSTRUCTURE_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    "clear_microstructure_cache",
    # Amihud
    "compute_micro_amihud",
    "compute_micro_amihud_10",
    "compute_micro_amihud_20",
    # Roll Spread
    "compute_micro_roll_spread",
    "compute_micro_roll_spread_pct",
    # Kyle's Lambda
    "compute_micro_kyle_lambda",
    # Corwin-Schultz
    "compute_micro_cs_spread",
    # Relative Spread
    "compute_micro_rel_spread",
    "compute_micro_rel_spread_10",
    # Volume Imbalance
    "compute_micro_volume_imbalance",
    "compute_micro_cum_imbalance_20",
    # Trade Intensity
    "compute_micro_trade_intensity_20",
    "compute_micro_trade_intensity_50",
    # Efficiency
    "compute_micro_efficiency_10",
    "compute_micro_vol_ratio",
]
