"""
Mean-reversion detection metrics for time series.

PHASE_19A ML Pipeline Enhancement: MEAN_REVERSION features.

These features detect mean-reverting behavior in price series,
useful for identifying trading opportunities and regime changes.
"""

import functools
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.features.compute._helpers import rolling_std, sma
from src.data.features.compute.entropy import compute_hurst_100

try:
    from numba import njit  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# mean_reversion uses min_periods=1 (different from the default min_periods=window)
_sma = functools.partial(sma, min_periods=1)
_rolling_std = functools.partial(rolling_std, min_periods=1)


# =============================================================================
# Z-SCORE FEATURES
# =============================================================================


def compute_mr_zscore_10(df: pd.DataFrame) -> pd.Series:
    """
    10-period Z-score of price relative to rolling mean.

    Extreme values (|z| > 2) may indicate mean-reversion opportunity.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with z-scores
    """
    prices = df["close"]
    rolling_mean = prices.rolling(window=10, min_periods=5).mean()
    rolling_std = prices.rolling(window=10, min_periods=5).std()

    zscore = (prices - rolling_mean) / rolling_std.replace(0, np.nan)

    return zscore


def compute_mr_zscore_20(df: pd.DataFrame) -> pd.Series:
    """
    20-period Z-score of price relative to rolling mean.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with z-scores
    """
    prices = df["close"]
    rolling_mean = prices.rolling(window=20, min_periods=10).mean()
    rolling_std = prices.rolling(window=20, min_periods=10).std()

    zscore = (prices - rolling_mean) / rolling_std.replace(0, np.nan)

    return zscore


def compute_mr_zscore_60(df: pd.DataFrame) -> pd.Series:
    """
    60-period Z-score of price relative to rolling mean.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with z-scores
    """
    prices = df["close"]
    rolling_mean = prices.rolling(window=60, min_periods=30).mean()
    rolling_std = prices.rolling(window=60, min_periods=30).std()

    zscore = (prices - rolling_mean) / rolling_std.replace(0, np.nan)

    return zscore


# =============================================================================
# ORNSTEIN-UHLENBECK HALF-LIFE
# =============================================================================


def _calc_halflife(x: np.ndarray) -> float:
    """Calculate OU half-life for a log-price series.

    Matches the Numba path: regresses log-returns on lagged log-prices
    (the standard OU specification: delta_X = a + beta * X_{t-1}).
    Half-life = -ln(2) / beta.
    """
    n = len(x)
    if n < 10:
        return np.nan

    # OU regression: delta_x = a + beta * x_{t-1}
    y = x[1:] - x[:-1]  # log-returns
    y_lag = x[:-1]  # lagged log-prices

    n_obs = len(y)
    if n_obs < 2:
        return np.nan

    x_mean = y_lag.mean()
    y_mean = y.mean()

    numerator = np.sum((y_lag - x_mean) * (y - y_mean))
    denominator = np.sum((y_lag - x_mean) ** 2)

    if denominator == 0:
        return np.nan

    beta = numerator / denominator

    # Mean reversion requires beta < 0
    MAX_HALFLIFE = 120.0
    if beta >= 0:
        return MAX_HALFLIFE  # No mean reversion

    halflife = -np.log(2) / beta
    return min(halflife, MAX_HALFLIFE)


if NUMBA_AVAILABLE:
    from numba import njit as _njit_halflife

    @_njit_halflife(cache=True)
    def _calc_halflife_numba(x: np.ndarray) -> float:
        """Numba-accelerated OU half-life calculation.

        Expects log prices as input (caller must pass np.log(close)).
        """
        n = len(x)
        if n < 10:
            return np.nan
        # Delta and lagged (x is already log prices)
        y = np.empty(n - 1)
        x_lag = np.empty(n - 1)
        for i in range(n - 1):
            y[i] = x[i + 1] - x[i]
            x_lag[i] = x[i]
        # Simple OLS: y = a + b * x_lag
        n_obs = n - 1
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        for i in range(n_obs):
            sum_x += x_lag[i]
            sum_y += y[i]
            sum_xx += x_lag[i] * x_lag[i]
            sum_xy += x_lag[i] * y[i]
        denom = n_obs * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            return np.nan
        beta = (n_obs * sum_xy - sum_x * sum_y) / denom
        if beta >= 0:
            return np.nan
        return -np.log(2) / beta

    _calc_halflife = _calc_halflife_numba


def compute_ou_halflife(df: pd.DataFrame) -> pd.Series:
    """
    Estimate Ornstein-Uhlenbeck mean-reversion half-life.

    Uses AR(1) regression to estimate the speed of mean reversion.
    Smaller values indicate faster mean reversion.

    Both paths (Numba and pure-Python) receive log prices as input
    to ensure consistent results regardless of which backend is active.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with half-life in bars (capped at MAX_HALFLIFE=120.0 if no mean reversion)
    """
    log_prices = np.log(df["close"])
    return log_prices.rolling(window=60, min_periods=20).apply(_calc_halflife, raw=True)


# =============================================================================
# VARIANCE RATIO FEATURES
# =============================================================================

# Try to import numba for accelerated variance ratio
try:
    from numba import jit as numba_jit

    @numba_jit(nopython=True)
    def _variance_ratio_numba(arr: np.ndarray, window: int, lag: int) -> np.ndarray:
        """
        Numba-accelerated rolling variance ratio calculation.

        VR = Var(k-period returns) / (k * Var(1-period returns))
        """
        n = len(arr)
        result = np.empty(n)
        result[:] = np.nan

        min_periods = max(lag * 2, 4)

        for i in range(window - 1, n):
            # Get window data
            window_data = arr[i - window + 1 : i + 1]

            # Count valid (non-nan) values
            valid_count = 0
            for v in window_data:
                if not np.isnan(v):
                    valid_count += 1

            if valid_count < min_periods:
                continue

            # Variance of 1-period returns
            var_sum = 0.0
            mean_1 = 0.0
            count_1 = 0
            for v in window_data:
                if not np.isnan(v):
                    mean_1 += v
                    count_1 += 1
            if count_1 == 0:
                continue
            mean_1 /= count_1

            for v in window_data:
                if not np.isnan(v):
                    var_sum += (v - mean_1) ** 2
            var_1 = var_sum / count_1

            if var_1 == 0:
                continue

            # Compute k-period returns (rolling sum of lag elements)
            # We have window_data of 1-period returns, compute sum of every lag consecutive
            k_returns_count = 0
            k_returns_sum = 0.0
            k_returns_sq_sum = 0.0

            for j in range(lag - 1, valid_count):
                k_sum = 0.0
                k_valid = True
                for k in range(lag):
                    idx = j - lag + 1 + k
                    if idx >= 0 and idx < len(window_data):
                        val = window_data[idx]
                        if np.isnan(val):
                            k_valid = False
                            break
                        k_sum += val
                    else:
                        k_valid = False
                        break

                if k_valid:
                    k_returns_sum += k_sum
                    k_returns_sq_sum += k_sum * k_sum
                    k_returns_count += 1

            if k_returns_count < 2:
                continue

            # Variance of k-period returns
            mean_k = k_returns_sum / k_returns_count
            var_k = (k_returns_sq_sum / k_returns_count) - (mean_k * mean_k)

            # VR = Var(k) / (k * Var(1))
            result[i] = var_k / (lag * var_1)

        return result

except ImportError:

    def _variance_ratio_numba(arr: np.ndarray, window: int, lag: int) -> np.ndarray:
        """Fallback non-numba version."""
        raise NotImplementedError("Numba not available")


def _calc_vr(x: pd.Series, lag: int) -> float:
    """Calculate variance ratio for a specific lag (fallback)."""
    if len(x) < lag * 2:
        return np.nan

    x_arr = x.values if hasattr(x, "values") else np.array(x)

    # Variance of 1-period returns
    var_1 = np.nanvar(x_arr)
    if var_1 == 0:
        return np.nan

    # Variance of k-period returns using cumsum trick
    cumsum = np.nancumsum(x_arr)
    k_returns = cumsum[lag - 1 :] - np.concatenate([[0], cumsum[:-lag]])
    if len(k_returns) < 2:
        return np.nan
    var_k = np.nanvar(k_returns)

    # VR = Var(k-period) / (k * Var(1-period))
    vr = var_k / (lag * var_1)
    return vr


def _compute_variance_ratio(df: pd.DataFrame, lag: int, window: int = 60) -> pd.Series:
    """
    Compute variance ratio with specified lag.

    Uses Numba-accelerated computation when available for ~10x speedup.
    """
    log_returns = np.log(df["close"]).diff()

    if NUMBA_AVAILABLE:
        # Use Numba-accelerated version
        result = _variance_ratio_numba(log_returns.values, window, lag)
        return pd.Series(result, index=log_returns.index)
    else:
        # Fallback to pandas apply
        min_periods = max(lag * 2, 4)
        return log_returns.rolling(window=window, min_periods=min_periods).apply(
            lambda x: _calc_vr(x, lag=lag), raw=True
        )


def compute_variance_ratio_2(df: pd.DataFrame) -> pd.Series:
    """
    Variance ratio test with lag 2.

    VR < 1 suggests mean reversion, VR > 1 suggests momentum.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with VR statistic
    """
    return _compute_variance_ratio(df, lag=2)


def compute_variance_ratio_4(df: pd.DataFrame) -> pd.Series:
    """
    Variance ratio test with lag 4.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with VR statistic
    """
    return _compute_variance_ratio(df, lag=4)


def compute_variance_ratio_8(df: pd.DataFrame) -> pd.Series:
    """
    Variance ratio test with lag 8.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with VR statistic
    """
    return _compute_variance_ratio(df, lag=8)


def compute_variance_ratio_16(df: pd.DataFrame) -> pd.Series:
    """
    Variance ratio test with lag 16.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with VR statistic
    """
    return _compute_variance_ratio(df, lag=16)


# =============================================================================
# MEAN REVERSION STRENGTH FEATURES
# =============================================================================


def compute_mr_strength(df: pd.DataFrame) -> pd.Series:
    """
    Mean reversion strength indicator.

    Combines z-score magnitude with Hurst exponent to indicate
    strength of mean-reversion signal.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with mean reversion strength (higher = stronger signal)
    """
    zscore = compute_mr_zscore_20(df)
    hurst = compute_hurst_100(df)

    # Strong MR: high |zscore| + low Hurst
    # Invert Hurst contribution (1 - Hurst) so low Hurst gives high value
    hurst_contrib = (0.5 - hurst.fillna(0.5)).clip(lower=0)

    strength = zscore.abs() * (1 + hurst_contrib)

    return strength


# =============================================================================
# AGGREGATE FEATURE COMPUTATION
# =============================================================================


def compute_mean_reversion_features(
    prices: pd.Series, windows: list[int] | None = None
) -> pd.DataFrame:
    """
    Compute all mean-reversion features for a price series.

    Args:
        prices: Price series (typically close prices)
        windows: List of windows for z-score calculation

    Returns:
        DataFrame with all mean-reversion features
    """
    if windows is None:
        windows = [10, 20, 60]

    # Create a DataFrame to work with
    df = pd.DataFrame({"close": prices})

    features = pd.DataFrame(index=prices.index)

    # Z-scores at different windows
    for w in windows:
        rolling_mean = prices.rolling(window=w, min_periods=w // 2).mean()
        rolling_std = prices.rolling(window=w, min_periods=w // 2).std()
        features[f"mr_zscore_{w}"] = (prices - rolling_mean) / rolling_std.replace(0, np.nan)

    # OU half-life (longer window needed)
    features["ou_halflife"] = compute_ou_halflife(df)

    # Hurst exponent (canonical implementation from entropy.py, with shift(1))
    features["hurst_exponent"] = compute_hurst_100(df)

    # Variance ratios
    features["variance_ratio_2"] = compute_variance_ratio_2(df)
    features["variance_ratio_4"] = compute_variance_ratio_4(df)
    features["variance_ratio_8"] = compute_variance_ratio_8(df)
    features["variance_ratio_16"] = compute_variance_ratio_16(df)

    return features


# =============================================================================
# FEATURE MAP
# =============================================================================

MEAN_REVERSION_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Z-Scores
    "mr_zscore_10": compute_mr_zscore_10,
    "mr_zscore_20": compute_mr_zscore_20,
    "mr_zscore_60": compute_mr_zscore_60,
    # OU Half-Life
    "ou_halflife": compute_ou_halflife,
    # Variance Ratios
    "variance_ratio_2": compute_variance_ratio_2,
    "variance_ratio_4": compute_variance_ratio_4,
    "variance_ratio_8": compute_variance_ratio_8,
    "variance_ratio_16": compute_variance_ratio_16,
    # Mean Reversion Strength
    "mr_strength": compute_mr_strength,
}

# Feature family metadata
FEATURE_FAMILY = "mean_reversion"
FEATURE_COUNT = 9

__all__ = [
    "MEAN_REVERSION_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # Z-Scores
    "compute_mr_zscore_10",
    "compute_mr_zscore_20",
    "compute_mr_zscore_60",
    # OU Half-Life
    "compute_ou_halflife",
    # Variance Ratios
    "compute_variance_ratio_2",
    "compute_variance_ratio_4",
    "compute_variance_ratio_8",
    "compute_variance_ratio_16",
    # Mean Reversion Strength
    "compute_mr_strength",
    # Aggregate
    "compute_mean_reversion_features",
]
