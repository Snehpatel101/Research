"""
Mean-reversion detection metrics for time series.

PHASE_19A ML Pipeline Enhancement: MEAN_REVERSION features.

These features detect mean-reverting behavior in price series,
useful for identifying trading opportunities and regime changes.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=1).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=window, min_periods=1).std()


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

    zscore = (prices - rolling_mean) / (rolling_std + 1e-10)

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

    zscore = (prices - rolling_mean) / (rolling_std + 1e-10)

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

    zscore = (prices - rolling_mean) / (rolling_std + 1e-10)

    return zscore


# =============================================================================
# ORNSTEIN-UHLENBECK HALF-LIFE
# =============================================================================


def _calc_halflife(x: np.ndarray) -> float:
    """Calculate OU half-life for a price series."""
    if len(x) < 10:
        return np.nan

    # AR(1): y_t = a + b * y_{t-1} + e
    y = x[1:]
    y_lag = x[:-1]

    # Simple OLS for speed
    n = len(y)
    if n < 2:
        return np.nan

    x_mean = y_lag.mean()
    y_mean = y.mean()

    numerator = np.sum((y_lag - x_mean) * (y - y_mean))
    denominator = np.sum((y_lag - x_mean) ** 2)

    if denominator == 0:
        return np.nan

    beta = numerator / denominator

    # Half-life = -ln(2) / ln(beta)
    # Cap at MAX_HALFLIFE instead of returning inf to prevent gradient explosion
    MAX_HALFLIFE = 120.0
    if beta <= 0 or beta >= 1:
        return MAX_HALFLIFE  # No mean reversion - return max cap instead of inf

    halflife = -np.log(2) / np.log(beta)
    return min(halflife, MAX_HALFLIFE)  # Cap at 2x window


def compute_ou_halflife(df: pd.DataFrame) -> pd.Series:
    """
    Estimate Ornstein-Uhlenbeck mean-reversion half-life.

    Uses AR(1) regression to estimate the speed of mean reversion.
    Smaller values indicate faster mean reversion.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with half-life in bars (capped at MAX_HALFLIFE=120.0 if no mean reversion)
    """
    log_prices = np.log(df["close"])

    return log_prices.rolling(window=60, min_periods=20).apply(_calc_halflife, raw=True)


# =============================================================================
# HURST EXPONENT
# =============================================================================

# Try to import numba for accelerated Hurst calculation
try:
    from numba import jit as hurst_jit

    @hurst_jit(nopython=True)
    def _hurst_rs_numba(x: np.ndarray, max_lag: int) -> float:
        """
        Numba-accelerated Hurst exponent using R/S analysis.

        O(max_lag * n/max_lag) = O(n) per window.
        """
        n = len(x)
        if n < max_lag * 2:
            return np.nan

        # Pre-compute lags
        n_lags = 0
        for _lag in range(2, min(max_lag + 1, n // 2)):
            n_lags += 1

        if n_lags < 3:
            return np.nan

        log_lags = np.empty(n_lags)
        log_rs = np.empty(n_lags)
        valid_count = 0

        lag_idx = 0
        for lag in range(2, min(max_lag + 1, n // 2)):
            n_chunks = n // lag
            if n_chunks < 1:
                continue

            rs_sum = 0.0
            rs_count = 0

            for i in range(n_chunks):
                start = i * lag
                end = (i + 1) * lag

                # Compute mean
                chunk_sum = 0.0
                for j in range(start, end):
                    chunk_sum += x[j]
                chunk_mean = chunk_sum / lag

                # Compute cumsum and std
                cumsum = 0.0
                cumsum_min = 0.0
                cumsum_max = 0.0
                var_sum = 0.0

                for j in range(start, end):
                    val = x[j] - chunk_mean
                    cumsum += val
                    if cumsum < cumsum_min:
                        cumsum_min = cumsum
                    if cumsum > cumsum_max:
                        cumsum_max = cumsum
                    var_sum += val * val

                r = cumsum_max - cumsum_min
                s = np.sqrt(var_sum / lag)

                if s > 0:
                    rs_sum += r / s
                    rs_count += 1

            if rs_count > 0:
                log_lags[valid_count] = np.log(lag)
                log_rs[valid_count] = np.log(rs_sum / rs_count)
                valid_count += 1

            lag_idx += 1

        if valid_count < 3:
            return np.nan

        # Simple linear regression for slope
        x_mean = 0.0
        y_mean = 0.0
        for i in range(valid_count):
            x_mean += log_lags[i]
            y_mean += log_rs[i]
        x_mean /= valid_count
        y_mean /= valid_count

        numerator = 0.0
        denominator = 0.0
        for i in range(valid_count):
            x_diff = log_lags[i] - x_mean
            numerator += x_diff * (log_rs[i] - y_mean)
            denominator += x_diff * x_diff

        if denominator == 0:
            return np.nan

        slope = numerator / denominator

        # Clip to valid range
        if slope < 0:
            return 0.0
        elif slope > 1:
            return 1.0
        return slope

    @hurst_jit(nopython=True)
    def _rolling_hurst_numba(
        arr: np.ndarray, window: int, min_periods: int, max_lag: int
    ) -> np.ndarray:
        """
        Numba-accelerated rolling Hurst calculation.

        Total complexity: O(n * max_lag) instead of O(n^2).
        """
        n = len(arr)
        result = np.empty(n)
        result[:] = np.nan

        for i in range(window - 1, n):
            window_data = arr[i - window + 1 : i + 1]
            # Count non-nan values
            valid = 0
            for v in window_data:
                if not np.isnan(v):
                    valid += 1
            if valid >= min_periods:
                result[i] = _hurst_rs_numba(window_data, max_lag)

        return result

    HURST_NUMBA_AVAILABLE = True
except ImportError:
    HURST_NUMBA_AVAILABLE = False


def _calc_hurst(x: np.ndarray, max_lag: int = 20) -> float:
    """Calculate Hurst exponent using R/S analysis (fallback)."""
    if len(x) < max_lag * 2:
        return np.nan

    lags = range(2, min(max_lag + 1, len(x) // 2))
    rs_values = []

    for lag in lags:
        n_chunks = len(x) // lag
        if n_chunks < 1:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = x[i * lag : (i + 1) * lag]
            if len(chunk) < 2:
                continue

            mean_adj = chunk - np.mean(chunk)
            cumsum = np.cumsum(mean_adj)
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(chunk)

            if s > 0:
                rs_chunk.append(r / s)

        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))

    if len(rs_values) < 3:
        return np.nan

    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)

    return np.clip(slope, 0, 1)


def compute_hurst_exponent(df: pd.DataFrame) -> pd.Series:
    """
    Estimate Hurst exponent using R/S analysis.

    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending/momentum

    Uses Numba-accelerated O(n * max_lag) algorithm when available
    instead of O(n^2) pandas rolling approach.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with Hurst exponent estimates
    """
    log_prices = np.log(df["close"])

    if HURST_NUMBA_AVAILABLE:
        result = _rolling_hurst_numba(log_prices.values, window=100, min_periods=40, max_lag=20)
        return pd.Series(result, index=log_prices.index)
    else:
        # Fallback to pandas apply
        return log_prices.rolling(window=100, min_periods=40).apply(
            lambda x: _calc_hurst(x, max_lag=20), raw=True
        )


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

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

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
    hurst = compute_hurst_exponent(df)

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
        features[f"mr_zscore_{w}"] = (prices - rolling_mean) / (rolling_std + 1e-10)

    # OU half-life (longer window needed)
    features["ou_halflife"] = compute_ou_halflife(df)

    # Hurst exponent
    features["hurst_exponent"] = compute_hurst_exponent(df)

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
    # Hurst Exponent
    "hurst_exponent": compute_hurst_exponent,
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
FEATURE_COUNT = 10

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
    # Hurst Exponent
    "compute_hurst_exponent",
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
