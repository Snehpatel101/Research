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
    if beta <= 0 or beta >= 1:
        return np.inf  # No mean reversion

    halflife = -np.log(2) / np.log(beta)
    return min(halflife, 120.0)  # Cap at 2x window


def compute_ou_halflife(df: pd.DataFrame) -> pd.Series:
    """
    Estimate Ornstein-Uhlenbeck mean-reversion half-life.

    Uses AR(1) regression to estimate the speed of mean reversion.
    Smaller values indicate faster mean reversion.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with half-life in bars (np.inf if no mean reversion)
    """
    log_prices = np.log(df["close"])

    return log_prices.rolling(window=60, min_periods=20).apply(_calc_halflife, raw=True)


# =============================================================================
# HURST EXPONENT
# =============================================================================


def _calc_hurst(x: np.ndarray, max_lag: int = 20) -> float:
    """Calculate Hurst exponent using R/S analysis."""
    if len(x) < max_lag * 2:
        return np.nan

    lags = range(2, min(max_lag + 1, len(x) // 2))
    rs_values = []

    for lag in lags:
        # Split into non-overlapping chunks of size lag
        n_chunks = len(x) // lag
        if n_chunks < 1:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = x[i * lag : (i + 1) * lag]
            if len(chunk) < 2:
                continue

            # Cumulative deviation from mean
            mean_adj = chunk - np.mean(chunk)
            cumsum = np.cumsum(mean_adj)

            # Range
            r = np.max(cumsum) - np.min(cumsum)
            # Standard deviation
            s = np.std(chunk)

            if s > 0:
                rs_chunk.append(r / s)

        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))

    if len(rs_values) < 3:
        return np.nan

    # Log-log regression to find Hurst exponent
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

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with Hurst exponent estimates
    """
    log_prices = np.log(df["close"])

    return log_prices.rolling(window=100, min_periods=40).apply(
        lambda x: _calc_hurst(x, max_lag=20), raw=True
    )


# =============================================================================
# VARIANCE RATIO FEATURES
# =============================================================================


def _calc_vr(x: pd.Series, lag: int) -> float:
    """Calculate variance ratio for a specific lag."""
    if len(x) < lag * 2:
        return np.nan

    # Variance of 1-period returns
    var_1 = np.var(x)
    if var_1 == 0:
        return np.nan

    # Variance of k-period returns
    k_returns = x.rolling(window=lag).sum().dropna()
    if len(k_returns) < 2:
        return np.nan
    var_k = np.var(k_returns)

    # VR = Var(k-period) / (k * Var(1-period))
    vr = var_k / (lag * var_1)
    return vr


def compute_variance_ratio_2(df: pd.DataFrame) -> pd.Series:
    """
    Variance ratio test with lag 2.

    VR < 1 suggests mean reversion, VR > 1 suggests momentum.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with VR statistic
    """
    log_returns = np.log(df["close"]).diff()

    return log_returns.rolling(window=60, min_periods=4).apply(
        lambda x: _calc_vr(x, lag=2), raw=False
    )


def compute_variance_ratio_4(df: pd.DataFrame) -> pd.Series:
    """
    Variance ratio test with lag 4.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with VR statistic
    """
    log_returns = np.log(df["close"]).diff()

    return log_returns.rolling(window=60, min_periods=8).apply(
        lambda x: _calc_vr(x, lag=4), raw=False
    )


def compute_variance_ratio_8(df: pd.DataFrame) -> pd.Series:
    """
    Variance ratio test with lag 8.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with VR statistic
    """
    log_returns = np.log(df["close"]).diff()

    return log_returns.rolling(window=60, min_periods=16).apply(
        lambda x: _calc_vr(x, lag=8), raw=False
    )


def compute_variance_ratio_16(df: pd.DataFrame) -> pd.Series:
    """
    Variance ratio test with lag 16.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with VR statistic
    """
    log_returns = np.log(df["close"]).diff()

    return log_returns.rolling(window=60, min_periods=32).apply(
        lambda x: _calc_vr(x, lag=16), raw=False
    )


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
