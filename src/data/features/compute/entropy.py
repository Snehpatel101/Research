"""
Entropy feature computation - Shannon, Lempel-Ziv, Approximate Entropy, Hurst.

PHASE_1 Unified Features: 12 ENTROPY features.

These features measure market complexity, randomness, and predictability.
"""

from collections.abc import Callable

import numba
import numpy as np
import pandas as pd

# =============================================================================
# NUMBA-ACCELERATED HELPER FUNCTIONS
# =============================================================================


@numba.njit(cache=True)
def _count_matches_numba(patterns: np.ndarray, r: float) -> int:
    """
    Numba-accelerated pattern matching for sample entropy.

    Counts pairs of patterns (i, j) where i < j and max|patterns[i] - patterns[j]| <= r.
    This replaces the O(n^2) Python loop with JIT-compiled code for ~50-100x speedup.

    Args:
        patterns: 2D array of shape (n_patterns, m) containing embedded patterns
        r: Tolerance threshold for pattern matching

    Returns:
        Count of matching pattern pairs
    """
    n_patterns = patterns.shape[0]
    m = patterns.shape[1]
    count = 0

    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            # Compute max absolute difference (Chebyshev distance)
            max_diff = 0.0
            for k in range(m):
                diff = abs(patterns[i, k] - patterns[j, k])
                if diff > max_diff:
                    max_diff = diff
            if max_diff <= r:
                count += 1

    return count


@numba.njit(cache=True)
def _count_matches_per_pattern_numba(patterns: np.ndarray, r: float) -> np.ndarray:
    """
    Numba-accelerated per-pattern match counting for approximate entropy.

    For each pattern i, counts ALL patterns j (including i itself) where
    max|patterns[i] - patterns[j]| <= r. This is needed for ApEn's phi calculation.

    Args:
        patterns: 2D array of shape (n_patterns, m) containing embedded patterns
        r: Tolerance threshold for pattern matching

    Returns:
        Array of match counts (one per pattern), normalized by n_patterns
    """
    n_patterns = patterns.shape[0]
    m = patterns.shape[1]
    counts = np.zeros(n_patterns, dtype=np.float64)

    for i in range(n_patterns):
        match_count = 0
        for j in range(n_patterns):
            # Compute max absolute difference (Chebyshev distance)
            max_diff = 0.0
            for k in range(m):
                diff = abs(patterns[i, k] - patterns[j, k])
                if diff > max_diff:
                    max_diff = diff
            if max_diff <= r:
                match_count += 1
        # Normalize by total patterns
        counts[i] = match_count / n_patterns

    return counts

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _log_returns(close: pd.Series) -> pd.Series:
    """Calculate log returns."""
    return np.log(close / close.shift(1))


def _discretize_returns(returns: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Discretize continuous returns into bins for entropy calculation.

    Uses percentile-based binning to handle varying return distributions.
    """
    if len(returns) < n_bins:
        return np.zeros(len(returns), dtype=int)

    # Remove NaN values for percentile calculation
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) < n_bins:
        return np.zeros(len(returns), dtype=int)

    # Create percentile-based bins
    percentiles = np.percentile(valid_returns, np.linspace(0, 100, n_bins + 1))
    bins = np.digitize(returns, percentiles[:-1]) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    return bins


def _shannon_entropy(x: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Shannon entropy of a discretized series.

    H = -sum(p * log2(p)) for all probability bins
    Higher values indicate more randomness/unpredictability.
    """
    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan

    # Discretize
    bins = _discretize_returns(x, n_bins)

    # Calculate probabilities
    _, counts = np.unique(bins[~np.isnan(x)], return_counts=True)
    probs = counts / counts.sum()

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return float(entropy)


def _lempel_ziv_complexity(binary_seq: np.ndarray) -> float:
    """
    Calculate Lempel-Ziv complexity of a binary sequence.

    Higher values indicate more complexity/randomness.
    """
    n = len(binary_seq)
    if n == 0:
        return np.nan

    binary_str = "".join(binary_seq.astype(str))

    # LZ76 algorithm (l = length, standard academic notation)
    i = 0
    c = 1
    l = 1  # noqa: E741 - standard LZ76 variable name for "length"
    k = 1
    k_max = 1

    while True:
        if binary_str[i + k - 1] != binary_str[l + k - 1]:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max  # noqa: E741
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
        else:
            k += 1
            if l + k > n:
                c += 1
                break

    # Normalize by theoretical maximum (n / log2(n))
    normalized = c * np.log2(n) / n if n > 1 else 0

    return normalized


def _approximate_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """
    Calculate Approximate Entropy (ApEn).

    ApEn measures the amount of regularity and unpredictability in a time series.
    Lower values indicate more regularity/predictability.

    Uses numba-accelerated pattern matching for O(n^2) loops (~50-100x speedup).

    Args:
        x: Time series data
        m: Embedding dimension (pattern length)
        r: Tolerance (typically 0.1-0.25 * std of x)
    """
    n = len(x)
    if n < m + 1:
        return np.nan

    x = x[~np.isnan(x)]
    n = len(x)
    if n < m + 1:
        return np.nan

    if r is None:
        r = 0.2 * np.std(x)

    def _phi(m_val: int) -> float:
        """Calculate phi for given m using numba acceleration."""
        # Create patterns array - contiguous for numba
        n_patterns = n - m_val + 1
        patterns = np.empty((n_patterns, m_val), dtype=np.float64)
        for i in range(n_patterns):
            patterns[i] = x[i : i + m_val]

        # Use numba-accelerated per-pattern match counting
        counts = _count_matches_per_pattern_numba(patterns, r)

        return float(np.mean(np.log(counts + 1e-10)))

    return float(_phi(m) - _phi(m + 1))


def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """
    Calculate Sample Entropy (SampEn).

    Similar to ApEn but less biased for short time series.
    Does not count self-matches.

    Uses numba-accelerated pattern matching for O(n^2) loops.
    """
    n = len(x)
    if n < m + 1:
        return np.nan

    x = x[~np.isnan(x)]
    n = len(x)
    if n < m + 1:
        return np.nan

    if r is None:
        r = 0.2 * np.std(x)

    def _count_matches(m_val: int) -> int:
        """Count matches for embedding dimension m using numba acceleration."""
        # Create patterns array - contiguous for numba
        n_patterns = n - m_val
        patterns = np.empty((n_patterns, m_val), dtype=np.float64)
        for i in range(n_patterns):
            patterns[i] = x[i : i + m_val]

        return _count_matches_numba(patterns, r)

    a = _count_matches(m + 1)
    b = _count_matches(m)

    if b == 0:
        return np.nan

    return -np.log(a / b) if a > 0 else np.nan


def _hurst_exponent(x: np.ndarray) -> float:
    """
    Calculate Hurst exponent using R/S analysis.

    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending/persistent
    """
    n = len(x)
    if n < 20:
        return np.nan

    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20:
        return np.nan

    # Different chunk sizes
    sizes = []
    rs_values = []

    # Use various window sizes
    for size in [10, 20, 30, 40, 50]:
        if size >= n:
            continue

        # Number of chunks
        n_chunks = n // size

        rs_chunk = []
        for i in range(n_chunks):
            chunk = x[i * size : (i + 1) * size]

            # Mean-adjusted series
            mean_adj = chunk - np.mean(chunk)

            # Cumulative sum
            cum_sum = np.cumsum(mean_adj)

            # Range
            r = np.max(cum_sum) - np.min(cum_sum)

            # Standard deviation
            s = np.std(chunk)

            if s > 0:
                rs_chunk.append(r / s)

        if rs_chunk:
            sizes.append(size)
            rs_values.append(np.mean(rs_chunk))

    if len(sizes) < 2:
        return np.nan

    # Linear regression on log-log scale
    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)

    # Hurst exponent is the slope
    slope = np.polyfit(log_sizes, log_rs, 1)[0]

    return float(slope)


# =============================================================================
# SHANNON ENTROPY FEATURES
# =============================================================================


def compute_entropy_shannon_10(df: pd.DataFrame) -> pd.Series:
    """10-period Shannon entropy of returns."""
    returns = _log_returns(df["close"])

    return returns.rolling(window=10, min_periods=10).apply(_shannon_entropy, raw=True)


def compute_entropy_shannon_20(df: pd.DataFrame) -> pd.Series:
    """20-period Shannon entropy of returns."""
    returns = _log_returns(df["close"])

    return returns.rolling(window=20, min_periods=20).apply(_shannon_entropy, raw=True)


def compute_entropy_shannon_50(df: pd.DataFrame) -> pd.Series:
    """50-period Shannon entropy of returns."""
    returns = _log_returns(df["close"])

    return returns.rolling(window=50, min_periods=50).apply(_shannon_entropy, raw=True)


def compute_entropy_shannon_norm_20(df: pd.DataFrame) -> pd.Series:
    """
    Normalized 20-period Shannon entropy.

    Normalized by maximum possible entropy (log2(n_bins)).
    """
    returns = _log_returns(df["close"])
    n_bins = 10
    max_entropy = np.log2(n_bins)

    def rolling_shannon_norm(x):
        return _shannon_entropy(x, n_bins) / max_entropy

    return returns.rolling(window=20, min_periods=20).apply(rolling_shannon_norm, raw=True)


# =============================================================================
# LEMPEL-ZIV COMPLEXITY FEATURES
# =============================================================================


def compute_entropy_lz_20(df: pd.DataFrame) -> pd.Series:
    """20-period Lempel-Ziv complexity."""
    returns = _log_returns(df["close"])

    def rolling_lz(x):
        # Binarize: 1 if return > 0, 0 otherwise
        binary = (x > 0).astype(int)
        return _lempel_ziv_complexity(binary)

    return returns.rolling(window=20, min_periods=20).apply(rolling_lz, raw=True)


def compute_entropy_lz_50(df: pd.DataFrame) -> pd.Series:
    """50-period Lempel-Ziv complexity."""
    returns = _log_returns(df["close"])

    def rolling_lz(x):
        binary = (x > 0).astype(int)
        return _lempel_ziv_complexity(binary)

    return returns.rolling(window=50, min_periods=50).apply(rolling_lz, raw=True)


# =============================================================================
# APPROXIMATE ENTROPY FEATURES
# =============================================================================


def compute_entropy_apen_20(df: pd.DataFrame) -> pd.Series:
    """20-period Approximate Entropy."""
    returns = _log_returns(df["close"])

    def rolling_apen(x):
        return _approximate_entropy(x, m=2)

    return returns.rolling(window=20, min_periods=20).apply(rolling_apen, raw=True)


def compute_entropy_apen_50(df: pd.DataFrame) -> pd.Series:
    """50-period Approximate Entropy."""
    returns = _log_returns(df["close"])

    def rolling_apen(x):
        return _approximate_entropy(x, m=2)

    return returns.rolling(window=50, min_periods=50).apply(rolling_apen, raw=True)


# =============================================================================
# SAMPLE ENTROPY FEATURES
# =============================================================================


def compute_sample_entropy_20(df: pd.DataFrame) -> pd.Series:
    """20-period Sample Entropy."""
    returns = _log_returns(df["close"])

    def rolling_sampen(x):
        return _sample_entropy(x, m=2)

    return returns.rolling(window=20, min_periods=20).apply(rolling_sampen, raw=True)


# =============================================================================
# HURST EXPONENT FEATURES
# =============================================================================


def compute_hurst_50(df: pd.DataFrame) -> pd.Series:
    """50-period Hurst exponent."""
    returns = _log_returns(df["close"])

    return returns.rolling(window=50, min_periods=50).apply(_hurst_exponent, raw=True)


def compute_hurst_100(df: pd.DataFrame) -> pd.Series:
    """100-period Hurst exponent."""
    returns = _log_returns(df["close"])

    return returns.rolling(window=100, min_periods=100).apply(_hurst_exponent, raw=True)


def compute_hurst_regime(df: pd.DataFrame) -> pd.Series:
    """
    Hurst-based regime indicator.

    Returns:
        -1: Mean-reverting (H < 0.45)
        0: Random walk (0.45 <= H <= 0.55)
        1: Trending (H > 0.55)
    """
    hurst = compute_hurst_50(df)

    regime = pd.Series(0.0, index=df.index)
    regime[hurst < 0.45] = -1.0
    regime[hurst > 0.55] = 1.0
    regime[hurst.isna()] = np.nan

    return regime


# =============================================================================
# FEATURE MAP
# =============================================================================

ENTROPY_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Shannon
    "entropy_shannon_10": compute_entropy_shannon_10,
    "entropy_shannon_20": compute_entropy_shannon_20,
    "entropy_shannon_50": compute_entropy_shannon_50,
    "entropy_shannon_norm_20": compute_entropy_shannon_norm_20,
    # Lempel-Ziv
    "entropy_lz_20": compute_entropy_lz_20,
    "entropy_lz_50": compute_entropy_lz_50,
    # Approximate Entropy
    "entropy_apen_20": compute_entropy_apen_20,
    "entropy_apen_50": compute_entropy_apen_50,
    # Sample Entropy
    "sample_entropy_20": compute_sample_entropy_20,
    # Hurst
    "hurst_50": compute_hurst_50,
    "hurst_100": compute_hurst_100,
    "hurst_regime": compute_hurst_regime,
}

# Feature family metadata
FEATURE_FAMILY = "entropy"
FEATURE_COUNT = 12

__all__ = [
    "ENTROPY_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # Shannon
    "compute_entropy_shannon_10",
    "compute_entropy_shannon_20",
    "compute_entropy_shannon_50",
    "compute_entropy_shannon_norm_20",
    # Lempel-Ziv
    "compute_entropy_lz_20",
    "compute_entropy_lz_50",
    # Approximate Entropy
    "compute_entropy_apen_20",
    "compute_entropy_apen_50",
    # Sample Entropy
    "compute_sample_entropy_20",
    # Hurst
    "compute_hurst_50",
    "compute_hurst_100",
    "compute_hurst_regime",
]
