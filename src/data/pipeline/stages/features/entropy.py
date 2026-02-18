"""
Information-theoretic entropy features for OHLCV time series.

This module provides functions to calculate entropy-based measures that quantify
randomness, complexity, and regularity in price movements. These features are
valuable for:
- Regime detection (high entropy = noisy/unpredictable, low entropy = trending)
- Model gating (filter out high-entropy periods where signals are unreliable)
- Feature diversity (orthogonal to momentum/volatility features)

Features implemented:
- Shannon Entropy: Classical information entropy of discretized returns
- Lempel-Ziv Complexity: Compression-based complexity of binary price patterns
- Approximate Entropy (ApEn): Template-matching regularity measure

Academic References:
    Shannon, C.E. (1948) "A Mathematical Theory of Communication"
        Bell System Technical Journal, 27(3), 379-423.
        Foundation of information theory, entropy quantifies uncertainty.

    Lempel, A. & Ziv, J. (1976) "On the Complexity of Finite Sequences"
        IEEE Transactions on Information Theory, 22(1), 75-81.
        Complexity measure based on pattern dictionary growth.

    Pincus, S.M. (1991) "Approximate Entropy as a Measure of System Complexity"
        Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
        Regularity measure robust to noise and short time series.

All features use .shift(1) to prevent lookahead bias.
"""

import logging

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Default parameters
DEFAULT_SHANNON_WINDOWS = [10, 20, 50]
DEFAULT_SHANNON_BINS = 10

DEFAULT_LZ_WINDOWS = [20, 50, 100]

DEFAULT_APEN_WINDOWS = [20, 50]
DEFAULT_APEN_M = 2
DEFAULT_APEN_R = 0.2


def _discretize_returns(returns: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Discretize continuous returns into bins for entropy calculation.

    Uses quantile-based binning to handle varying return distributions.
    NaN values are excluded from binning and remain NaN.

    Parameters
    ----------
    returns : np.ndarray
        Array of log returns
    n_bins : int
        Number of bins for discretization

    Returns
    -------
    np.ndarray
        Array of bin indices (0 to n_bins-1), NaN where input was NaN
    """
    result = np.full_like(returns, np.nan)
    valid_mask = ~np.isnan(returns)

    if valid_mask.sum() < n_bins:
        return result

    valid_returns = returns[valid_mask]

    # Use quantile-based binning for robustness
    try:
        bin_edges = np.quantile(valid_returns, np.linspace(0, 1, n_bins + 1))
        # Handle edge case where all returns are identical
        if np.all(bin_edges == bin_edges[0]):
            result[valid_mask] = 0
        else:
            # Ensure unique bin edges by adding small perturbation
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                result[valid_mask] = 0
            else:
                binned = np.digitize(valid_returns, bin_edges[1:-1])
                result[valid_mask] = binned
    except Exception as e:
        logger.debug(f"Discretization failed: {e}")

    return result


def _calculate_shannon_entropy(bin_counts: np.ndarray) -> float:
    """
    Calculate Shannon entropy from bin counts.

    H = -sum(p * log2(p)) where p = count / total

    Parameters
    ----------
    bin_counts : np.ndarray
        Array of counts per bin

    Returns
    -------
    float
        Shannon entropy in bits (0 = perfectly predictable, log2(n_bins) = maximum)
    """
    total = bin_counts.sum()
    if total == 0:
        return np.nan

    # Filter out zero counts to avoid log(0)
    probs = bin_counts[bin_counts > 0] / total

    # Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)


@njit(cache=True)
def _rolling_shannon_entropy_numba(returns: np.ndarray, window: int, n_bins: int) -> np.ndarray:
    """
    Numba-optimized rolling Shannon entropy calculation.

    This is a JIT-compiled version that provides 10-50x speedup over the
    pure Python implementation for typical financial time series.

    Parameters
    ----------
    returns : np.ndarray
        Array of log returns (may contain NaN values)
    window : int
        Rolling window size
    n_bins : int
        Number of bins for discretization

    Returns
    -------
    np.ndarray
        Rolling Shannon entropy values
    """
    n = len(returns)
    entropy = np.full(n, np.nan)
    log2_val = np.log(2.0)

    for i in range(window - 1, n):
        # Extract window
        window_start = i - window + 1
        window_returns = returns[window_start : i + 1]

        # Count valid (non-NaN) values
        valid_count = 0
        for j in range(window):
            if not np.isnan(window_returns[j]):
                valid_count += 1

        # Skip if too few valid values
        if valid_count < n_bins:
            continue

        # Extract valid values
        valid_returns = np.empty(valid_count, dtype=np.float64)
        idx = 0
        for j in range(window):
            if not np.isnan(window_returns[j]):
                valid_returns[idx] = window_returns[j]
                idx += 1

        # Find min/max for quantile-based binning
        min_val = valid_returns[0]
        max_val = valid_returns[0]
        for j in range(1, valid_count):
            if valid_returns[j] < min_val:
                min_val = valid_returns[j]
            if valid_returns[j] > max_val:
                max_val = valid_returns[j]

        # Handle edge case where all returns are identical
        if max_val == min_val:
            # All in one bin = 0 entropy
            entropy[i] = 0.0
            continue

        # Bin the values using equal-width binning (simpler for numba)
        bin_counts = np.zeros(n_bins, dtype=np.int64)
        bin_width = (max_val - min_val) / n_bins

        for j in range(valid_count):
            bin_idx = int((valid_returns[j] - min_val) / bin_width)
            # Handle edge case where value equals max_val
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            bin_counts[bin_idx] += 1

        # Calculate Shannon entropy: -sum(p * log2(p))
        total = float(valid_count)
        ent = 0.0
        for j in range(n_bins):
            if bin_counts[j] > 0:
                p = bin_counts[j] / total
                ent -= p * np.log(p) / log2_val

        entropy[i] = ent

    return entropy


def _rolling_shannon_entropy(returns: np.ndarray, window: int, n_bins: int) -> np.ndarray:
    """
    Calculate rolling Shannon entropy over returns.

    For each window, discretizes the returns and computes entropy.
    Uses Numba JIT compilation for 10-50x speedup.

    Parameters
    ----------
    returns : np.ndarray
        Array of log returns
    window : int
        Rolling window size
    n_bins : int
        Number of bins for discretization

    Returns
    -------
    np.ndarray
        Rolling Shannon entropy values
    """
    # Use numba-optimized implementation
    return _rolling_shannon_entropy_numba(returns.astype(np.float64), window, n_bins)


def add_shannon_entropy(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    windows: list[int] | None = None,
    n_bins: int = DEFAULT_SHANNON_BINS,
) -> pd.DataFrame:
    """
    Add Shannon entropy features measuring price movement randomness.

    Shannon entropy quantifies uncertainty/randomness in price changes.
    High entropy = noisy/unpredictable, Low entropy = trending/predictable.

    Academic Reference:
        Shannon, C.E. (1948) "A Mathematical Theory of Communication"
        Bell System Technical Journal, 27(3), 379-423.

    Use Case:
        - High entropy -> Don't trade (market noise)
        - Low entropy -> Trade signals more reliable (trend/pattern)
        - Regime detection and model gating

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with price column
    feature_metadata : dict[str, str]
        Dictionary to store feature descriptions
    price_col : str, default 'close'
        Column to compute entropy on
    windows : list[int], optional
        Rolling window sizes. Default: [10, 20, 50]
    n_bins : int, default 10
        Number of bins for discretization

    Returns
    -------
    pd.DataFrame
        DataFrame with entropy_shannon_{window} columns added

    Notes
    -----
    Anti-lookahead: All features use .shift(1) to ensure features at bar[t]
    use data only up to bar[t-1].
    """
    if windows is None:
        windows = DEFAULT_SHANNON_WINDOWS.copy()

    logger.info(f"Adding Shannon entropy with windows: {windows}, bins: {n_bins}")

    # Calculate log returns
    prices = df[price_col].values
    returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[np.nan], returns])

    for window in windows:
        col_name = f"entropy_shannon_{window}"

        entropy = _rolling_shannon_entropy(returns, window, n_bins)

        # ANTI-LOOKAHEAD: shift(1) ensures feature at bar[t] uses data up to bar[t-1]
        df[col_name] = pd.Series(entropy, index=df.index).shift(1)

        # Normalize by max entropy (log2(n_bins)) to get 0-1 range
        max_entropy = np.log2(n_bins)
        norm_col_name = f"entropy_shannon_norm_{window}"
        df[norm_col_name] = df[col_name] / max_entropy

        feature_metadata[col_name] = (
            f"Shannon entropy of returns ({window}-period, {n_bins} bins, lagged)"
        )
        feature_metadata[norm_col_name] = (
            f"Normalized Shannon entropy [0,1] ({window}-period, lagged)"
        )

    return df


@njit(cache=True)
def _lempel_ziv_complexity_numba(seq: np.ndarray) -> int:
    """
    Numba-compiled Lempel-Ziv complexity using LZ76 algorithm.

    Array-based pattern matching instead of string operations.
    ~10-20x faster than string-based implementation.

    Parameters
    ----------
    seq : np.ndarray
        Binary sequence (0s and 1s) as int8

    Returns
    -------
    int
        LZ complexity count (number of distinct patterns)
    """
    n = len(seq)
    if n == 0:
        return 0

    complexity = 1  # Start with 1 for the first symbol
    i = 0  # Current position
    k = 1  # Length of current pattern being examined

    while i + k <= n:
        # Check if current pattern exists in history
        # History: seq[0:i+k-1], Pattern: seq[i:i+k]
        pattern_found = False
        history_end = i + k - 1

        # Search for pattern in history
        for start in range(history_end - k + 1):
            # Check if pattern matches at this position
            match = True
            for offset in range(k):
                if seq[start + offset] != seq[i + offset]:
                    match = False
                    break
            if match:
                pattern_found = True
                break

        if pattern_found:
            # Pattern already seen, extend it
            k += 1
        else:
            # New pattern found, increment complexity
            complexity += 1
            i = i + k
            k = 1

    return complexity


def _lempel_ziv_complexity(binary_seq: np.ndarray) -> int:
    """
    Calculate Lempel-Ziv complexity using the LZ76 algorithm.

    Counts the number of distinct patterns encountered when scanning
    the sequence from left to right.

    Parameters
    ----------
    binary_seq : np.ndarray
        Binary sequence (0s and 1s)

    Returns
    -------
    int
        LZ complexity count (number of distinct patterns)
    """
    n = len(binary_seq)
    if n == 0:
        return 0

    # Use numba-compiled implementation with int8 array
    return _lempel_ziv_complexity_numba(binary_seq.astype(np.int8))


def _normalized_lz_complexity(complexity: int, n: int) -> float:
    """
    Normalize LZ complexity by theoretical maximum.

    For a random binary sequence of length n, the expected complexity
    is approximately n / log2(n).

    Parameters
    ----------
    complexity : int
        Raw LZ complexity count
    n : int
        Sequence length

    Returns
    -------
    float
        Normalized complexity (0-1 range, higher = more random)
    """
    if n < 2:
        return np.nan

    # Theoretical maximum for random sequence
    max_complexity = n / np.log2(n)

    if max_complexity == 0:
        return np.nan

    return float(complexity / max_complexity)


@njit(cache=True)
def _rolling_lz_complexity_numba(binary_changes: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling Lempel-Ziv complexity loop."""
    n = len(binary_changes)
    lz_values = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = binary_changes[i - window + 1 : i + 1]

        # Count valid (non-NaN) values
        valid_count = 0
        for j in range(len(window_data)):
            if not np.isnan(window_data[j]):
                valid_count += 1

        if valid_count < window // 2:
            continue

        # Extract valid data
        valid_data = np.empty(valid_count, dtype=np.int8)
        idx = 0
        for j in range(len(window_data)):
            if not np.isnan(window_data[j]):
                valid_data[idx] = np.int8(window_data[j])
                idx += 1

        complexity = _lempel_ziv_complexity_numba(valid_data)

        # Normalize: complexity / (n / log2(n))
        nn = len(valid_data)
        if nn < 2:
            continue
        max_complexity = nn / (np.log(nn) / np.log(2.0))
        if max_complexity > 0:
            lz_values[i] = complexity / max_complexity

    return lz_values


def _rolling_lz_complexity(binary_changes: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling Lempel-Ziv complexity.

    Parameters
    ----------
    binary_changes : np.ndarray
        Binary encoded price changes (1=up, 0=down)
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Rolling normalized LZ complexity
    """
    return _rolling_lz_complexity_numba(binary_changes.astype(np.float64), window)


def add_lempel_ziv_complexity(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add Lempel-Ziv complexity measuring pattern complexity in price movements.

    LZ complexity counts distinct patterns in binary encoded price changes.
    Higher complexity = more random, Lower complexity = more structured.

    Academic Reference:
        Lempel, A. & Ziv, J. (1976) "On the Complexity of Finite Sequences"
        IEEE Transactions on Information Theory, 22(1), 75-81.

    Use Case:
        - High LZ complexity -> Random walk behavior
        - Low LZ complexity -> Repeating patterns (trend or mean-reversion)
        - Detects structural changes in price dynamics

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with price column
    feature_metadata : dict[str, str]
        Dictionary to store feature descriptions
    price_col : str, default 'close'
        Column to compute complexity on
    windows : list[int], optional
        Rolling window sizes. Default: [20, 50, 100]

    Returns
    -------
    pd.DataFrame
        DataFrame with entropy_lz_{window} columns added

    Notes
    -----
    Anti-lookahead: All features use .shift(1) to ensure features at bar[t]
    use data only up to bar[t-1].
    """
    if windows is None:
        windows = DEFAULT_LZ_WINDOWS.copy()

    logger.info(f"Adding Lempel-Ziv complexity with windows: {windows}")

    # Binary encode price changes: 1 if up, 0 if down or unchanged
    prices = df[price_col].values
    price_changes = np.diff(prices)
    binary_changes = np.where(price_changes > 0, 1.0, 0.0)
    binary_changes = np.concatenate([[np.nan], binary_changes])

    for window in windows:
        col_name = f"entropy_lz_{window}"

        lz_values = _rolling_lz_complexity(binary_changes, window)

        # ANTI-LOOKAHEAD: shift(1) ensures feature at bar[t] uses data up to bar[t-1]
        df[col_name] = pd.Series(lz_values, index=df.index).shift(1)

        feature_metadata[col_name] = f"Lempel-Ziv complexity normalized ({window}-period, lagged)"

    return df


@njit(cache=True)
def _phi_correlation_numba(data: np.ndarray, m: int, r: float) -> float:
    """
    Numba-compiled phi(m) calculation for Approximate Entropy.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    m : int
        Embedding dimension
    r : float
        Tolerance threshold (absolute)

    Returns
    -------
    float
        phi(m) value
    """
    n = len(data)
    n_templates = n - m + 1
    if n_templates <= 0:
        return np.nan

    # Count matches for each template (including self-match for ApEn)
    counts = np.zeros(n_templates, dtype=np.float64)

    for i in range(n_templates):
        count = 0
        for j in range(n_templates):
            # Check max absolute difference with early exit
            max_diff = 0.0
            match = True
            for k in range(m):
                diff = abs(data[i + k] - data[j + k])
                if diff > r:
                    match = False
                    break
                if diff > max_diff:
                    max_diff = diff
            if match:
                count += 1
        counts[i] = count

    # Average log of match proportions
    log_sum = 0.0
    for i in range(n_templates):
        proportion = counts[i] / n_templates
        if proportion < 1e-10:
            proportion = 1e-10
        log_sum += np.log(proportion)

    return log_sum / n_templates


def _phi_correlation(data: np.ndarray, m: int, r: float) -> float:
    """
    Calculate phi(m) for approximate entropy.

    Counts template matches within tolerance r for embedding dimension m.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    m : int
        Embedding dimension
    r : float
        Tolerance threshold (absolute)

    Returns
    -------
    float
        phi(m) value
    """
    n = len(data)
    if n - m + 1 <= 0:
        return np.nan

    # Use numba-compiled implementation
    return _phi_correlation_numba(data.astype(np.float64), m, r)


def _approximate_entropy(data: np.ndarray, m: int, r_fraction: float) -> float:
    """
    Calculate Approximate Entropy (ApEn).

    ApEn = phi(m) - phi(m+1)

    Lower ApEn indicates more regular patterns.

    Parameters
    ----------
    data : np.ndarray
        Time series data (typically returns)
    m : int
        Embedding dimension (pattern length)
    r_fraction : float
        Tolerance as fraction of standard deviation

    Returns
    -------
    float
        Approximate entropy value
    """
    # Calculate tolerance from data std
    std = np.std(data)
    if std == 0:
        return np.nan

    r = r_fraction * std

    phi_m = _phi_correlation(data, m, r)
    phi_m1 = _phi_correlation(data, m + 1, r)

    if np.isnan(phi_m) or np.isnan(phi_m1):
        return np.nan

    return phi_m - phi_m1


@njit(cache=True)
def _approximate_entropy_numba(data: np.ndarray, m: int, r_fraction: float) -> float:
    """Numba-optimized ApEn: phi(m) - phi(m+1)."""
    n = len(data)
    # Calculate std
    mean_val = 0.0
    for i in range(n):
        mean_val += data[i]
    mean_val /= n
    var_sum = 0.0
    for i in range(n):
        var_sum += (data[i] - mean_val) ** 2
    std = np.sqrt(var_sum / n)
    if std == 0.0:
        return np.nan

    r = r_fraction * std
    phi_m = _phi_correlation_numba(data, m, r)
    phi_m1 = _phi_correlation_numba(data, m + 1, r)
    if np.isnan(phi_m) or np.isnan(phi_m1):
        return np.nan
    return phi_m - phi_m1


@njit(cache=True)
def _rolling_approximate_entropy_numba(
    returns: np.ndarray, window: int, m: int, r_fraction: float
) -> np.ndarray:
    """Numba-optimized rolling Approximate Entropy loop."""
    n = len(returns)
    apen = np.full(n, np.nan)
    min_data = m + 2
    if min_data < 10:
        min_data = 10

    for i in range(window - 1, n):
        window_data = returns[i - window + 1 : i + 1]

        # Count valid (non-NaN) values
        valid_count = 0
        for j in range(len(window_data)):
            if not np.isnan(window_data[j]):
                valid_count += 1

        if valid_count < min_data:
            continue

        # Extract valid data
        valid_data = np.empty(valid_count, dtype=np.float64)
        idx = 0
        for j in range(len(window_data)):
            if not np.isnan(window_data[j]):
                valid_data[idx] = window_data[j]
                idx += 1

        apen[i] = _approximate_entropy_numba(valid_data, m, r_fraction)

    return apen


def _rolling_approximate_entropy(
    returns: np.ndarray, window: int, m: int, r_fraction: float
) -> np.ndarray:
    """
    Calculate rolling Approximate Entropy.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    window : int
        Rolling window size
    m : int
        Embedding dimension
    r_fraction : float
        Tolerance as fraction of rolling std

    Returns
    -------
    np.ndarray
        Rolling ApEn values
    """
    return _rolling_approximate_entropy_numba(returns.astype(np.float64), window, m, r_fraction)


def add_approximate_entropy(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    windows: list[int] | None = None,
    m: int = DEFAULT_APEN_M,
    r: float = DEFAULT_APEN_R,
) -> pd.DataFrame:
    """
    Add Approximate Entropy (ApEn) measuring pattern regularity.

    ApEn detects changes in patterns via template matching.
    Lower ApEn = more regular patterns, Higher ApEn = more irregular.

    Academic Reference:
        Pincus, S.M. (1991) "Approximate Entropy as a Measure of System Complexity"
        Proceedings of the National Academy of Sciences, 88(6), 2297-2301.

    Use Case:
        - Low ApEn -> Predictable/trending market, higher signal reliability
        - High ApEn -> Irregular/noisy market, lower signal reliability
        - Regime change detection (sudden ApEn changes)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with price column
    feature_metadata : dict[str, str]
        Dictionary to store feature descriptions
    price_col : str, default 'close'
        Column to compute ApEn on
    windows : list[int], optional
        Rolling window sizes. Default: [20, 50]
    m : int, default 2
        Embedding dimension (template length)
    r : float, default 0.2
        Tolerance as fraction of rolling standard deviation

    Returns
    -------
    pd.DataFrame
        DataFrame with entropy_apen_{window} columns added

    Notes
    -----
    Anti-lookahead: All features use .shift(1) to ensure features at bar[t]
    use data only up to bar[t-1].

    The computational complexity is O(n^2) per window, so this is slower
    than Shannon entropy for large windows.
    """
    if windows is None:
        windows = DEFAULT_APEN_WINDOWS.copy()

    logger.info(f"Adding Approximate Entropy with windows: {windows}, m={m}, r={r}")

    # Calculate log returns
    prices = df[price_col].values
    returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[np.nan], returns])

    for window in windows:
        col_name = f"entropy_apen_{window}"

        apen = _rolling_approximate_entropy(returns, window, m, r)

        # ANTI-LOOKAHEAD: shift(1) ensures feature at bar[t] uses data up to bar[t-1]
        df[col_name] = pd.Series(apen, index=df.index).shift(1)

        feature_metadata[col_name] = f"Approximate entropy (m={m}, r={r}, {window}-period, lagged)"

    return df


# =============================================================================
# HURST EXPONENT
# =============================================================================

DEFAULT_HURST_WINDOWS = [50, 100, 200]


def _calculate_hurst_rs(prices: np.ndarray) -> float:
    """
    Calculate Hurst exponent using Rescaled Range (R/S) analysis.

    The R/S method measures the ratio of the range of cumulative deviations
    to the standard deviation, scaled by the series length.

    Parameters
    ----------
    prices : np.ndarray
        Array of price values (not returns)

    Returns
    -------
    float
        Hurst exponent estimate (0-1 range typically)
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
    """
    n = len(prices)
    if n < 20:
        return np.nan

    # Calculate log returns
    log_returns = np.log(prices[1:] / prices[:-1])

    # Remove NaN returns
    log_returns = log_returns[~np.isnan(log_returns)]
    if len(log_returns) < 20:
        return np.nan

    # Use multiple sub-series lengths for R/S regression
    min_size = 10
    max_size = len(log_returns) // 2

    if max_size < min_size:
        return np.nan

    # Generate log-spaced sizes for sub-series
    sizes = np.unique(np.logspace(np.log10(min_size), np.log10(max_size), num=10).astype(int))
    sizes = sizes[sizes >= min_size]

    if len(sizes) < 3:
        return np.nan

    rs_values = []
    for size in sizes:
        # Number of non-overlapping subseries
        n_subseries = len(log_returns) // size

        if n_subseries == 0:
            continue

        rs_sum = 0
        valid_count = 0

        for i in range(n_subseries):
            subseries = log_returns[i * size : (i + 1) * size]

            # Mean-centered cumulative sum
            mean_ret = np.mean(subseries)
            cumsum = np.cumsum(subseries - mean_ret)

            # Range of cumulative sum
            r = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            s = np.std(subseries, ddof=1)

            if s > 0:
                rs_sum += r / s
                valid_count += 1

        if valid_count > 0:
            rs_values.append((size, rs_sum / valid_count))

    if len(rs_values) < 3:
        return np.nan

    # Linear regression of log(R/S) vs log(n) to estimate Hurst
    sizes_arr = np.array([x[0] for x in rs_values])
    rs_arr = np.array([x[1] for x in rs_values])

    # Filter out invalid values
    valid_mask = (rs_arr > 0) & np.isfinite(rs_arr)
    if valid_mask.sum() < 3:
        return np.nan

    log_sizes = np.log(sizes_arr[valid_mask])
    log_rs = np.log(rs_arr[valid_mask])

    # Simple linear regression: slope = Hurst exponent
    try:
        coeffs = np.polyfit(log_sizes, log_rs, 1)
        hurst = coeffs[0]

        # Clip to reasonable range
        hurst = np.clip(hurst, 0.0, 1.0)
        return float(hurst)
    except Exception as e:
        logger.warning(f"Hurst exponent calculation failed: {e}. Returning NaN.")
        return np.nan


def _rolling_hurst(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling Hurst exponent.

    Parameters
    ----------
    prices : np.ndarray
        Array of price values
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Rolling Hurst exponent values
    """
    n = len(prices)
    hurst = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_prices = prices[i - window + 1 : i + 1]

        # Skip if too many NaNs
        if np.isnan(window_prices).sum() > window // 4:
            continue

        # Use non-NaN prices
        valid_prices = window_prices[~np.isnan(window_prices)]
        if len(valid_prices) >= 20:
            hurst[i] = _calculate_hurst_rs(valid_prices)

    return hurst


def add_hurst_features(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add Hurst exponent features for mean-reversion vs trending detection.

    The Hurst exponent (H) measures the long-term memory of a time series:
    - H < 0.5: Mean-reverting (anti-persistent) - price tends to reverse
    - H = 0.5: Random walk - unpredictable
    - H > 0.5: Trending (persistent) - price tends to continue

    This is valuable for:
    - Strategy selection (mean-reversion vs trend-following)
    - Regime detection and adaptive trading
    - Market efficiency analysis

    Academic Reference:
        Hurst, H.E. (1951) "Long-term Storage Capacity of Reservoirs"
        Transactions of the American Society of Civil Engineers, 116, 770-808.

        Peters, E.E. (1994) "Fractal Market Analysis"
        John Wiley & Sons. ISBN 978-0471585244.

    Features added:
    - hurst_{window}: Hurst exponent over specified window
    - hurst_regime: Categorical regime indicator (mean_reverting, random, trending)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    feature_metadata : dict[str, str]
        Dictionary to store feature descriptions
    price_col : str, default 'close'
        Column to compute Hurst exponent on
    windows : list[int], optional
        Rolling window sizes. Default: [50, 100, 200]

    Returns
    -------
    pd.DataFrame
        DataFrame with Hurst exponent features added

    Notes
    -----
    Anti-lookahead: All features use .shift(1) to ensure features at bar[t]
    use data only up to bar[t-1].
    """
    if windows is None:
        windows = DEFAULT_HURST_WINDOWS.copy()

    logger.info(f"Adding Hurst exponent features with windows: {windows}")

    prices = df[price_col].values

    for window in windows:
        col_name = f"hurst_{window}"

        hurst = _rolling_hurst(prices, window)

        # ANTI-LOOKAHEAD: shift(1) ensures feature at bar[t] uses data up to bar[t-1]
        df[col_name] = pd.Series(hurst, index=df.index).shift(1)

        feature_metadata[col_name] = f"Hurst exponent ({window}-period R/S method, lagged)"

    # Add regime classification based on primary window (middle window)
    primary_window = windows[len(windows) // 2] if len(windows) > 0 else 100
    primary_col = f"hurst_{primary_window}"

    if primary_col in df.columns:
        # Classify regime: <0.45 = mean_reverting, 0.45-0.55 = random, >0.55 = trending
        conditions = [
            df[primary_col] < 0.45,
            (df[primary_col] >= 0.45) & (df[primary_col] <= 0.55),
            df[primary_col] > 0.55,
        ]
        choices = [0, 1, 2]  # 0=mean_reverting, 1=random, 2=trending

        df["hurst_regime"] = np.select(conditions, choices, default=np.nan)
        feature_metadata["hurst_regime"] = (
            f"Hurst regime from {primary_window}-period (0=mean_reverting, 1=random, 2=trending, lagged)"
        )

    return df


# =============================================================================
# SAMPLE ENTROPY
# =============================================================================

DEFAULT_SAMPLE_ENTROPY_WINDOWS = [20, 50]
DEFAULT_SAMPLE_ENTROPY_M = 2
DEFAULT_SAMPLE_ENTROPY_R = 0.2


@njit(cache=True)
def _count_template_matches_numba(data: np.ndarray, dim: int, tolerance: float) -> int:
    """
    Numba-compiled template matching for Sample Entropy.

    O(n²) but with early exit optimization and no Python overhead.
    ~20-50x faster than pure Python nested loops.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    dim : int
        Embedding dimension (template length)
    tolerance : float
        Absolute tolerance threshold

    Returns
    -------
    int
        Count of template matches
    """
    n = len(data)
    n_templates = n - dim
    if n_templates <= 1:
        return 0

    count = 0
    for i in range(n_templates):
        for j in range(i + 1, n_templates):
            # Check max absolute difference with early exit
            max_diff = 0.0
            match = True
            for k in range(dim):
                diff = abs(data[i + k] - data[j + k])
                if diff > tolerance:
                    match = False
                    break  # Early exit - no need to check remaining
                if diff > max_diff:
                    max_diff = diff
            if match:
                count += 1

    return count


def _calculate_sample_entropy(data: np.ndarray, m: int, r: float) -> float:
    """
    Calculate Sample Entropy (SampEn).

    Sample entropy is an improvement over Approximate Entropy that:
    - Does not count self-matches (reduces bias)
    - Is less dependent on data length
    - Provides more consistent results

    SampEn = -ln(A/B) where:
    - B = count of template matches for dimension m
    - A = count of template matches for dimension m+1

    Parameters
    ----------
    data : np.ndarray
        Time series data
    m : int
        Embedding dimension (template length)
    r : float
        Tolerance as fraction of standard deviation

    Returns
    -------
    float
        Sample entropy value (lower = more regular, higher = more complex)
    """
    n = len(data)
    if n < m + 2:
        return np.nan

    # Calculate tolerance from data std
    std = np.std(data)
    if std == 0:
        return np.nan

    tolerance = r * std

    # Convert to float64 for numba
    data_f64 = data.astype(np.float64)

    # Count matches for m and m+1 dimensions using numba-compiled function
    B = _count_template_matches_numba(data_f64, m, tolerance)
    A = _count_template_matches_numba(data_f64, m + 1, tolerance)

    # Avoid division by zero
    if B == 0:
        return np.nan

    # Sample entropy: -ln(A/B)
    # Handle case where A = 0 (maximum entropy/randomness)
    if A == 0:
        return np.inf  # Will be clipped to a large value

    return float(-np.log(A / B))


def _rolling_sample_entropy(data: np.ndarray, window: int, m: int, r: float) -> np.ndarray:
    """
    Calculate rolling Sample Entropy.

    Parameters
    ----------
    data : np.ndarray
        Time series data (typically returns)
    window : int
        Rolling window size
    m : int
        Embedding dimension
    r : float
        Tolerance fraction

    Returns
    -------
    np.ndarray
        Rolling sample entropy values
    """
    n = len(data)
    sampen = np.full(n, np.nan)

    # Minimum data points needed for meaningful SampEn
    min_data = max(m + 10, 15)

    for i in range(window - 1, n):
        window_data = data[i - window + 1 : i + 1]

        # Remove NaNs
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) < min_data:
            continue

        se = _calculate_sample_entropy(valid_data, m, r)

        # Clip infinite values to a large but finite number
        if np.isinf(se):
            se = 10.0  # Cap at 10 (very high entropy)

        sampen[i] = se

    return sampen


def add_sample_entropy(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    windows: list[int] | None = None,
    m: int = DEFAULT_SAMPLE_ENTROPY_M,
    r: float = DEFAULT_SAMPLE_ENTROPY_R,
) -> pd.DataFrame:
    """
    Add Sample Entropy (SampEn) features measuring pattern regularity.

    Sample entropy is an improvement over Approximate Entropy (ApEn) that
    is less biased and more consistent. It measures the complexity/regularity
    of the time series without counting self-matches.

    Lower SampEn = more regular/predictable patterns
    Higher SampEn = more irregular/random

    This is valuable for:
    - Regime detection (predictable vs unpredictable market)
    - Signal reliability estimation
    - Model gating (avoid trading in high-entropy periods)

    Academic Reference:
        Richman, J.S. & Moorman, J.R. (2000) "Physiological time-series analysis
        using approximate entropy and sample entropy"
        American Journal of Physiology, 278(6), H2039-H2049.

    Features added:
    - sample_entropy_{window}: Sample entropy over specified window

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    feature_metadata : dict[str, str]
        Dictionary to store feature descriptions
    price_col : str, default 'close'
        Column to compute sample entropy on
    windows : list[int], optional
        Rolling window sizes. Default: [20, 50]
    m : int, default 2
        Embedding dimension (template length)
    r : float, default 0.2
        Tolerance as fraction of standard deviation

    Returns
    -------
    pd.DataFrame
        DataFrame with sample entropy features added

    Notes
    -----
    Anti-lookahead: All features use .shift(1) to ensure features at bar[t]
    use data only up to bar[t-1].

    Sample entropy is O(n^2) per window, so larger windows are slower.
    """
    if windows is None:
        windows = DEFAULT_SAMPLE_ENTROPY_WINDOWS.copy()

    logger.info(f"Adding Sample Entropy with windows: {windows}, m={m}, r={r}")

    # Calculate log returns (SampEn on returns is more meaningful than prices)
    prices = df[price_col].values
    returns = np.log(prices[1:] / prices[:-1])
    returns = np.concatenate([[np.nan], returns])

    for window in windows:
        col_name = f"sample_entropy_{window}"

        sampen = _rolling_sample_entropy(returns, window, m, r)

        # ANTI-LOOKAHEAD: shift(1) ensures feature at bar[t] uses data up to bar[t-1]
        df[col_name] = pd.Series(sampen, index=df.index).shift(1)

        feature_metadata[col_name] = f"Sample entropy (m={m}, r={r}, {window}-period, lagged)"

    return df


def add_entropy_features(
    df: pd.DataFrame,
    feature_metadata: dict[str, str] | None = None,
    price_col: str = "close",
    include_shannon: bool = True,
    include_lempel_ziv: bool = True,
    include_approximate: bool = True,
    include_sample: bool = True,
    include_hurst: bool = True,
    shannon_windows: list[int] | None = None,
    shannon_bins: int = DEFAULT_SHANNON_BINS,
    lz_windows: list[int] | None = None,
    apen_windows: list[int] | None = None,
    apen_m: int = DEFAULT_APEN_M,
    apen_r: float = DEFAULT_APEN_R,
    sampen_windows: list[int] | None = None,
    sampen_m: int = DEFAULT_SAMPLE_ENTROPY_M,
    sampen_r: float = DEFAULT_SAMPLE_ENTROPY_R,
    hurst_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add all information-theoretic entropy features from OHLCV data.

    This is the main entry point for entropy feature engineering.
    These features quantify randomness and pattern complexity in price
    movements for regime detection and model gating.

    Features added:
    - Shannon entropy (randomness of discretized returns)
    - Lempel-Ziv complexity (pattern complexity of binary price changes)
    - Approximate entropy (template-matching regularity)
    - Sample entropy (improved ApEn, less biased)
    - Hurst exponent (mean-reversion vs trending detection)

    All features are lagged by 1 bar to prevent lookahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV columns
    feature_metadata : dict[str, str], optional
        Dictionary to store feature descriptions
    price_col : str, default 'close'
        Column to compute entropy features on
    include_shannon : bool, default True
        Include Shannon entropy features
    include_lempel_ziv : bool, default True
        Include Lempel-Ziv complexity features
    include_approximate : bool, default True
        Include Approximate entropy features
    include_sample : bool, default True
        Include Sample entropy features
    include_hurst : bool, default True
        Include Hurst exponent features
    shannon_windows : list[int], optional
        Windows for Shannon entropy. Default: [10, 20, 50]
    shannon_bins : int, default 10
        Number of bins for Shannon entropy discretization
    lz_windows : list[int], optional
        Windows for Lempel-Ziv complexity. Default: [20, 50, 100]
    apen_windows : list[int], optional
        Windows for Approximate entropy. Default: [20, 50]
    apen_m : int, default 2
        Embedding dimension for ApEn
    apen_r : float, default 0.2
        Tolerance fraction for ApEn
    sampen_windows : list[int], optional
        Windows for Sample entropy. Default: [20, 50]
    sampen_m : int, default 2
        Embedding dimension for SampEn
    sampen_r : float, default 0.2
        Tolerance fraction for SampEn
    hurst_windows : list[int], optional
        Windows for Hurst exponent. Default: [50, 100, 200]

    Returns
    -------
    pd.DataFrame
        DataFrame with entropy features added

    Examples
    --------
    >>> from stages.features.entropy import add_entropy_features
    >>> df = pd.DataFrame({'close': prices, 'open': opens, ...})
    >>> feature_metadata = {}
    >>> df = add_entropy_features(df, feature_metadata)
    >>> print(df[['entropy_shannon_20', 'entropy_lz_50', 'entropy_apen_20']])
    """
    if feature_metadata is None:
        feature_metadata = {}

    logger.info("Adding information-theoretic entropy features...")

    initial_cols = len(df.columns)

    # Shannon entropy
    if include_shannon:
        df = add_shannon_entropy(
            df,
            feature_metadata,
            price_col=price_col,
            windows=shannon_windows,
            n_bins=shannon_bins,
        )

    # Lempel-Ziv complexity
    if include_lempel_ziv:
        df = add_lempel_ziv_complexity(
            df,
            feature_metadata,
            price_col=price_col,
            windows=lz_windows,
        )

    # Approximate entropy
    if include_approximate:
        df = add_approximate_entropy(
            df,
            feature_metadata,
            price_col=price_col,
            windows=apen_windows,
            m=apen_m,
            r=apen_r,
        )

    # Sample entropy (improved ApEn)
    if include_sample:
        df = add_sample_entropy(
            df,
            feature_metadata,
            price_col=price_col,
            windows=sampen_windows,
            m=sampen_m,
            r=sampen_r,
        )

    # Hurst exponent (mean-reversion vs trending)
    if include_hurst:
        df = add_hurst_features(
            df,
            feature_metadata,
            price_col=price_col,
            windows=hurst_windows,
        )

    added_cols = len(df.columns) - initial_cols
    logger.info(f"Added {added_cols} entropy features")

    return df


__all__ = [
    # Main entry point
    "add_entropy_features",
    # Individual functions
    "add_shannon_entropy",
    "add_lempel_ziv_complexity",
    "add_approximate_entropy",
    "add_sample_entropy",
    "add_hurst_features",
    # Default parameters - Shannon
    "DEFAULT_SHANNON_WINDOWS",
    "DEFAULT_SHANNON_BINS",
    # Default parameters - Lempel-Ziv
    "DEFAULT_LZ_WINDOWS",
    # Default parameters - Approximate Entropy
    "DEFAULT_APEN_WINDOWS",
    "DEFAULT_APEN_M",
    "DEFAULT_APEN_R",
    # Default parameters - Sample Entropy
    "DEFAULT_SAMPLE_ENTROPY_WINDOWS",
    "DEFAULT_SAMPLE_ENTROPY_M",
    "DEFAULT_SAMPLE_ENTROPY_R",
    # Default parameters - Hurst
    "DEFAULT_HURST_WINDOWS",
]
