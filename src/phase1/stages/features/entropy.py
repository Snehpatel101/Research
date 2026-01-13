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
from typing import Optional

import numpy as np
import pandas as pd

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

    return entropy


def _rolling_shannon_entropy(
    returns: np.ndarray, window: int, n_bins: int
) -> np.ndarray:
    """
    Calculate rolling Shannon entropy over returns.

    For each window, discretizes the returns and computes entropy.

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
    n = len(returns)
    entropy = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_returns = returns[i - window + 1 : i + 1]

        # Skip if too many NaNs
        valid_count = np.sum(~np.isnan(window_returns))
        if valid_count < n_bins:
            continue

        # Discretize window returns
        binned = _discretize_returns(window_returns, n_bins)
        valid_binned = binned[~np.isnan(binned)]

        if len(valid_binned) == 0:
            continue

        # Count occurrences in each bin
        bin_counts = np.bincount(valid_binned.astype(int), minlength=n_bins)

        entropy[i] = _calculate_shannon_entropy(bin_counts)

    return entropy


def add_shannon_entropy(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    windows: Optional[list[int]] = None,
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

    # Convert to string for efficient substring operations
    seq_str = "".join(binary_seq.astype(int).astype(str))

    complexity = 1  # Start with 1 for the first symbol
    i = 0  # Current position
    k = 1  # Length of current pattern being examined

    while i + k <= n:
        # Check if current pattern (seq[i:i+k]) exists in the history (seq[0:i+k-1])
        current_pattern = seq_str[i : i + k]
        history = seq_str[: i + k - 1]

        if current_pattern in history:
            # Pattern already seen, extend it
            k += 1
        else:
            # New pattern found, increment complexity
            complexity += 1
            i = i + k
            k = 1

    return complexity


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

    return complexity / max_complexity


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
    n = len(binary_changes)
    lz_values = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = binary_changes[i - window + 1 : i + 1]

        # Skip if too many NaNs
        valid_mask = ~np.isnan(window_data)
        if valid_mask.sum() < window // 2:
            continue

        valid_data = window_data[valid_mask]
        complexity = _lempel_ziv_complexity(valid_data)
        lz_values[i] = _normalized_lz_complexity(complexity, len(valid_data))

    return lz_values


def add_lempel_ziv_complexity(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    windows: Optional[list[int]] = None,
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

        feature_metadata[col_name] = (
            f"Lempel-Ziv complexity normalized ({window}-period, lagged)"
        )

    return df


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

    # Create templates of length m
    n_templates = n - m + 1
    templates = np.array([data[i : i + m] for i in range(n_templates)])

    # Count matches for each template
    counts = np.zeros(n_templates)

    for i in range(n_templates):
        # Calculate max absolute difference between template i and all others
        diffs = np.abs(templates - templates[i]).max(axis=1)
        # Count templates within tolerance
        counts[i] = np.sum(diffs <= r)

    # Average log of match proportions
    # Use counts/n_templates for proportion
    proportions = counts / n_templates

    # Avoid log(0)
    proportions = np.clip(proportions, 1e-10, 1.0)

    return np.mean(np.log(proportions))


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
    n = len(returns)
    apen = np.full(n, np.nan)

    # Minimum data points needed for meaningful ApEn
    min_data = max(m + 2, 10)

    for i in range(window - 1, n):
        window_data = returns[i - window + 1 : i + 1]

        # Remove NaNs
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) < min_data:
            continue

        apen[i] = _approximate_entropy(valid_data, m, r_fraction)

    return apen


def add_approximate_entropy(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    windows: Optional[list[int]] = None,
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

        feature_metadata[col_name] = (
            f"Approximate entropy (m={m}, r={r}, {window}-period, lagged)"
        )

    return df


def add_entropy_features(
    df: pd.DataFrame,
    feature_metadata: Optional[dict[str, str]] = None,
    price_col: str = "close",
    include_shannon: bool = True,
    include_lempel_ziv: bool = True,
    include_approximate: bool = True,
    shannon_windows: Optional[list[int]] = None,
    shannon_bins: int = DEFAULT_SHANNON_BINS,
    lz_windows: Optional[list[int]] = None,
    apen_windows: Optional[list[int]] = None,
    apen_m: int = DEFAULT_APEN_M,
    apen_r: float = DEFAULT_APEN_R,
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
    # Default parameters
    "DEFAULT_SHANNON_WINDOWS",
    "DEFAULT_SHANNON_BINS",
    "DEFAULT_LZ_WINDOWS",
    "DEFAULT_APEN_WINDOWS",
    "DEFAULT_APEN_M",
    "DEFAULT_APEN_R",
]
