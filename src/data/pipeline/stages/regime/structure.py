"""
Market structure regime detection based on Hurst exponent.

This module classifies market structure into mean-reverting, random,
and trending regimes using the Hurst exponent calculated via R/S analysis.
"""

import logging

import numpy as np
import pandas as pd

from .base import (
    RegimeDetector,
    RegimeType,
    StructureRegimeLabel,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def calculate_hurst_exponent(prices: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent using Rescaled Range (R/S) analysis.

    The Hurst exponent H characterizes the autocorrelation of a time series:
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Trending (persistent)

    Args:
        prices: Price series (log prices work best)
        min_lag: Minimum lag for R/S calculation
        max_lag: Maximum lag for R/S calculation

    Returns:
        Hurst exponent value in range [0, 1], or NaN if insufficient data
    """
    n = len(prices)

    if n < max_lag + 1:
        return np.nan

    # Use log returns for stationarity
    log_prices = np.log(prices)
    returns = np.diff(log_prices)

    if len(returns) < max_lag:
        return np.nan

    # Calculate R/S for different lags
    lags = []
    rs_values = []

    for lag in range(min_lag, min(max_lag + 1, len(returns))):
        # Number of non-overlapping segments
        n_segments = len(returns) // lag

        if n_segments < 1:
            continue

        rs_list = []

        for i in range(n_segments):
            segment = returns[i * lag : (i + 1) * lag]

            if len(segment) < lag:
                continue

            # Mean-adjusted cumulative sum
            mean_ret = np.mean(segment)
            cumsum = np.cumsum(segment - mean_ret)

            # Range
            R = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            S = np.std(segment, ddof=1)

            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))

    if len(lags) < 2:
        return np.nan

    # Linear regression of log(R/S) vs log(lag)
    log_lags = np.log(np.array(lags))
    log_rs = np.log(np.array(rs_values))

    # Simple linear regression
    n_points = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_lags * log_rs)
    sum_x2 = np.sum(log_lags**2)

    denominator = n_points * sum_x2 - sum_x**2
    if denominator == 0:
        return np.nan

    hurst = (n_points * sum_xy - sum_x * sum_y) / denominator

    # Clip to valid range
    return np.clip(hurst, 0.0, 1.0)


def calculate_rolling_hurst(
    prices: np.ndarray, lookback: int, min_lag: int = 2, max_lag: int = 20
) -> np.ndarray:
    """
    Calculate rolling Hurst exponent.

    Args:
        prices: Price series
        lookback: Rolling window size
        min_lag: Minimum lag for R/S calculation
        max_lag: Maximum lag for R/S calculation

    Returns:
        Array of Hurst values with NaN for warmup period
    """
    n = len(prices)
    hurst = np.full(n, np.nan)

    # Need at least lookback points for calculation
    for i in range(lookback - 1, n):
        window = prices[i - lookback + 1 : i + 1]
        hurst[i] = calculate_hurst_exponent(window, min_lag, max_lag)

    return hurst


class MarketStructureDetector(RegimeDetector):
    """
    Detect market structure regime based on Hurst exponent.

    Classification logic:
    - MEAN_REVERTING: H < mean_reverting_threshold (default 0.4)
    - RANDOM: mean_reverting_threshold <= H <= trending_threshold
    - TRENDING: H > trending_threshold (default 0.6)

    The Hurst exponent is calculated using Rescaled Range (R/S) analysis
    over a rolling window.

    Attributes:
        lookback: Rolling window for Hurst calculation (default 100)
        min_lag: Minimum lag for R/S analysis (default 2)
        max_lag: Maximum lag for R/S analysis (default 20)
        mean_reverting_threshold: Upper bound for mean-reverting (default 0.4)
        trending_threshold: Lower bound for trending (default 0.6)

    Example:
        >>> detector = MarketStructureDetector(lookback=100)
        >>> regimes = detector.detect(df)
        >>> print(regimes.value_counts())
        random           450
        mean_reverting   300
        trending         250

    Note:
        The Hurst exponent calculation is computationally intensive.
        Consider using a larger lookback for smoother estimates at
        the cost of less responsiveness.
    """

    def __init__(
        self,
        lookback: int = 100,
        min_lag: int = 2,
        max_lag: int = 20,
        mean_reverting_threshold: float = 0.4,
        trending_threshold: float = 0.6,
    ):
        """
        Initialize market structure detector.

        Args:
            lookback: Rolling window for Hurst calculation
            min_lag: Minimum lag for R/S analysis
            max_lag: Maximum lag for R/S analysis
            mean_reverting_threshold: Upper H bound for mean-reverting
            trending_threshold: Lower H bound for trending

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(RegimeType.STRUCTURE)

        if lookback < 20:
            raise ValueError(f"lookback must be >= 20 for meaningful Hurst, got {lookback}")
        if min_lag < 2:
            raise ValueError(f"min_lag must be >= 2, got {min_lag}")
        if max_lag <= min_lag:
            raise ValueError(f"max_lag ({max_lag}) must be > min_lag ({min_lag})")
        if max_lag >= lookback:
            raise ValueError(f"max_lag ({max_lag}) must be < lookback ({lookback})")
        if not (0 < mean_reverting_threshold < 0.5):
            raise ValueError(
                f"mean_reverting_threshold must be in (0, 0.5), got {mean_reverting_threshold}"
            )
        if not (0.5 < trending_threshold < 1.0):
            raise ValueError(f"trending_threshold must be in (0.5, 1.0), got {trending_threshold}")

        self.lookback = lookback
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.mean_reverting_threshold = mean_reverting_threshold
        self.trending_threshold = trending_threshold

    def get_required_columns(self) -> list[str]:
        """Get required columns for detection."""
        return ["close"]

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market structure regime for each bar.

        Args:
            df: DataFrame with close prices

        Returns:
            Series with regime labels: 'mean_reverting', 'random', 'trending'

        Raises:
            ValueError: If required columns missing or insufficient data
        """
        self.validate_input(df)

        prices = df["close"].values

        # Check for sufficient data
        min_required = self.lookback + self.max_lag
        if len(prices) < min_required:
            logger.warning(
                f"Insufficient data for Hurst calculation: {len(prices)} < {min_required}. "
                f"Returning all NaN."
            )
            return pd.Series(np.nan, index=df.index, dtype="object")

        # Calculate rolling Hurst
        hurst = calculate_rolling_hurst(prices, self.lookback, self.min_lag, self.max_lag)

        hurst_series = pd.Series(hurst, index=df.index)

        # Classify regime
        regimes = pd.Series(StructureRegimeLabel.RANDOM.value, index=df.index, dtype="object")

        regimes[hurst_series < self.mean_reverting_threshold] = (
            StructureRegimeLabel.MEAN_REVERTING.value
        )
        regimes[hurst_series > self.trending_threshold] = StructureRegimeLabel.TRENDING.value

        # Handle NaN from warmup period
        regimes[hurst_series.isna()] = np.nan

        logger.debug(
            f"Structure regime distribution: {regimes.value_counts(dropna=False).to_dict()}"
        )

        return regimes

    def detect_with_hurst(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Detect regime and return Hurst values for debugging.

        Args:
            df: DataFrame with close prices

        Returns:
            Tuple of (regimes, hurst_values)
        """
        self.validate_input(df)

        prices = df["close"].values
        hurst = calculate_rolling_hurst(prices, self.lookback, self.min_lag, self.max_lag)

        regimes = self.detect(df)
        hurst_series = pd.Series(hurst, index=df.index)

        return regimes, hurst_series


__all__ = [
    "MarketStructureDetector",
    "calculate_hurst_exponent",
    "calculate_rolling_hurst",
]
