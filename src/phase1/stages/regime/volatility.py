"""
Volatility regime detection based on ATR percentile.

This module classifies market volatility into low, normal, and high
regimes using rolling percentile analysis of Average True Range (ATR).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import (
    RegimeDetector,
    RegimeType,
    VolatilityRegimeLabel,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Calculate Average True Range.

    Pure function for ATR calculation without external dependencies.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR values with NaN for warmup period
    """
    n = len(high)
    tr = np.zeros(n)
    atr = np.full(n, np.nan)

    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Calculate ATR using EMA-style smoothing
    if period < n:
        atr[period] = np.mean(tr[1:period + 1])

        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


class VolatilityRegimeDetector(RegimeDetector):
    """
    Detect volatility regime based on ATR percentile analysis.

    Classification logic:
    - LOW: ATR < low_percentile (default 25th)
    - NORMAL: low_percentile <= ATR <= high_percentile (default 25th-75th)
    - HIGH: ATR > high_percentile (default 75th)

    The lookback period defines the rolling window for percentile calculation.

    Attributes:
        atr_period: Period for ATR calculation (default 14)
        lookback: Rolling window for percentile calculation (default 100)
        low_percentile: Threshold for low volatility (default 25)
        high_percentile: Threshold for high volatility (default 75)

    Example:
        >>> detector = VolatilityRegimeDetector(atr_period=14, lookback=100)
        >>> regimes = detector.detect(df)
        >>> print(regimes.value_counts())
        normal    450
        low       275
        high      275
    """

    def __init__(
        self,
        atr_period: int = 14,
        lookback: int = 100,
        low_percentile: float = 25.0,
        high_percentile: float = 75.0,
        atr_column: Optional[str] = None
    ):
        """
        Initialize volatility regime detector.

        Args:
            atr_period: Period for ATR calculation
            lookback: Rolling window for percentile calculation
            low_percentile: Percentile threshold for low volatility
            high_percentile: Percentile threshold for high volatility
            atr_column: Pre-computed ATR column name (optional)

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(RegimeType.VOLATILITY)

        # Validate parameters at boundary
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if not (0 < low_percentile < 100):
            raise ValueError(f"low_percentile must be in (0, 100), got {low_percentile}")
        if not (0 < high_percentile < 100):
            raise ValueError(f"high_percentile must be in (0, 100), got {high_percentile}")
        if low_percentile >= high_percentile:
            raise ValueError(
                f"low_percentile ({low_percentile}) must be < high_percentile ({high_percentile})"
            )

        self.atr_period = atr_period
        self.lookback = lookback
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.atr_column = atr_column

    def get_required_columns(self) -> list[str]:
        """Get required columns for detection.

        Note: Returns OHLC columns since we can always calculate ATR from them.
        If atr_column is specified and present, we'll use it; otherwise compute.
        """
        # Always require OHLC for fallback calculation
        return ['high', 'low', 'close']

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regime for each bar.

        Args:
            df: DataFrame with OHLC data or pre-computed ATR

        Returns:
            Series with regime labels: 'low', 'normal', 'high'

        Raises:
            ValueError: If required columns missing or insufficient data
        """
        self.validate_input(df)

        # Get ATR values - prefer pre-computed if available
        if self.atr_column and self.atr_column in df.columns:
            atr = df[self.atr_column].values
            logger.debug(f"Using pre-computed ATR from column '{self.atr_column}'")
        else:
            if self.atr_column:
                logger.debug(
                    f"ATR column '{self.atr_column}' not found, computing from OHLC"
                )
            atr = calculate_atr(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                self.atr_period
            )

        # Calculate rolling percentiles
        atr_series = pd.Series(atr, index=df.index)
        low_threshold = atr_series.rolling(window=self.lookback, min_periods=1).quantile(
            self.low_percentile / 100.0
        )
        high_threshold = atr_series.rolling(window=self.lookback, min_periods=1).quantile(
            self.high_percentile / 100.0
        )

        # Classify regime
        regimes = pd.Series(
            VolatilityRegimeLabel.NORMAL.value,
            index=df.index,
            dtype='object'
        )

        regimes[atr_series < low_threshold] = VolatilityRegimeLabel.LOW.value
        regimes[atr_series > high_threshold] = VolatilityRegimeLabel.HIGH.value

        # Handle NaN in ATR (warmup period)
        regimes[atr_series.isna()] = np.nan

        logger.debug(
            f"Volatility regime distribution: {regimes.value_counts(dropna=False).to_dict()}"
        )

        return regimes

    def detect_with_thresholds(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Detect regime and return threshold values for debugging.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (regimes, low_threshold, high_threshold)
        """
        self.validate_input(df)

        # Get ATR values
        if self.atr_column and self.atr_column in df.columns:
            atr = df[self.atr_column].values
        else:
            atr = calculate_atr(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                self.atr_period
            )

        atr_series = pd.Series(atr, index=df.index)
        low_threshold = atr_series.rolling(window=self.lookback, min_periods=1).quantile(
            self.low_percentile / 100.0
        )
        high_threshold = atr_series.rolling(window=self.lookback, min_periods=1).quantile(
            self.high_percentile / 100.0
        )

        regimes = self.detect(df)

        return regimes, low_threshold, high_threshold


__all__ = [
    'VolatilityRegimeDetector',
    'calculate_atr',
]
