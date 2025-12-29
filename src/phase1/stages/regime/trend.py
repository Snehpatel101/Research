"""
Trend regime detection based on ADX and SMA alignment.

This module classifies market trend state into uptrend, downtrend,
and sideways regimes using ADX strength and price-to-SMA relationship.
"""

import logging

import numpy as np
import pandas as pd

from .base import (
    RegimeDetector,
    RegimeType,
    TrendRegimeLabel,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ADX, +DI, -DI.

    Pure function for ADX calculation without external dependencies.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period

    Returns:
        Tuple of (ADX, +DI, -DI) arrays
    """
    n = len(close)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    adx = np.full(n, np.nan)

    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff

    # Smooth TR and DM
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    if period < n:
        tr_smooth[period] = np.sum(tr[1:period + 1])
        plus_dm_smooth[period] = np.sum(plus_dm[1:period + 1])
        minus_dm_smooth[period] = np.sum(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]

    # Calculate DI
    for i in range(period, n):
        if tr_smooth[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]

    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX is smoothed DX
    if 2 * period - 1 < n:
        adx[2 * period - 1] = np.nanmean(dx[period:2 * period])
        for i in range(2 * period, n):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


def calculate_sma(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Args:
        arr: Input array
        period: SMA period

    Returns:
        SMA values with NaN for warmup period
    """
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        result[i] = np.mean(arr[i - period + 1:i + 1])

    return result


class TrendRegimeDetector(RegimeDetector):
    """
    Detect trend regime based on ADX strength and price-SMA alignment.

    Classification logic:
    - UPTREND: ADX > threshold AND close > SMA
    - DOWNTREND: ADX > threshold AND close < SMA
    - SIDEWAYS: ADX <= threshold (regardless of price position)

    The ADX threshold determines trend strength requirement.
    Higher threshold = more selective trend identification.

    Attributes:
        adx_period: Period for ADX calculation (default 14)
        sma_period: Period for SMA calculation (default 50)
        adx_threshold: ADX value to consider trending (default 25)

    Example:
        >>> detector = TrendRegimeDetector(adx_threshold=25)
        >>> regimes = detector.detect(df)
        >>> print(regimes.value_counts())
        sideways     450
        uptrend      300
        downtrend    250
    """

    def __init__(
        self,
        adx_period: int = 14,
        sma_period: int = 50,
        adx_threshold: float = 25.0,
        adx_column: str | None = None,
        sma_column: str | None = None
    ):
        """
        Initialize trend regime detector.

        Args:
            adx_period: Period for ADX calculation
            sma_period: Period for SMA calculation
            adx_threshold: Minimum ADX for trending market
            adx_column: Pre-computed ADX column name (optional)
            sma_column: Pre-computed SMA column name (optional)

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(RegimeType.TREND)

        if adx_period < 1:
            raise ValueError(f"adx_period must be >= 1, got {adx_period}")
        if sma_period < 1:
            raise ValueError(f"sma_period must be >= 1, got {sma_period}")
        if adx_threshold < 0 or adx_threshold > 100:
            raise ValueError(f"adx_threshold must be in [0, 100], got {adx_threshold}")

        self.adx_period = adx_period
        self.sma_period = sma_period
        self.adx_threshold = adx_threshold
        self.adx_column = adx_column
        self.sma_column = sma_column

    def get_required_columns(self) -> list[str]:
        """Get required columns for detection."""
        required = ['close']

        # Only require OHLC if no pre-computed ADX
        if not self.adx_column:
            required.extend(['high', 'low'])

        return required

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect trend regime for each bar.

        Args:
            df: DataFrame with OHLC data or pre-computed indicators

        Returns:
            Series with regime labels: 'uptrend', 'downtrend', 'sideways'

        Raises:
            ValueError: If required columns missing or insufficient data
        """
        self.validate_input(df)

        # Get ADX values
        if self.adx_column and self.adx_column in df.columns:
            adx = df[self.adx_column].values
        else:
            adx, _, _ = calculate_adx(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                self.adx_period
            )

        # Get SMA values
        if self.sma_column and self.sma_column in df.columns:
            sma = df[self.sma_column].values
        else:
            sma = calculate_sma(df['close'].values, self.sma_period)

        close = df['close'].values

        # Classify regime
        regimes = pd.Series(
            TrendRegimeLabel.SIDEWAYS.value,
            index=df.index,
            dtype='object'
        )

        adx_series = pd.Series(adx, index=df.index)
        sma_series = pd.Series(sma, index=df.index)
        close_series = pd.Series(close, index=df.index)

        # Strong trend conditions
        is_trending = adx_series > self.adx_threshold
        is_above_sma = close_series > sma_series
        is_below_sma = close_series < sma_series

        regimes[is_trending & is_above_sma] = TrendRegimeLabel.UPTREND.value
        regimes[is_trending & is_below_sma] = TrendRegimeLabel.DOWNTREND.value

        # Handle NaN from warmup periods
        warmup_mask = adx_series.isna() | sma_series.isna()
        regimes[warmup_mask] = np.nan

        logger.debug(
            f"Trend regime distribution: {regimes.value_counts(dropna=False).to_dict()}"
        )

        return regimes

    def detect_with_components(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Detect regime and return component values for debugging.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (regimes, adx_values, sma_values)
        """
        self.validate_input(df)

        # Get ADX values
        if self.adx_column and self.adx_column in df.columns:
            adx = df[self.adx_column].values
        else:
            adx, _, _ = calculate_adx(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                self.adx_period
            )

        # Get SMA values
        if self.sma_column and self.sma_column in df.columns:
            sma = df[self.sma_column].values
        else:
            sma = calculate_sma(df['close'].values, self.sma_period)

        regimes = self.detect(df)
        adx_series = pd.Series(adx, index=df.index)
        sma_series = pd.Series(sma, index=df.index)

        return regimes, adx_series, sma_series


__all__ = [
    'TrendRegimeDetector',
    'calculate_adx',
    'calculate_sma',
]
