"""
RegimeDetector - Unified regime detection for regime-aware training.

PHASE_3: Training Orchestration

This module provides a comprehensive RegimeDetector class that detects
volatility and trend regimes using multiple methods:
- volatility_percentile: Rolling volatility percentiles
- trend_adx: ADX-based trend detection
- combined: Composite regimes (volatility x trend)

The detector is used by RegimeAwareTrainer to split data by regime
and train separate models per regime.

Example:
    from src.training.regime_detector import RegimeDetector
    from src.core import PipelineConfig

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./experiments",
        training_mode="regime_aware",
        regime_detection_method="combined",
        n_regimes=3,
    )

    detector = RegimeDetector.from_config(config)
    regimes = detector.detect(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core import PipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================


class RegimeDetectionMethod(StrEnum):
    """Supported regime detection methods."""

    VOLATILITY_PERCENTILE = "volatility_percentile"
    TREND_ADX = "trend_adx"
    COMBINED = "combined"


class VolatilityRegime(int, Enum):
    """Volatility regime states."""

    LOW = 0
    MEDIUM = 1  # Only used when n_regimes=3
    HIGH = 2


class TrendRegime(int, Enum):
    """Trend regime states."""

    DOWNTREND = 0
    SIDEWAYS = 1
    UPTREND = 2


# Regime labels for display
VOLATILITY_LABELS = {0: "low_vol", 1: "medium_vol", 2: "high_vol"}
TREND_LABELS = {0: "downtrend", 1: "sideways", 2: "uptrend"}


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class RegimeDetectorConfig:
    """
    Configuration for RegimeDetector.

    Attributes:
        method: Detection method (volatility_percentile, trend_adx, combined)
        lookback: Lookback period for regime detection
        n_regimes: Number of regime states (2 or 3)
        volatility_window: Rolling window for volatility calculation
        adx_threshold: ADX threshold for trending classification
        adx_period: Period for ADX calculation
        percentile_low: Percentile threshold for low regime (default 33.3)
        percentile_high: Percentile threshold for high regime (default 66.7)
    """

    method: str = "volatility_percentile"
    lookback: int = 60
    n_regimes: int = 3
    volatility_window: int = 20
    adx_threshold: float = 25.0
    adx_period: int = 14
    percentile_low: float = 33.33
    percentile_high: float = 66.67

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_methods = [m.value for m in RegimeDetectionMethod]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {valid_methods}")

        if self.n_regimes not in [2, 3]:
            raise ValueError(f"n_regimes must be 2 or 3, got {self.n_regimes}")

        if self.lookback < 10:
            raise ValueError(f"lookback must be >= 10, got {self.lookback}")

    @classmethod
    def from_pipeline_config(cls, config: PipelineConfig) -> RegimeDetectorConfig:
        """Create from PipelineConfig."""
        return cls(
            method=config.regime_detection_method,
            lookback=config.regime_lookback,
            n_regimes=config.n_regimes,
            volatility_window=config.regime_volatility_window,
            adx_threshold=config.regime_adx_threshold,
        )


@dataclass
class RegimeResult:
    """
    Result from regime detection.

    Attributes:
        regimes: Series with regime labels for each sample
        regime_counts: Dict mapping regime label to count
        regime_fractions: Dict mapping regime label to fraction
        method: Detection method used
        n_regimes: Number of unique regimes detected
    """

    regimes: pd.Series
    regime_counts: dict[str, int] = field(default_factory=dict)
    regime_fractions: dict[str, float] = field(default_factory=dict)
    method: str = ""
    n_regimes: int = 0

    def __post_init__(self) -> None:
        """Compute counts and fractions if not provided."""
        if not self.regime_counts:
            self.regime_counts = self.regimes.value_counts().to_dict()
        if not self.regime_fractions:
            total = len(self.regimes)
            self.regime_fractions = {k: v / total for k, v in self.regime_counts.items()}
        if self.n_regimes == 0:
            self.n_regimes = len(self.regime_counts)

    def get_mask(self, regime_label: str) -> pd.Series:
        """Get boolean mask for a specific regime."""
        return self.regimes == regime_label

    def get_regime_labels(self) -> list[str]:
        """Get list of unique regime labels."""
        return list(self.regime_counts.keys())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _compute_returns(close: pd.Series) -> pd.Series:
    """Compute log returns."""
    return np.log(close / close.shift(1))


def _compute_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Compute rolling volatility (standard deviation of returns)."""
    return returns.rolling(window=window, min_periods=window).std()


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX).

    ADX measures trend strength regardless of direction.
    Values > 25 typically indicate a strong trend.
    """
    # True Range
    tr = _compute_atr(high, low, close, period=1)  # Raw TR

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Smoothed TR, +DM, -DM
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return adx


def _compute_trend_direction(close: pd.Series, lookback: int) -> pd.Series:
    """
    Compute trend direction using price momentum.

    Returns:
        1 for uptrend, -1 for downtrend, 0 for sideways
    """
    # Use simple momentum: compare to lookback period
    momentum = close - close.shift(lookback)

    # Threshold for sideways (use ATR or volatility-based threshold)
    returns = _compute_returns(close)
    vol = _compute_volatility(returns, lookback)
    threshold = vol * close * 0.5  # Adjust multiplier as needed

    direction = pd.Series(0, index=close.index)  # Default sideways
    direction[momentum > threshold] = 1  # Uptrend
    direction[momentum < -threshold] = -1  # Downtrend

    return direction


# =============================================================================
# REGIME DETECTOR
# =============================================================================


class RegimeDetector:
    """
    Unified regime detector for market regime classification.

    Supports multiple detection methods:
    - volatility_percentile: Classifies regimes by rolling volatility percentiles
    - trend_adx: Classifies regimes by ADX (trend strength)
    - combined: Creates composite regimes (volatility x trend)

    Usage:
        # From PipelineConfig
        from src.core import PipelineConfig
        config = PipelineConfig(...)
        detector = RegimeDetector.from_config(config)

        # Or directly
        detector = RegimeDetector(
            method="combined",
            n_regimes=3,
            lookback=60,
        )

        # Detect regimes
        result = detector.detect(df)
        print(result.regime_counts)

        # Get data for a specific regime
        mask = result.get_mask("high_vol_uptrend")
        df_regime = df[mask]
    """

    def __init__(
        self,
        method: str = "volatility_percentile",
        lookback: int = 60,
        n_regimes: int = 3,
        volatility_window: int = 20,
        adx_threshold: float = 25.0,
        adx_period: int = 14,
        percentile_low: float = 33.33,
        percentile_high: float = 66.67,
    ) -> None:
        """
        Initialize RegimeDetector.

        Args:
            method: Detection method (volatility_percentile, trend_adx, combined)
            lookback: Lookback period for calculations
            n_regimes: Number of regime states (2 or 3)
            volatility_window: Rolling window for volatility
            adx_threshold: ADX threshold for trending
            adx_period: Period for ADX calculation
            percentile_low: Low percentile threshold (for 3 regimes)
            percentile_high: High percentile threshold (for 3 regimes)
        """
        self.config = RegimeDetectorConfig(
            method=method,
            lookback=lookback,
            n_regimes=n_regimes,
            volatility_window=volatility_window,
            adx_threshold=adx_threshold,
            adx_period=adx_period,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )

        logger.debug(
            f"Initialized RegimeDetector: method={method}, "
            f"n_regimes={n_regimes}, lookback={lookback}"
        )

    @classmethod
    def from_config(cls, config: PipelineConfig) -> RegimeDetector:
        """
        Create RegimeDetector from PipelineConfig.

        Args:
            config: PipelineConfig instance

        Returns:
            Configured RegimeDetector
        """
        return cls(
            method=config.regime_detection_method,
            lookback=config.regime_lookback,
            n_regimes=config.n_regimes,
            volatility_window=config.regime_volatility_window,
            adx_threshold=config.regime_adx_threshold,
        )

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """
        Detect regimes in the given DataFrame.

        Args:
            df: DataFrame with OHLCV columns (close required, high/low for ADX)

        Returns:
            RegimeResult with regime labels and statistics
        """
        method = RegimeDetectionMethod(self.config.method)

        if method == RegimeDetectionMethod.VOLATILITY_PERCENTILE:
            regimes = self._detect_volatility_percentile(df)
        elif method == RegimeDetectionMethod.TREND_ADX:
            regimes = self._detect_trend_adx(df)
        elif method == RegimeDetectionMethod.COMBINED:
            regimes = self._detect_combined(df)
        else:
            raise ValueError(f"Unknown method: {method}")

        return RegimeResult(
            regimes=regimes,
            method=self.config.method,
        )

    def _detect_volatility_percentile(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regimes using rolling percentiles.

        Classifies each sample based on where its volatility falls
        in the rolling percentile distribution.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with regime labels (low_vol, medium_vol, high_vol)
        """
        close = self._get_close(df)
        returns = _compute_returns(close)
        volatility = _compute_volatility(returns, self.config.volatility_window)

        # Compute rolling percentile rank
        lookback = self.config.lookback

        def percentile_rank(x: pd.Series) -> float:
            """Compute percentile rank of last value in series."""
            if len(x) < 2:
                return 50.0
            last_val = x.iloc[-1]
            return float(100 * (x < last_val).sum() / len(x))

        pct_rank = volatility.rolling(lookback, min_periods=lookback // 2).apply(
            percentile_rank, raw=False
        )

        # Classify based on percentile thresholds
        if self.config.n_regimes == 2:
            regimes = pd.Series("low_vol", index=df.index)
            regimes[pct_rank >= 50] = "high_vol"
        else:  # n_regimes == 3
            regimes = pd.Series("medium_vol", index=df.index)
            regimes[pct_rank < self.config.percentile_low] = "low_vol"
            regimes[pct_rank >= self.config.percentile_high] = "high_vol"

        return regimes

    def _detect_trend_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect trend regimes using ADX.

        Uses ADX for trend strength and price momentum for direction.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Series with regime labels (downtrend, sideways, uptrend)
        """
        high, low, close = self._get_ohlc(df)

        # Compute ADX
        adx = _compute_adx(high, low, close, self.config.adx_period)

        # Compute trend direction
        direction = _compute_trend_direction(close, self.config.lookback)

        # Classify based on ADX threshold and direction
        regimes = pd.Series("sideways", index=df.index)

        trending = adx >= self.config.adx_threshold

        if self.config.n_regimes == 2:
            # Simple trending vs non-trending
            regimes[trending] = "trending"
            regimes[~trending] = "ranging"
        else:  # n_regimes == 3
            # Uptrend, downtrend, sideways
            regimes[trending & (direction > 0)] = "uptrend"
            regimes[trending & (direction < 0)] = "downtrend"
            # Sideways remains default

        return regimes

    def _detect_combined(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect combined volatility-trend regimes.

        Creates composite regimes by combining volatility and trend states.
        For n_regimes=2: 4 states (low/high vol x trending/ranging)
        For n_regimes=3: 9 states (low/medium/high vol x down/sideways/up trend)

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Series with composite regime labels
        """
        # Get volatility regime
        vol_regimes = self._detect_volatility_percentile(df)

        # Get trend regime
        trend_regimes = self._detect_trend_adx(df)

        # Combine into composite labels
        regimes = vol_regimes + "_" + trend_regimes

        return regimes

    def _get_close(self, df: pd.DataFrame) -> pd.Series:
        """Get close price from DataFrame."""
        for col in ["close", "Close", "CLOSE"]:
            if col in df.columns:
                return df[col]
        raise ValueError("No close price column found in DataFrame")

    def _get_ohlc(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Get high, low, close from DataFrame."""
        high = low = close = None

        for col in ["high", "High", "HIGH"]:
            if col in df.columns:
                high = df[col]
                break

        for col in ["low", "Low", "LOW"]:
            if col in df.columns:
                low = df[col]
                break

        for col in ["close", "Close", "CLOSE"]:
            if col in df.columns:
                close = df[col]
                break

        if high is None or low is None or close is None:
            raise ValueError("DataFrame must have high, low, and close columns")

        return high, low, close

    def get_regime_masks(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Get boolean masks for each regime.

        Args:
            df: DataFrame to detect regimes in

        Returns:
            Dict mapping regime label to boolean mask
        """
        result = self.detect(df)
        return {label: result.get_mask(label) for label in result.get_regime_labels()}

    def split_by_regime(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Split DataFrame by regime.

        Args:
            df: DataFrame to split

        Returns:
            Dict mapping regime label to DataFrame subset
        """
        masks = self.get_regime_masks(df)
        return {label: df[mask].copy() for label, mask in masks.items()}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def detect_regimes(
    df: pd.DataFrame,
    method: str = "volatility_percentile",
    n_regimes: int = 3,
    lookback: int = 60,
    **kwargs: Any,
) -> RegimeResult:
    """
    Convenience function to detect regimes.

    Args:
        df: DataFrame with OHLCV data
        method: Detection method
        n_regimes: Number of regime states
        lookback: Lookback period
        **kwargs: Additional arguments passed to RegimeDetector

    Returns:
        RegimeResult with regime labels
    """
    detector = RegimeDetector(
        method=method,
        n_regimes=n_regimes,
        lookback=lookback,
        **kwargs,
    )
    return detector.detect(df)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RegimeDetector",
    "RegimeDetectorConfig",
    "RegimeResult",
    "RegimeDetectionMethod",
    "VolatilityRegime",
    "TrendRegime",
    "detect_regimes",
]
