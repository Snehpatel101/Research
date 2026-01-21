"""
Base classes and enums for regime detection.

This module defines the abstract base class for all regime detectors
and the common enumerations used across the regime detection system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class RegimeType(Enum):
    """Types of market regime classification."""

    VOLATILITY = "volatility"
    TREND = "trend"
    STRUCTURE = "structure"
    SESSION = "session"
    COMPOSITE = "composite"


class VolatilityRegimeLabel(Enum):
    """Volatility regime classification labels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TrendRegimeLabel(Enum):
    """Trend regime classification labels."""

    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


class StructureRegimeLabel(Enum):
    """Market structure regime labels based on Hurst exponent."""

    MEAN_REVERTING = "mean_reverting"
    RANDOM = "random"
    TRENDING = "trending"


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for a regime detector.

    Attributes:
        detector_type: Type of regime detector
        params: Detector-specific parameters
        column_name: Output column name in DataFrame
    """

    detector_type: RegimeType
    params: dict[str, Any]
    column_name: str


class RegimeDetector(ABC):
    """Abstract base class for all regime detectors.

    Regime detectors classify market conditions into discrete states
    based on technical analysis. All implementations must:

    1. Implement the `detect()` method returning a categorical Series
    2. Validate input DataFrame at the boundary
    3. Handle edge cases (insufficient data, NaN values)
    4. Document the classification logic in the docstring

    Example:
        >>> detector = VolatilityRegimeDetector(atr_period=14)
        >>> regimes = detector.detect(df)
        >>> print(regimes.value_counts())
    """

    def __init__(self, regime_type: RegimeType):
        """Initialize base detector.

        Args:
            regime_type: Type of regime this detector classifies
        """
        self._regime_type = regime_type

    @property
    def regime_type(self) -> RegimeType:
        """Get the type of regime this detector classifies."""
        return self._regime_type

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Detect regime labels for each bar in the DataFrame.

        Args:
            df: DataFrame with OHLCV data and any required indicators

        Returns:
            Series with regime labels (categorical) indexed same as input

        Raises:
            ValueError: If required columns are missing or data is insufficient
        """
        pass

    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """Get list of required DataFrame columns for detection.

        Returns:
            List of column names that must be present in input DataFrame
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns and sufficient data.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails with specific error message
        """
        if df is None or len(df) == 0:
            raise ValueError(f"{self.__class__.__name__}: Input DataFrame is empty")

        required = self.get_required_columns()
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def get_regime_summary(self, regimes: pd.Series) -> dict[str, Any]:
        """Generate summary statistics for detected regimes.

        Args:
            regimes: Series of regime labels

        Returns:
            Dict with regime distribution and transition statistics
        """
        if len(regimes) == 0:
            return {"distribution": {}, "transitions": 0, "avg_duration": 0}

        # Distribution
        distribution = regimes.value_counts(normalize=True).to_dict()

        # Count transitions (regime changes)
        transitions = (regimes != regimes.shift(1)).sum() - 1
        transitions = max(0, transitions)  # Handle single-element case

        # Average duration in each regime
        avg_duration = len(regimes) / (transitions + 1) if transitions >= 0 else len(regimes)

        return {
            "distribution": distribution,
            "transitions": int(transitions),
            "avg_duration": float(avg_duration),
        }


__all__ = [
    "RegimeType",
    "VolatilityRegimeLabel",
    "TrendRegimeLabel",
    "StructureRegimeLabel",
    "RegimeConfig",
    "RegimeDetector",
]
