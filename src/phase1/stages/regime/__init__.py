"""
Regime Detection System for Market State Classification.

This package provides a comprehensive regime detection framework for
classifying market conditions into discrete states. These regime labels
can be used as:

1. **Features**: Add regime columns to the feature DataFrame for model training
2. **Filters**: Use regime to select which model to apply (model per regime)
3. **Adaptive Parameters**: Adjust triple-barrier parameters per regime

Components:
    base: Abstract base classes and enums for regime detection
    volatility: Volatility regime detection (ATR percentile)
    trend: Trend regime detection (ADX + SMA alignment)
    structure: Market structure detection (Hurst exponent)
    composite: Combined regime detection and feature generation

Example:
    >>> from stages.regime import CompositeRegimeDetector
    >>>
    >>> # Create detector with defaults
    >>> detector = CompositeRegimeDetector.with_defaults()
    >>>
    >>> # Add regime columns to DataFrame
    >>> df_with_regimes = detector.add_regime_columns(df)
    >>>
    >>> # Or use the convenience function
    >>> from stages.regime import add_regime_features_to_dataframe
    >>> df_with_regimes = add_regime_features_to_dataframe(df)

Configuration Example:
    >>> from src.phase1.config import REGIME_CONFIG
    >>> detector = CompositeRegimeDetector.from_config(REGIME_CONFIG)

Author: ML Pipeline
Created: 2025-12-22
"""

# Base classes and enums
from .base import (
    RegimeType,
    VolatilityRegimeLabel,
    TrendRegimeLabel,
    StructureRegimeLabel,
    RegimeConfig,
    RegimeDetector,
)

# Individual detectors
from .volatility import (
    VolatilityRegimeDetector,
    calculate_atr,
)

from .trend import (
    TrendRegimeDetector,
    calculate_adx,
    calculate_sma,
)

from .structure import (
    MarketStructureDetector,
    calculate_hurst_exponent,
    calculate_rolling_hurst,
)

# Composite detector
from .composite import (
    CompositeRegimeDetector,
    CompositeRegimeResult,
    add_regime_features_to_dataframe,
)


__all__ = [
    # Base classes
    'RegimeType',
    'VolatilityRegimeLabel',
    'TrendRegimeLabel',
    'StructureRegimeLabel',
    'RegimeConfig',
    'RegimeDetector',
    # Volatility
    'VolatilityRegimeDetector',
    'calculate_atr',
    # Trend
    'TrendRegimeDetector',
    'calculate_adx',
    'calculate_sma',
    # Structure
    'MarketStructureDetector',
    'calculate_hurst_exponent',
    'calculate_rolling_hurst',
    # Composite
    'CompositeRegimeDetector',
    'CompositeRegimeResult',
    'add_regime_features_to_dataframe',
]
