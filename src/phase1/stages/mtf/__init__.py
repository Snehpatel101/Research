"""
Multi-Timeframe (MTF) Feature Integration Package.

This package provides tools for computing features from higher timeframes
and aligning them to the base timeframe without lookahead bias.

Supported Timeframes:
    - 5min (base)
    - 15min, 30min
    - 1h (60min)
    - 4h (240min)
    - daily (1440min)

MTF Modes:
    - BARS: Generate only OHLCV data at higher timeframes
    - INDICATORS: Generate only technical indicators at higher timeframes
    - BOTH: Generate both bars and indicators (default)

Main Components:
    MTFFeatureGenerator: Main class for generating MTF features
    MTFMode: Enum for selecting what to generate (bars, indicators, or both)
    add_mtf_features: Convenience function for adding MTF features
    validate_mtf_alignment: Validate proper alignment without lookahead

Example:
    >>> from stages.mtf import MTFFeatureGenerator, MTFMode
    >>>
    >>> # Generate both bars and indicators (default)
    >>> generator = MTFFeatureGenerator(base_timeframe='5min')
    >>> df_with_mtf = generator.generate_mtf_features(df)
    >>>
    >>> # Generate only bars
    >>> generator = MTFFeatureGenerator(mode=MTFMode.BARS)
    >>> df_bars = generator.generate_mtf_features(df)
    >>>
    >>> # Generate only indicators
    >>> generator = MTFFeatureGenerator(mode='indicators')
    >>> df_indicators = generator.generate_mtf_features(df)
    >>>
    >>> # Custom timeframes with 4h and daily
    >>> generator = MTFFeatureGenerator(
    ...     mtf_timeframes=['1h', '4h', 'daily'],
    ...     mode=MTFMode.BOTH
    ... )
    >>> df_full = generator.generate_mtf_features(df)
"""

from .constants import (
    DEFAULT_BASE_TIMEFRAME,
    DEFAULT_MTF_MODE,
    DEFAULT_MTF_TIMEFRAMES,
    MTF_TIMEFRAMES,
    PANDAS_FREQ_MAP,
    REQUIRED_OHLCV_COLS,
    MTFMode,
)
from .convenience import (
    add_mtf_bars,
    add_mtf_features,
    add_mtf_indicators,
    validate_mtf_alignment,
)
from .generator import MTFFeatureGenerator
from .validators import validate_ohlcv_dataframe, validate_timeframe_format

__all__ = [
    # Main class
    'MTFFeatureGenerator',
    # MTF Mode enum
    'MTFMode',
    # Convenience functions
    'add_mtf_features',
    'add_mtf_bars',
    'add_mtf_indicators',
    'validate_mtf_alignment',
    # Validation functions
    'validate_ohlcv_dataframe',
    'validate_timeframe_format',
    # Constants
    'MTF_TIMEFRAMES',
    'REQUIRED_OHLCV_COLS',
    'DEFAULT_MTF_TIMEFRAMES',
    'DEFAULT_MTF_MODE',
    'DEFAULT_BASE_TIMEFRAME',
    'PANDAS_FREQ_MAP',
]
