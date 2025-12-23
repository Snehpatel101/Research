"""
Multi-Timeframe (MTF) Feature Integration.

This module re-exports everything from the mtf package for backward compatibility.

The functionality has been refactored into:
- mtf/constants.py: Constants and configuration
- mtf/validators.py: Validation functions
- mtf/generator.py: MTFFeatureGenerator class
- mtf/convenience.py: Convenience functions

Example:
    >>> from stages.mtf_features import MTFFeatureGenerator, add_mtf_features
    >>> generator = MTFFeatureGenerator(base_timeframe='5min')
    >>> df_with_mtf = generator.generate_mtf_features(df)
"""

# Re-export everything from the mtf package
from src.stages.mtf import (
    # Main class
    MTFFeatureGenerator,
    # Convenience functions
    add_mtf_features,
    validate_mtf_alignment,
    # Validation functions
    validate_ohlcv_dataframe,
    validate_timeframe_format,
    # Constants
    MTF_TIMEFRAMES,
    REQUIRED_OHLCV_COLS,
)

__all__ = [
    # Main class
    'MTFFeatureGenerator',
    # Convenience functions
    'add_mtf_features',
    'validate_mtf_alignment',
    # Validation functions
    'validate_ohlcv_dataframe',
    'validate_timeframe_format',
    # Constants
    'MTF_TIMEFRAMES',
    'REQUIRED_OHLCV_COLS',
]
