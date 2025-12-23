"""
Multi-Timeframe (MTF) Feature Integration Package.

This package provides tools for computing features from higher timeframes
and aligning them to the base timeframe without lookahead bias.

Main Components:
    MTFFeatureGenerator: Main class for generating MTF features
    add_mtf_features: Convenience function for adding MTF features
    validate_mtf_alignment: Validate proper alignment without lookahead

Example:
    >>> from stages.mtf import MTFFeatureGenerator, add_mtf_features
    >>> generator = MTFFeatureGenerator(base_timeframe='5min')
    >>> df_with_mtf = generator.generate_mtf_features(df)
"""

from .generator import MTFFeatureGenerator
from .convenience import add_mtf_features, validate_mtf_alignment
from .constants import MTF_TIMEFRAMES, REQUIRED_OHLCV_COLS
from .validators import validate_ohlcv_dataframe, validate_timeframe_format

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
