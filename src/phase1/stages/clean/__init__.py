"""
Stage 2: Data Cleaning Module

Production-ready data cleaning with gap detection, outlier removal, and quality checks.

This module handles:
- OHLC validation and correction
- Gap detection and quantification
- Gap filling strategies (forward fill, interpolation)
- Duplicate timestamp detection and removal
- Outlier detection (z-score, IQR, ATR methods)
- Spike removal
- Contract roll/stitch handling for futures
- Multi-Timeframe (MTF) resampling (configurable: 5min, 15min, 30min, etc.)
- Complete cleaning pipeline wrappers
- Comprehensive quality reporting

Usage:
    from stages.clean import DataCleaner, clean_symbol_data

    # Simple usage - clean a single file with default 5-minute resampling
    cleaned_df = clean_symbol_data(
        Path('data/raw/MES.parquet'),
        Path('data/clean/MES.parquet'),
        'MES'
    )

    # Custom timeframe (15-minute bars)
    cleaned_df = clean_symbol_data(
        Path('data/raw/MES.parquet'),
        Path('data/clean/MES_15min.parquet'),
        'MES',
        target_timeframe='15min'
    )

    # Multi-timeframe processing (creates 5min, 15min, 30min outputs)
    results = clean_symbol_data_multi_timeframe(
        Path('data/raw/MES.parquet'),
        Path('data/clean/'),
        'MES',
        timeframes=['5min', '15min', '30min']
    )

    # Or use DataCleaner class for batch processing with configurable timeframe
    cleaner = DataCleaner(
        input_dir='data/raw',
        output_dir='data/clean',
        timeframe='1min',
        target_timeframe='15min',  # MTF: resample to 15-minute bars
        gap_fill_method='forward',
        outlier_method='atr'
    )
    results = cleaner.clean_directory(pattern='*.parquet')

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-22 - Added MTF (Multi-Timeframe) support
"""

# Utilities
from .utils import (
    calculate_atr_numba,
    validate_ohlc,
    detect_gaps_simple,
    fill_gaps_simple,
    resample_ohlcv,
    resample_to_5min,  # Backward compatibility
    get_resampling_info,
    add_roll_flags,
    add_session_id,
    DEFAULT_ROLL_GAP_THRESHOLD,
    DEFAULT_ROLL_WINDOW_BARS,
    SESSION_ID_OUTSIDE,
)

# Gap handler
from .gap_handler import GapHandler, create_gap_handler

# Cleaner class
from .cleaner import DataCleaner

# Pipeline functions
from .pipeline import clean_symbol_data, clean_symbol_data_multi_timeframe

__all__ = [
    # Utilities
    'calculate_atr_numba',
    'validate_ohlc',
    'detect_gaps_simple',
    'fill_gaps_simple',
    'resample_ohlcv',
    'resample_to_5min',
    'get_resampling_info',
    'add_roll_flags',
    'add_session_id',
    'DEFAULT_ROLL_GAP_THRESHOLD',
    'DEFAULT_ROLL_WINDOW_BARS',
    'SESSION_ID_OUTSIDE',
    # Gap handler
    'GapHandler',
    'create_gap_handler',
    # Cleaner
    'DataCleaner',
    # Pipeline
    'clean_symbol_data',
    'clean_symbol_data_multi_timeframe',
]
