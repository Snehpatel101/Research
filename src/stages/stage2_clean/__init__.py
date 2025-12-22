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
- Resampling to 5-minute bars
- Complete cleaning pipeline wrappers
- Comprehensive quality reporting

Usage:
    from stages.stage2_clean import DataCleaner, clean_symbol_data

    # Simple usage - clean a single file
    cleaned_df = clean_symbol_data(
        Path('data/raw/MES.parquet'),
        Path('data/clean/MES.parquet'),
        'MES'
    )

    # Or use DataCleaner class for batch processing
    cleaner = DataCleaner(
        input_dir='data/raw',
        output_dir='data/clean',
        timeframe='1min',
        gap_fill_method='forward',
        outlier_method='atr'
    )
    results = cleaner.clean_directory(pattern='*.parquet')

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Refactored into modular package
"""

# Utilities
from .utils import (
    calculate_atr_numba,
    validate_ohlc,
    detect_gaps_simple,
    fill_gaps_simple,
    resample_to_5min,
)

# Cleaner class
from .cleaner import DataCleaner

# Pipeline functions
from .pipeline import clean_symbol_data

__all__ = [
    # Utilities
    'calculate_atr_numba',
    'validate_ohlc',
    'detect_gaps_simple',
    'fill_gaps_simple',
    'resample_to_5min',
    # Cleaner
    'DataCleaner',
    # Pipeline
    'clean_symbol_data',
]
