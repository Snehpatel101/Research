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
    from src.data.pipeline.stages.clean import clean_symbol_data

    cleaned_df = clean_symbol_data(
        Path('data/raw/MES.parquet'),
        Path('data/clean/MES.parquet'),
        'MES',
        target_timeframe='15min',
    )

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-22 - Added MTF (Multi-Timeframe) support
"""

from src.data.pipeline.stages.features.numba_functions import calculate_atr_numba

from .pipeline import clean_symbol_data, clean_symbol_data_multi_timeframe
from .utils import (
    DEFAULT_ROLL_GAP_THRESHOLD,
    DEFAULT_ROLL_WINDOW_BARS,
    SESSION_ID_OUTSIDE,
    add_roll_flags,
    add_session_id,
    detect_gaps_simple,
    fill_gaps_simple,
    resample_ohlcv,
    validate_ohlc,
)

__all__ = [
    # Utilities
    "calculate_atr_numba",
    "validate_ohlc",
    "detect_gaps_simple",
    "fill_gaps_simple",
    "resample_ohlcv",
    "add_roll_flags",
    "add_session_id",
    "DEFAULT_ROLL_GAP_THRESHOLD",
    "DEFAULT_ROLL_WINDOW_BARS",
    "SESSION_ID_OUTSIDE",
    # Pipeline
    "clean_symbol_data",
    "clean_symbol_data_multi_timeframe",
]
