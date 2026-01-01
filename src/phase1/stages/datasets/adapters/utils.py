"""
Utility functions for Multi-Resolution adapters.

Provides helper functions for:
- Timeframe suffix conversion
- Column extraction by timeframe
- Feature map building
- Sequence index generation
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES


# =============================================================================
# CONSTANTS
# =============================================================================

# Default features to extract per timeframe (without suffix)
DEFAULT_MTF_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'sma_50', 'ema_9', 'ema_21',
    'rsi_14', 'atr_14', 'bb_position', 'macd_hist',
    'close_sma20_ratio',
]

# Timeframe suffix patterns for column matching
TIMEFRAME_SUFFIX_PATTERNS = {
    '1min': r'_1m$',
    '5min': r'_5m$',
    '10min': r'_10m$',
    '15min': r'_15m$',
    '20min': r'_20m$',
    '25min': r'_25m$',
    '30min': r'_30m$',
    '45min': r'_45m$',
    '60min': r'_1h$',
    '1h': r'_1h$',
    '4h': r'_4h$',
    'daily': r'_1d$',
    '1d': r'_1d$',
}


# =============================================================================
# TIMEFRAME UTILITIES
# =============================================================================

def get_timeframe_suffix(tf: str) -> str:
    """
    Convert timeframe string to column suffix.

    Examples:
        '15min' -> '_15m'
        '1h' -> '_1h'
        '4h' -> '_4h'
        'daily' -> '_1d'
    """
    minutes = MTF_TIMEFRAMES.get(tf)
    if minutes is None:
        raise ValueError(f"Unknown timeframe: {tf}")

    if minutes >= 1440:  # Daily or longer
        days = minutes // 1440
        return f"_{days}d"
    elif minutes >= 60 and minutes % 60 == 0:
        hours = minutes // 60
        return f"_{hours}h"
    return f"_{minutes}m"


def extract_timeframe_columns(
    df: pd.DataFrame,
    timeframe: str,
    feature_bases: list[str] | None = None
) -> list[str]:
    """
    Extract columns belonging to a specific timeframe.

    Args:
        df: DataFrame with MTF columns
        timeframe: Timeframe string (e.g., '15min', '1h')
        feature_bases: Base feature names to look for (e.g., ['close', 'rsi_14'])
                       If None, extracts all columns with matching suffix

    Returns:
        List of column names for the timeframe, sorted consistently
    """
    suffix = get_timeframe_suffix(timeframe)
    pattern = TIMEFRAME_SUFFIX_PATTERNS.get(timeframe, re.escape(suffix) + '$')

    if feature_bases is not None:
        columns = []
        for base in feature_bases:
            col_name = f"{base}{suffix}"
            if col_name in df.columns:
                columns.append(col_name)
        return columns
    else:
        regex = re.compile(pattern)
        columns = [col for col in df.columns if regex.search(col)]
        return sorted(columns)


def build_timeframe_feature_map(
    df: pd.DataFrame,
    timeframes: list[str],
    feature_bases: list[str] | None = None,
    include_base_features: bool = True
) -> dict[str, list[str]]:
    """
    Build a mapping of timeframes to their feature columns.

    Args:
        df: DataFrame with MTF columns
        timeframes: List of timeframes to extract
        feature_bases: Base feature names (without suffix)
        include_base_features: Include non-MTF features for base timeframe

    Returns:
        Dict mapping timeframe -> list of column names
    """
    feature_map: dict[str, list[str]] = {}

    for tf in timeframes:
        columns = extract_timeframe_columns(df, tf, feature_bases)
        feature_map[tf] = columns

    # For base timeframe, optionally include non-suffixed features
    if include_base_features and timeframes:
        base_tf = timeframes[0]
        all_suffixes = [get_timeframe_suffix(tf) for tf in MTF_TIMEFRAMES.keys()]
        base_cols = []
        for col in df.columns:
            if col in ['datetime', 'symbol', 'date', 'time']:
                continue
            if col.startswith('label_') or col.startswith('sample_weight_'):
                continue
            has_suffix = any(col.endswith(suffix) for suffix in all_suffixes)
            if not has_suffix and col not in feature_map.get(base_tf, []):
                base_cols.append(col)

        if base_cols and base_tf in feature_map:
            feature_map[base_tf] = sorted(base_cols) + feature_map[base_tf]

    return feature_map


# =============================================================================
# SEQUENCE INDEX UTILITIES
# =============================================================================

def build_4d_sequence_indices(
    n_samples: int,
    seq_len: int,
    stride: int,
    symbol_boundaries: list[int] | None = None
) -> np.ndarray:
    """
    Build valid sequence start indices for 4D tensor generation.

    Args:
        n_samples: Total number of samples (rows in base timeframe)
        seq_len: Sequence length
        stride: Step size between sequences
        symbol_boundaries: Indices where symbol changes

    Returns:
        Array of valid sequence start indices
    """
    if n_samples < seq_len:
        return np.array([], dtype=np.int64)

    if symbol_boundaries is None or len(symbol_boundaries) == 0:
        max_start = n_samples - seq_len
        return np.arange(0, max_start + 1, stride)

    boundaries = [0] + sorted(symbol_boundaries) + [n_samples]
    valid_indices = []

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg_len = seg_end - seg_start

        if seg_len < seq_len:
            continue

        max_start = seg_end - seq_len
        seg_indices = np.arange(seg_start, max_start + 1, stride)
        valid_indices.append(seg_indices)

    if not valid_indices:
        return np.array([], dtype=np.int64)

    return np.concatenate(valid_indices)


def find_symbol_boundaries(df: pd.DataFrame, symbol_column: str) -> list[int]:
    """Find indices where symbol changes."""
    if symbol_column not in df.columns:
        return []

    symbols = df[symbol_column].values
    return [i for i in range(1, len(symbols)) if symbols[i] != symbols[i - 1]]


__all__ = [
    "DEFAULT_MTF_FEATURES",
    "TIMEFRAME_SUFFIX_PATTERNS",
    "get_timeframe_suffix",
    "extract_timeframe_columns",
    "build_timeframe_feature_map",
    "build_4d_sequence_indices",
    "find_symbol_boundaries",
]
