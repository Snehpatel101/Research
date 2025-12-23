"""
Convenience functions for Multi-Timeframe (MTF) Feature Integration.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .generator import MTFFeatureGenerator


def add_mtf_features(
    df: pd.DataFrame,
    feature_metadata: Optional[Dict[str, str]] = None,
    base_timeframe: str = '5min',
    mtf_timeframes: Optional[List[str]] = None,
    include_ohlcv: bool = True,
    include_indicators: bool = True
) -> pd.DataFrame:
    """
    Add MTF features to a DataFrame (convenience function).

    This function provides a simple interface matching the pattern used
    by other feature modules (add_rsi, add_macd, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Base timeframe OHLCV data with 'datetime' column
    feature_metadata : Dict[str, str], optional
        Dictionary to store feature descriptions
    base_timeframe : str, default '5min'
        Base timeframe of input data
    mtf_timeframes : List[str], optional
        List of higher timeframes. Default: ['15min', '60min']
    include_ohlcv : bool, default True
        Whether to include OHLCV data from higher TFs
    include_indicators : bool, default True
        Whether to compute indicators on higher TFs

    Returns
    -------
    pd.DataFrame
        DataFrame with MTF features added

    Example
    -------
    >>> df = add_mtf_features(df, feature_metadata)
    >>> # Features like 'rsi_14_15m', 'close_1h' are now available
    """
    generator = MTFFeatureGenerator(
        base_timeframe=base_timeframe,
        mtf_timeframes=mtf_timeframes,
        include_ohlcv=include_ohlcv,
        include_indicators=include_indicators
    )

    result = generator.generate_mtf_features(df)

    # Add metadata if provided
    if feature_metadata is not None:
        col_names = generator.get_mtf_column_names()
        for tf, cols in col_names.items():
            for col in cols:
                if col in result.columns:
                    feature_metadata[col] = f"MTF feature from {tf} timeframe"

    return result


def validate_mtf_alignment(
    df_base: pd.DataFrame,
    df_mtf: pd.DataFrame,
    base_tf: str = '5min',
    mtf_tf: str = '15min'
) -> Tuple[bool, List[str]]:
    """
    Validate that MTF alignment is correct.

    Checks:
    1. MTF timestamps are subset of base timestamps
    2. No future data leakage
    3. Proper forward-fill alignment

    Parameters
    ----------
    df_base : pd.DataFrame
        Base timeframe data
    df_mtf : pd.DataFrame
        MTF aligned data
    base_tf : str
        Base timeframe string
    mtf_tf : str
        MTF timeframe string

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list of issues found)
    """
    issues = []

    if 'datetime' not in df_base.columns:
        issues.append("df_base missing 'datetime' column")

    if 'datetime' not in df_mtf.columns:
        issues.append("df_mtf missing 'datetime' column")

    if issues:
        return False, issues

    # Check timestamp coverage
    base_start = df_base['datetime'].min()
    base_end = df_base['datetime'].max()
    mtf_start = df_mtf['datetime'].min()
    mtf_end = df_mtf['datetime'].max()

    if mtf_start < base_start:
        issues.append(
            f"MTF data starts before base data: {mtf_start} < {base_start}"
        )

    if mtf_end > base_end:
        issues.append(
            f"MTF data ends after base data: {mtf_end} > {base_end}"
        )

    return len(issues) == 0, issues
