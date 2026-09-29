"""
Convenience functions for Multi-Timeframe (MTF) Feature Integration.
"""

import pandas as pd

from .constants import DEFAULT_MTF_MODE, MTFMode
from .generator import MTFFeatureGenerator


def add_mtf_features(
    df: pd.DataFrame,
    feature_metadata: dict[str, str] | None = None,
    base_timeframe: str = "5min",
    mtf_timeframes: list[str] | None = None,
    mode: MTFMode | str = DEFAULT_MTF_MODE,
    include_ohlcv: bool | None = None,
    include_indicators: bool | None = None,
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
        List of higher timeframes.
        Default: ['15min', '30min', '1h', '4h', 'daily']
    mode : MTFMode or str, default 'both'
        What to generate:
        - 'bars': Only OHLCV data at higher timeframes
        - 'indicators': Only technical indicators at higher timeframes
        - 'both': Both OHLCV bars and indicators
    include_ohlcv : bool, optional (deprecated)
        Use mode='bars' or mode='both' instead
    include_indicators : bool, optional (deprecated)
        Use mode='indicators' or mode='both' instead

    Returns
    -------
    pd.DataFrame
        DataFrame with MTF features added

    Example
    -------
    >>> # Generate both bars and indicators (default)
    >>> df = add_mtf_features(df, feature_metadata)
    >>>
    >>> # Generate only bars
    >>> df = add_mtf_features(df, mode='bars')
    >>>
    >>> # Generate only indicators
    >>> df = add_mtf_features(df, mode=MTFMode.INDICATORS)
    >>>
    >>> # Custom timeframes
    >>> df = add_mtf_features(
    ...     df,
    ...     mtf_timeframes=['1h', '4h', 'daily'],
    ...     mode='both'
    ... )
    """
    generator = MTFFeatureGenerator(
        base_timeframe=base_timeframe,
        mtf_timeframes=mtf_timeframes,
        mode=mode,
        include_ohlcv=include_ohlcv,
        include_indicators=include_indicators,
    )

    result = generator.generate_mtf_features(df)

    # Add metadata if provided
    if feature_metadata is not None:
        col_names = generator.get_mtf_column_names()
        for tf, cols in col_names.items():
            for col in cols:
                if col in result.columns:
                    # Determine feature type from column name
                    if any(
                        col.startswith(ohlcv)
                        for ohlcv in ["open", "high", "low", "close", "volume"]
                    ):
                        feature_metadata[col] = f"MTF OHLCV bar from {tf} timeframe"
                    else:
                        feature_metadata[col] = f"MTF indicator from {tf} timeframe"

    return result
