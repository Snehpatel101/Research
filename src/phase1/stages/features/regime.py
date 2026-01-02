"""
Regime indicator features for feature engineering.

This module provides functions to calculate market regime indicators
including volatility regime, trend regime, and market structure classifications.

The regime features can be used as:
1. Features: Add regime columns to feature DataFrame for model training
2. Filters: Use regime to select which model to apply (model per regime)
3. Adaptive Parameters: Adjust triple-barrier parameters per regime

For advanced regime detection with Hurst exponent and composite regimes,
see the `stages.regime` package.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def add_regime_features(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    use_advanced: bool = False,
    regime_config: dict | None = None,
) -> pd.DataFrame:
    """
    Add regime indicators to DataFrame.

    Calculates volatility regime (high/normal/low) and trend regime
    (up/down/sideways). Optionally uses advanced regime detection
    including Hurst exponent for market structure classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with required columns (depends on features used):
        - For basic: hvol_20, sma_50, sma_200, close
        - For advanced: high, low, close (computes indicators internally)
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    use_advanced : bool, default False
        If True, use advanced regime detection from stages.regime package
    regime_config : Optional[Dict]
        Configuration for advanced regime detection

    Returns
    -------
    pd.DataFrame
        DataFrame with regime features added

    Notes
    -----
    Basic regime detection uses pre-computed features (hvol_20, sma_50, etc.)
    and is faster but less sophisticated.

    Advanced regime detection computes indicators internally and includes
    market structure analysis via Hurst exponent.
    """
    logger.info("Adding regime features...")

    if use_advanced:
        return _add_advanced_regime_features(df, feature_metadata, regime_config)
    else:
        return _add_basic_regime_features(df, feature_metadata)


def _add_basic_regime_features(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add basic regime features using pre-computed indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hvol_20, sma_50, sma_200, and close columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with basic regime features added
    """
    # ANTI-LOOKAHEAD: Regime features at bar[t] must use data up to bar[t-1]
    # Since input features (hvol_20, sma_50, etc.) are already lagged by 1 bar,
    # we compare them directly but ensure any use of close is also lagged

    # Volatility regime (high/low based on historical volatility)
    if "hvol_20" in df.columns:
        # hvol_20 is already lagged, so median calculation is safe
        hvol_median = df["hvol_20"].rolling(window=100, min_periods=1).median()
        df["volatility_regime"] = (df["hvol_20"] > hvol_median).astype(int)
        feature_metadata["volatility_regime"] = "Volatility regime (1=high, 0=low, lagged)"
    else:
        logger.debug("hvol_20 not found, skipping basic volatility regime")

    # Trend regime based on price vs moving averages
    if "sma_50" in df.columns and "sma_200" in df.columns:
        # sma_50 and sma_200 are already lagged, use lagged close for comparison
        close_lagged = df["close"].shift(1)

        # Uptrend: price > SMA50 > SMA200 (all using t-1 data)
        uptrend = (close_lagged > df["sma_50"]) & (df["sma_50"] > df["sma_200"])

        # Downtrend: price < SMA50 < SMA200
        downtrend = (close_lagged < df["sma_50"]) & (df["sma_50"] < df["sma_200"])

        # Trend regime: 1=up, -1=down, 0=sideways
        df["trend_regime"] = np.where(uptrend, 1, np.where(downtrend, -1, 0))
        feature_metadata["trend_regime"] = "Trend regime (1=up, -1=down, 0=sideways, lagged)"
    else:
        logger.debug("SMA features not found, skipping basic trend regime")

    return df


def _add_advanced_regime_features(
    df: pd.DataFrame, feature_metadata: dict[str, str], regime_config: dict | None = None
) -> pd.DataFrame:
    """
    Add advanced regime features using the regime detection package.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    regime_config : Optional[Dict]
        Configuration for regime detectors

    Returns
    -------
    pd.DataFrame
        DataFrame with advanced regime features added
    """
    try:
        from src.phase1.stages.regime import (
            add_regime_features_to_dataframe,
        )
    except ImportError:
        try:
            from stages.regime import (
                add_regime_features_to_dataframe,
            )
        except ImportError as e:
            logger.warning(
                f"Advanced regime detection not available: {e}. "
                f"Falling back to basic regime features."
            )
            return _add_basic_regime_features(df, feature_metadata)

    logger.info("Using advanced regime detection...")

    # Use the convenience function which handles everything
    df = add_regime_features_to_dataframe(
        df, config=regime_config, feature_metadata=feature_metadata
    )

    return df


def add_volatility_regime(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add volatility regime indicator only.

    Standalone function for volatility regime classification.
    Uses pre-computed hvol_20 for simple high/low classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hvol_20 column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with volatility regime added
    """
    logger.info("Adding volatility regime...")

    if "hvol_20" not in df.columns:
        logger.warning("hvol_20 not found, skipping volatility regime")
        return df

    # ANTI-LOOKAHEAD: hvol_20 is already lagged, so median is safe
    hvol_median = df["hvol_20"].rolling(window=100, min_periods=1).median()
    df["volatility_regime"] = (df["hvol_20"] > hvol_median).astype(int)
    feature_metadata["volatility_regime"] = "Volatility regime (1=high, 0=low, lagged)"

    return df


def add_trend_regime(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add trend regime indicator only.

    Standalone function for trend regime classification.
    Uses pre-computed SMA features for trend direction classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sma_50, sma_200, and close columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with trend regime added
    """
    logger.info("Adding trend regime...")

    if "sma_50" not in df.columns or "sma_200" not in df.columns:
        logger.warning("SMA features not found, skipping trend regime")
        return df

    # ANTI-LOOKAHEAD: sma_50 and sma_200 are already lagged, use lagged close
    close_lagged = df["close"].shift(1)

    # Uptrend: price > SMA50 > SMA200 (all using t-1 data)
    uptrend = (close_lagged > df["sma_50"]) & (df["sma_50"] > df["sma_200"])

    # Downtrend: price < SMA50 < SMA200
    downtrend = (close_lagged < df["sma_50"]) & (df["sma_50"] < df["sma_200"])

    # Trend regime: 1=up, -1=down, 0=sideways
    df["trend_regime"] = np.where(uptrend, 1, np.where(downtrend, -1, 0))
    feature_metadata["trend_regime"] = "Trend regime (1=up, -1=down, 0=sideways, lagged)"

    return df


def add_structure_regime(
    df: pd.DataFrame, feature_metadata: dict[str, str], lookback: int = 100
) -> pd.DataFrame:
    """
    Add market structure regime based on Hurst exponent.

    This function uses the advanced regime detection package for
    Hurst exponent calculation.

    ANTI-LOOKAHEAD: The regime is shifted by 1 bar to prevent lookahead bias.
    The regime at bar N reflects the market structure from bars 0..N-1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with close prices
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    lookback : int, default 100
        Rolling window for Hurst calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with structure regime added (shifted by 1 bar)
    """
    logger.info("Adding structure regime...")

    try:
        from src.phase1.stages.regime import MarketStructureDetector
    except ImportError:
        try:
            from stages.regime import MarketStructureDetector
        except ImportError as e:
            logger.warning(f"Structure regime not available: {e}")
            return df

    detector = MarketStructureDetector(lookback=lookback)
    regimes = detector.detect(df)

    # ANTI-LOOKAHEAD: Shift by 1 bar to prevent using current bar data
    df["structure_regime"] = regimes.shift(1)
    feature_metadata["structure_regime"] = (
        "Market structure (mean_reverting/random/trending) based on Hurst (shifted by 1 bar)"
    )

    return df


def add_all_regime_features(
    df: pd.DataFrame, feature_metadata: dict[str, str], config: dict | None = None
) -> pd.DataFrame:
    """
    Add all available regime features using advanced detection.

    This is a convenience function that adds volatility, trend,
    and structure regime features with optional configuration.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC data
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    config : Optional[Dict]
        Configuration dict for regime detectors

    Returns
    -------
    pd.DataFrame
        DataFrame with all regime features added
    """
    return add_regime_features(df, feature_metadata, use_advanced=True, regime_config=config)


__all__ = [
    "add_regime_features",
    "add_volatility_regime",
    "add_trend_regime",
    "add_structure_regime",
    "add_all_regime_features",
]
