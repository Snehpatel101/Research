"""
Cross-asset features for feature engineering.

This module provides functions to calculate cross-asset features
between MES (S&P 500 futures) and MGC (Gold futures), including
correlation, spread, beta, and relative strength indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

from .numba_functions import (
    calculate_rolling_correlation_numba,
    calculate_rolling_beta_numba,
)
from src.config.features import CROSS_ASSET_FEATURES

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def add_cross_asset_features(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    mes_close: Optional[np.ndarray] = None,
    mgc_close: Optional[np.ndarray] = None,
    current_symbol: str = ''
) -> pd.DataFrame:
    """
    Add cross-asset features between MES (S&P 500 futures) and MGC (Gold futures).

    These features capture:
    - Correlation: Rolling correlation between MES and MGC returns
    - Spread Z-score: Normalized spread between the two assets
    - Beta: Rolling beta of MES returns vs MGC returns
    - Relative Strength: Momentum divergence (MES return - MGC return)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add features to
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    mes_close : np.ndarray, optional
        MES close prices (aligned with df index)
    mgc_close : np.ndarray, optional
        MGC close prices (aligned with df index)
    current_symbol : str
        The symbol being processed ('MES' or 'MGC')

    Returns
    -------
    pd.DataFrame
        DataFrame with cross-asset features added

    Notes
    -----
    CRITICAL FIX (2025-12-21): Cross-asset features are computed using rolling
    windows on the provided arrays. The caller MUST ensure that mes_close and
    mgc_close arrays correspond to the SAME time period as df.

    LEAKAGE WARNING:
    - This function should be called BEFORE train/val/test splitting (e.g., Stage 3)
    - If called after splitting, the caller MUST provide subset arrays matching
      the split (not the full dataset arrays)
    - Rolling statistics use ONLY the data in the provided arrays
    - Using full-dataset arrays on a split subset would leak future information

    CORRECT USAGE:
    - Stage 3 (pre-split): Pass full aligned arrays -> cross-asset features use
      only past data via rolling windows (no leakage)
    - Post-split: Pass split-specific arrays matching df's time range

    INCORRECT USAGE (causes leakage):
    - Post-split: Passing full dataset arrays when df is a split subset
    """
    # === CONFIG-BASED EARLY RETURN ===
    # Check if cross-asset features are disabled in config
    if not CROSS_ASSET_FEATURES.get('enabled', True):
        logger.info("Cross-asset features disabled in config (CROSS_ASSET_FEATURES['enabled'] = False)")
        # Set cross-asset features to NaN when disabled
        df['mes_mgc_correlation_20'] = np.nan
        df['mes_mgc_spread_zscore'] = np.nan
        df['mes_mgc_beta'] = np.nan
        df['relative_strength'] = np.nan

        feature_metadata['mes_mgc_correlation_20'] = "20-bar rolling correlation between MES and MGC returns"
        feature_metadata['mes_mgc_spread_zscore'] = "Z-score of spread between normalized MES and MGC prices"
        feature_metadata['mes_mgc_beta'] = "Rolling beta of MES returns vs MGC returns"
        feature_metadata['relative_strength'] = "MES return minus MGC return (momentum divergence)"

        return df

    # === TIMESTAMP AND LENGTH VALIDATION ===
    # Validate that provided arrays match df length to prevent misalignment
    has_both_assets = (mes_close is not None and
                      mgc_close is not None and
                      len(mes_close) == len(df) and
                      len(mgc_close) == len(df))

    # Additional validation: Check that arrays are non-empty and properly sized
    if has_both_assets:
        if len(mes_close) == 0 or len(mgc_close) == 0:
            logger.warning("Empty arrays provided for cross-asset features")
            has_both_assets = False
        elif len(mes_close) != len(mgc_close):
            logger.warning(
                f"Array length mismatch: MES={len(mes_close)}, MGC={len(mgc_close)}. "
                "Skipping cross-asset features."
            )
            has_both_assets = False

    if not has_both_assets:
        logger.info("Skipping cross-asset features (requires both MES and MGC data)")
        # Set cross-asset features to NaN when only one symbol is present
        df['mes_mgc_correlation_20'] = np.nan
        df['mes_mgc_spread_zscore'] = np.nan
        df['mes_mgc_beta'] = np.nan
        df['relative_strength'] = np.nan

        feature_metadata['mes_mgc_correlation_20'] = "20-bar rolling correlation between MES and MGC returns"
        feature_metadata['mes_mgc_spread_zscore'] = "Z-score of spread between normalized MES and MGC prices"
        feature_metadata['mes_mgc_beta'] = "Rolling beta of MES returns vs MGC returns"
        feature_metadata['relative_strength'] = "MES return minus MGC return (momentum divergence)"

        return df

    logger.info("Adding cross-asset features (MES-MGC)...")
    logger.info(f"Array lengths validated: MES={len(mes_close)}, MGC={len(mgc_close)}, DF={len(df)}")

    # Calculate percentage returns for both assets (pct_change style)
    # Returns[t] = (Close[t] - Close[t-1]) / Close[t-1]
    # First element is 0 since there's no prior bar to compare against
    # Uses np.divide with where parameter to safely handle zero-division
    mes_returns = np.zeros(len(mes_close))
    mgc_returns = np.zeros(len(mgc_close))
    mes_returns[1:] = np.divide(
        mes_close[1:] - mes_close[:-1],
        mes_close[:-1],
        out=np.zeros_like(mes_close[1:]),
        where=mes_close[:-1] != 0
    )
    mgc_returns[1:] = np.divide(
        mgc_close[1:] - mgc_close[:-1],
        mgc_close[:-1],
        out=np.zeros_like(mgc_close[1:]),
        where=mgc_close[:-1] != 0
    )

    # 1. MES-MGC Correlation (20-bar rolling)
    # ANTI-LOOKAHEAD: Shift 1 bar forward so correlation at bar[t] uses data up to bar[t-1]
    correlation = calculate_rolling_correlation_numba(
        mes_returns.astype(np.float64),
        mgc_returns.astype(np.float64),
        period=20
    )
    df['mes_mgc_correlation_20'] = pd.Series(correlation).shift(1).values

    # 2. MES-MGC Spread Z-score
    # Normalize prices to allow comparison (z-score of each)
    period = 20

    # Calculate rolling mean and std for MES
    mes_rolling_mean = pd.Series(mes_close).rolling(window=period).mean().values
    mes_rolling_std = pd.Series(mes_close).rolling(window=period).std().values

    # Calculate rolling mean and std for MGC
    mgc_rolling_mean = pd.Series(mgc_close).rolling(window=period).mean().values
    mgc_rolling_std = pd.Series(mgc_close).rolling(window=period).std().values

    # Normalized prices (z-scores)
    mes_normalized = np.where(
        mes_rolling_std > 0,
        (mes_close - mes_rolling_mean) / mes_rolling_std,
        0.0
    )
    mgc_normalized = np.where(
        mgc_rolling_std > 0,
        (mgc_close - mgc_rolling_mean) / mgc_rolling_std,
        0.0
    )

    # Spread between normalized prices
    spread = mes_normalized - mgc_normalized

    # Z-score of the spread
    spread_mean = pd.Series(spread).rolling(window=period).mean().values
    spread_std = pd.Series(spread).rolling(window=period).std().values
    spread_zscore = np.where(
        spread_std > 0,
        (spread - spread_mean) / spread_std,
        0.0
    )
    # ANTI-LOOKAHEAD: Shift 1 bar forward so z-score at bar[t] uses data up to bar[t-1]
    df['mes_mgc_spread_zscore'] = pd.Series(spread_zscore).shift(1).values

    # 3. MES-MGC Beta (rolling beta of MES returns vs MGC returns)
    # ANTI-LOOKAHEAD: Shift 1 bar forward so beta at bar[t] uses data up to bar[t-1]
    beta = calculate_rolling_beta_numba(
        mes_returns.astype(np.float64),
        mgc_returns.astype(np.float64),
        period=20
    )
    df['mes_mgc_beta'] = pd.Series(beta).shift(1).values

    # 4. Relative Strength (momentum divergence)
    # 20-bar cumulative returns for each asset
    # ANTI-LOOKAHEAD: Shift returns 1 bar forward so cumulative return at bar[t]
    # uses data up to bar[t-1] and doesn't include current bar's return
    mes_returns_shifted = pd.Series(mes_returns).shift(1).values
    mgc_returns_shifted = pd.Series(mgc_returns).shift(1).values
    mes_cum_ret = pd.Series(mes_returns_shifted).rolling(window=20).sum().values
    mgc_cum_ret = pd.Series(mgc_returns_shifted).rolling(window=20).sum().values
    df['relative_strength'] = mes_cum_ret - mgc_cum_ret

    feature_metadata['mes_mgc_correlation_20'] = "20-bar rolling correlation between MES and MGC returns"
    feature_metadata['mes_mgc_spread_zscore'] = "Z-score of spread between normalized MES and MGC prices"
    feature_metadata['mes_mgc_beta'] = "Rolling beta of MES returns vs MGC returns"
    feature_metadata['relative_strength'] = "MES return minus MGC return (momentum divergence)"

    return df


__all__ = [
    'add_cross_asset_features',
]
