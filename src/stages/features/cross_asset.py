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

logger = logging.getLogger(__name__)


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
    """
    # Check if both assets are available
    has_both_assets = (mes_close is not None and
                      mgc_close is not None and
                      len(mes_close) == len(df) and
                      len(mgc_close) == len(df))

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
    df['mes_mgc_correlation_20'] = calculate_rolling_correlation_numba(
        mes_returns.astype(np.float64),
        mgc_returns.astype(np.float64),
        period=20
    )

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
    df['mes_mgc_spread_zscore'] = np.where(
        spread_std > 0,
        (spread - spread_mean) / spread_std,
        0.0
    )

    # 3. MES-MGC Beta (rolling beta of MES returns vs MGC returns)
    df['mes_mgc_beta'] = calculate_rolling_beta_numba(
        mes_returns.astype(np.float64),
        mgc_returns.astype(np.float64),
        period=20
    )

    # 4. Relative Strength (momentum divergence)
    # 20-bar cumulative returns for each asset
    mes_cum_ret = pd.Series(mes_returns).rolling(window=20).sum().values
    mgc_cum_ret = pd.Series(mgc_returns).rolling(window=20).sum().values
    df['relative_strength'] = mes_cum_ret - mgc_cum_ret

    feature_metadata['mes_mgc_correlation_20'] = "20-bar rolling correlation between MES and MGC returns"
    feature_metadata['mes_mgc_spread_zscore'] = "Z-score of spread between normalized MES and MGC prices"
    feature_metadata['mes_mgc_beta'] = "Rolling beta of MES returns vs MGC returns"
    feature_metadata['relative_strength'] = "MES return minus MGC return (momentum divergence)"

    return df


__all__ = [
    'add_cross_asset_features',
]
