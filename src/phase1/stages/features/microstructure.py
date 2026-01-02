"""
Market microstructure proxy features derived from OHLCV data.

This module provides functions to estimate microstructure characteristics
without requiring tick-level data. These features proxy for:
- Liquidity measures (Amihud illiquidity, Kyle's lambda)
- Spread estimators (Roll, Corwin-Schultz, relative spread)
- Order flow proxies (volume imbalance, trade intensity)
- Price efficiency measures

Academic References:
- Amihud (2002): "Illiquidity and Stock Returns"
- Roll (1984): "A Simple Implicit Measure of the Effective Bid-Ask Spread"
- Corwin & Schultz (2012): "A Simple Way to Estimate Bid-Ask Spreads"
- Kyle (1985): "Continuous Auctions and Insider Trading"
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _safe_divide(
    numerator: pd.Series, denominator: pd.Series, fill_value: float = np.nan
) -> pd.Series:
    """Safely divide, returning fill_value when denominator is zero or NaN."""
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(fill_value)


def add_amihud_illiquidity(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add Amihud illiquidity ratio.

    The Amihud ratio measures price impact per unit of volume traded.
    Higher values indicate lower liquidity (more price impact per trade).

    Formula: amihud = |return| / volume

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        Rolling periods for smoothed versions. Default: [10, 20]

    Returns
    -------
    pd.DataFrame
        DataFrame with Amihud illiquidity features added
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.info("Skipping Amihud illiquidity (no volume data)")
        return df

    if periods is None:
        periods = [10, 20]

    logger.info(f"Adding Amihud illiquidity with periods: {periods}")

    # Raw Amihud: |return| / volume
    # Use epsilon to avoid division by zero
    returns_abs = df["close"].pct_change().abs()
    volume_safe = df["volume"].replace(0, np.nan)
    amihud_raw = returns_abs / volume_safe

    # ANTI-LOOKAHEAD: shift(1) ensures we use data up to bar[t-1]
    df["micro_amihud"] = amihud_raw.shift(1)

    feature_metadata["micro_amihud"] = "Amihud illiquidity ratio (|ret|/vol, lagged)"

    # Rolling averages for smoothed signals
    for period in periods:
        col_name = f"micro_amihud_{period}"
        df[col_name] = amihud_raw.rolling(window=period).mean().shift(1)
        feature_metadata[col_name] = f"Amihud illiquidity {period}-period avg (lagged)"

    return df


def add_roll_spread(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20
) -> pd.DataFrame:
    """
    Add Roll spread estimator.

    Roll (1984) showed that the bid-ask spread can be estimated from
    the serial covariance of price changes. Negative serial covariance
    implies bid-ask bounce.

    Formula: spread = 2 * sqrt(max(-cov(delta_p, delta_p_lag), 0))

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Rolling window for covariance calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Roll spread estimate added
    """
    logger.info(f"Adding Roll spread with period: {period}")

    # Price changes
    delta_p = df["close"].diff()
    delta_p_lag = delta_p.shift(1)

    # Rolling covariance between price change and lagged price change
    # Negative covariance indicates bid-ask bounce
    roll_cov = delta_p.rolling(window=period).cov(delta_p_lag)

    # Roll spread: 2 * sqrt(-cov) when cov is negative
    # Use max(0, -cov) to handle positive covariance
    roll_spread_raw = 2 * np.sqrt(np.maximum(-roll_cov, 0))

    # ANTI-LOOKAHEAD: shift(1)
    df["micro_roll_spread"] = roll_spread_raw.shift(1)

    # Normalize by close price for comparability
    close_safe = df["close"].replace(0, np.nan)
    roll_spread_pct = (roll_spread_raw / close_safe) * 100
    df["micro_roll_spread_pct"] = roll_spread_pct.shift(1)

    feature_metadata["micro_roll_spread"] = f"Roll spread estimate ({period}, lagged)"
    feature_metadata["micro_roll_spread_pct"] = f"Roll spread as % of price ({period}, lagged)"

    return df


def add_kyle_lambda(
    df: pd.DataFrame, feature_metadata: dict[str, str], volume_period: int = 20
) -> pd.DataFrame:
    """
    Add Kyle's lambda (price impact coefficient).

    Kyle (1985) modeled price impact as lambda * order_flow.
    This proxy estimates lambda as the ratio of price change to volume.

    Formula: lambda = |return| / average_volume

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    volume_period : int, default 20
        Period for volume averaging

    Returns
    -------
    pd.DataFrame
        DataFrame with Kyle's lambda estimate added
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.info("Skipping Kyle's lambda (no volume data)")
        return df

    logger.info(f"Adding Kyle's lambda with volume period: {volume_period}")

    returns_abs = df["close"].pct_change().abs()
    vol_avg = df["volume"].rolling(window=volume_period).mean()
    vol_avg_safe = vol_avg.replace(0, np.nan)

    kyle_lambda_raw = returns_abs / vol_avg_safe

    # ANTI-LOOKAHEAD: shift(1)
    df["micro_kyle_lambda"] = kyle_lambda_raw.shift(1)

    feature_metadata["micro_kyle_lambda"] = (
        f"Kyle's lambda price impact ({volume_period}-period vol, lagged)"
    )

    return df


def add_corwin_schultz_spread(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add Corwin-Schultz high-low spread estimator.

    Corwin & Schultz (2012) derived a spread estimator from daily
    high-low prices, exploiting the fact that highs are more likely
    to be at ask and lows at bid.

    The estimator uses the ratio of 2-day to 1-day high-low ranges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'high' and 'low' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with Corwin-Schultz spread estimate added
    """
    logger.info("Adding Corwin-Schultz high-low spread")

    high = df["high"]
    low = df["low"]

    # Prevent log(0) or log(negative)
    hl_ratio = high / low.replace(0, np.nan)
    hl_ratio = hl_ratio.clip(lower=1e-10)

    # Beta: sum of squared log(high/low) over 2 consecutive bars
    log_hl_sq = np.log(hl_ratio) ** 2
    beta = log_hl_sq + log_hl_sq.shift(1)

    # Gamma: squared log of 2-bar high to 2-bar low ratio
    high_2bar = high.rolling(window=2).max()
    low_2bar = low.rolling(window=2).min()
    low_2bar_safe = low_2bar.replace(0, np.nan)
    gamma = np.log(high_2bar / low_2bar_safe).clip(lower=1e-10) ** 2

    # Alpha calculation
    # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    sqrt_2 = np.sqrt(2)
    denom = 3 - 2 * sqrt_2  # approximately 0.172

    beta_safe = np.maximum(beta, 0)
    gamma_safe = np.maximum(gamma, 0)

    alpha = (np.sqrt(2 * beta_safe) - np.sqrt(beta_safe)) / denom - np.sqrt(gamma_safe / denom)

    # Spread: 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    # This is bounded between 0 and 1
    exp_alpha = np.exp(alpha.clip(-10, 10))  # Clip to avoid overflow
    cs_spread_raw = 2 * (exp_alpha - 1) / (1 + exp_alpha)

    # Spread should be non-negative
    cs_spread_raw = np.maximum(cs_spread_raw, 0)

    # ANTI-LOOKAHEAD: shift(1)
    df["micro_cs_spread"] = pd.Series(cs_spread_raw, index=df.index).shift(1)

    feature_metadata["micro_cs_spread"] = "Corwin-Schultz HL spread estimate (lagged)"

    return df


def add_relative_spread(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add relative spread (high-low range normalized by close).

    A simple proxy for intrabar volatility and potential spread.

    Formula: rel_spread = (high - low) / close

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'high', 'low', and 'close' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        Rolling periods for smoothed versions. Default: [10, 20]

    Returns
    -------
    pd.DataFrame
        DataFrame with relative spread features added
    """
    if periods is None:
        periods = [10, 20]

    logger.info(f"Adding relative spread with periods: {periods}")

    close_safe = df["close"].replace(0, np.nan)
    rel_spread_raw = (df["high"] - df["low"]) / close_safe

    # ANTI-LOOKAHEAD: shift(1)
    df["micro_rel_spread"] = rel_spread_raw.shift(1)

    feature_metadata["micro_rel_spread"] = "Relative spread (HL range / close, lagged)"

    # Rolling averages
    for period in periods:
        col_name = f"micro_rel_spread_{period}"
        df[col_name] = rel_spread_raw.rolling(window=period).mean().shift(1)
        feature_metadata[col_name] = f"Relative spread {period}-period avg (lagged)"

    return df


def add_volume_imbalance(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add volume imbalance proxy.

    Estimates directional order flow using the bar's close position
    within its high-low range. A close near the high suggests buying
    pressure; near the low suggests selling pressure.

    Formula: imbalance = (close - open) / (high - low)

    Bounded between -1 (close at low, open at high) and +1 (close at high, open at low).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with volume imbalance feature added
    """
    logger.info("Adding volume imbalance")

    hl_range = df["high"] - df["low"]
    hl_range_safe = hl_range.replace(0, np.nan)

    # Close-open relative to range
    imbalance_raw = (df["close"] - df["open"]) / hl_range_safe

    # ANTI-LOOKAHEAD: shift(1)
    df["micro_volume_imbalance"] = imbalance_raw.shift(1)

    # Cumulative imbalance (order flow proxy)
    df["micro_cum_imbalance_20"] = imbalance_raw.rolling(window=20).sum().shift(1)

    feature_metadata["micro_volume_imbalance"] = (
        "Volume imbalance proxy (close-open)/range (lagged)"
    )
    feature_metadata["micro_cum_imbalance_20"] = "Cumulative volume imbalance 20-period (lagged)"

    return df


def add_trade_intensity(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add trade intensity (volume relative to recent average).

    High trade intensity can indicate informed trading or
    significant market events.

    Formula: intensity = volume / rolling_avg_volume

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'volume' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        Periods for volume averaging. Default: [20, 50]

    Returns
    -------
    pd.DataFrame
        DataFrame with trade intensity features added
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.info("Skipping trade intensity (no volume data)")
        return df

    if periods is None:
        periods = [20, 50]

    logger.info(f"Adding trade intensity with periods: {periods}")

    for period in periods:
        vol_avg = df["volume"].rolling(window=period).mean()
        vol_avg_safe = vol_avg.replace(0, np.nan)
        intensity_raw = df["volume"] / vol_avg_safe

        col_name = f"micro_trade_intensity_{period}"
        # ANTI-LOOKAHEAD: shift(1)
        df[col_name] = intensity_raw.shift(1)

        feature_metadata[col_name] = f"Trade intensity vs {period}-period vol (lagged)"

    return df


def add_price_efficiency(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add price efficiency ratio.

    Measures how efficiently price moves from point A to B.
    High efficiency (near 1) indicates trending markets.
    Low efficiency indicates choppy/mean-reverting markets.

    Formula: efficiency = |net_change| / sum(|changes|)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        Periods for efficiency calculation. Default: [10, 20]

    Returns
    -------
    pd.DataFrame
        DataFrame with price efficiency features added
    """
    if periods is None:
        periods = [10, 20]

    logger.info(f"Adding price efficiency with periods: {periods}")

    for period in periods:
        # Net change over period
        net_change = (df["close"] - df["close"].shift(period)).abs()

        # Sum of absolute bar-to-bar changes
        abs_changes = df["close"].diff().abs()
        sum_changes = abs_changes.rolling(window=period).sum()

        # Efficiency ratio
        sum_changes_safe = sum_changes.replace(0, np.nan)
        efficiency_raw = net_change / sum_changes_safe

        col_name = f"micro_efficiency_{period}"
        # ANTI-LOOKAHEAD: shift(1)
        df[col_name] = efficiency_raw.shift(1)

        feature_metadata[col_name] = f"Price efficiency ratio {period}-period (lagged)"

    return df


def add_realized_volatility_ratio(
    df: pd.DataFrame, feature_metadata: dict[str, str], short_period: int = 5, long_period: int = 20
) -> pd.DataFrame:
    """
    Add realized volatility ratio (short/long).

    Captures volatility clustering and regime changes.
    Ratio > 1 indicates volatility expansion; < 1 indicates contraction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    short_period : int, default 5
        Short-term volatility window
    long_period : int, default 20
        Long-term volatility window

    Returns
    -------
    pd.DataFrame
        DataFrame with volatility ratio feature added
    """
    logger.info(f"Adding realized volatility ratio (short={short_period}, long={long_period})")

    returns = df["close"].pct_change()

    vol_short = returns.rolling(window=short_period).std()
    vol_long = returns.rolling(window=long_period).std()
    vol_long_safe = vol_long.replace(0, np.nan)

    vol_ratio_raw = vol_short / vol_long_safe

    # ANTI-LOOKAHEAD: shift(1)
    df["micro_vol_ratio"] = vol_ratio_raw.shift(1)

    feature_metadata["micro_vol_ratio"] = (
        f"Realized vol ratio ({short_period}/{long_period}, lagged)"
    )

    return df


def add_microstructure_features(
    df: pd.DataFrame,
    feature_metadata: dict[str, str] | None = None,
    include_amihud: bool = True,
    include_roll: bool = True,
    include_kyle: bool = True,
    include_corwin_schultz: bool = True,
    include_relative_spread: bool = True,
    include_volume_imbalance: bool = True,
    include_trade_intensity: bool = True,
    include_efficiency: bool = True,
    include_vol_ratio: bool = True,
) -> pd.DataFrame:
    """
    Add all market microstructure proxy features from OHLCV data.

    This is the main entry point for microstructure feature engineering.
    These features estimate liquidity, spread, and price impact
    without requiring tick-level data.

    Features added:
    - Amihud illiquidity ratio (price impact per volume)
    - Roll spread estimator (bid-ask from serial covariance)
    - Kyle's lambda (price impact coefficient)
    - Corwin-Schultz spread (bid-ask from high-low)
    - Relative spread (high-low range normalized)
    - Volume imbalance (order flow proxy)
    - Trade intensity (volume vs average)
    - Price efficiency ratio (trending vs choppy)
    - Realized volatility ratio (vol clustering)

    All features are lagged by 1 bar to prevent lookahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV columns
    feature_metadata : Dict[str, str], optional
        Dictionary to store feature descriptions
    include_* : bool
        Flags to include/exclude specific feature groups

    Returns
    -------
    pd.DataFrame
        DataFrame with microstructure features added
    """
    if feature_metadata is None:
        feature_metadata = {}

    logger.info("Adding microstructure proxy features...")

    initial_cols = len(df.columns)

    # Add each feature group if enabled
    if include_amihud:
        df = add_amihud_illiquidity(df, feature_metadata)

    if include_roll:
        df = add_roll_spread(df, feature_metadata)

    if include_kyle:
        df = add_kyle_lambda(df, feature_metadata)

    if include_corwin_schultz:
        df = add_corwin_schultz_spread(df, feature_metadata)

    if include_relative_spread:
        df = add_relative_spread(df, feature_metadata)

    if include_volume_imbalance:
        df = add_volume_imbalance(df, feature_metadata)

    if include_trade_intensity:
        df = add_trade_intensity(df, feature_metadata)

    if include_efficiency:
        df = add_price_efficiency(df, feature_metadata)

    if include_vol_ratio:
        df = add_realized_volatility_ratio(df, feature_metadata)

    added_cols = len(df.columns) - initial_cols
    logger.info(f"Added {added_cols} microstructure features")

    return df


__all__ = [
    "add_microstructure_features",
    "add_amihud_illiquidity",
    "add_roll_spread",
    "add_kyle_lambda",
    "add_corwin_schultz_spread",
    "add_relative_spread",
    "add_volume_imbalance",
    "add_trade_intensity",
    "add_price_efficiency",
    "add_realized_volatility_ratio",
]
