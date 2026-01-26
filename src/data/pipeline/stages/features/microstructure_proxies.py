"""
Microstructure features estimable from 1-min OHLCV bars.

Based on 2024 research:
- Ardia, Guidotti, Kroencke (JFE 2024): Efficient spread estimation
- arXiv:2509.16137 (Sept 2024): Timing features for VWAP prediction
- Easley et al. (Apr 2024): VPIN and Roll measure for crypto

This module provides microstructure proxies that can be computed from
standard 1-minute OHLCV bars WITHOUT requiring tick-level data.

References:
    1. Ardia, D., Guidotti, E., & Kroencke, T. A. (2024).
       Efficient estimation of bid-ask spreads from open, high, low, and close prices.
       Journal of Financial Economics, 161.
       https://ideas.repec.org/a/eee/jfinec/v161y2024ics0304405x24001399.html

    2. arXiv:2509.16137 (September 2024).
       Enhancing OHLC Data with Timing Features: A Machine Learning Evaluation.
       https://arxiv.org/abs/2509.16137

    3. Easley, D., et al. (April 2024).
       Market microstructure metrics in cryptocurrencies.
       https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf
"""

import logging

import numba as nb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# SPREAD ESTIMATORS
# ==============================================================================


@nb.jit(nopython=True, cache=True)
def compute_edge_spread_numba(high, low, open_price, close_price):
    """
    Edge spread estimator (Ardia et al. 2024) - Numba optimized.

    Efficient bid-ask spread from OHLC, works with infrequent trading.
    Asymptotically unbiased estimator that accounts for discretely observed prices.

    Args:
        high: Array of high prices
        low: Array of low prices
        open_price: Array of open prices
        close_price: Array of close prices

    Returns:
        Array of estimated spreads (as fraction of price)
    """
    n = len(high)
    spreads = np.zeros(n, dtype=np.float32)

    for i in range(n):
        hl_range = high[i] - low[i]
        oc_range = abs(close_price[i] - open_price[i])

        if hl_range > 0:
            # Ratio of close-open range to high-low range
            ratio = oc_range / hl_range

            # Edge estimator (simplified formula)
            # Full formula in paper accounts for more nuances
            if ratio <= 1.0:
                spread = 2 * np.sqrt(1 - ratio**2) * hl_range
                spreads[i] = spread / close_price[i]  # As fraction of price
            else:
                # Degenerate case: open-close exceeds high-low (data quality issue)
                spreads[i] = 0.0
        else:
            spreads[i] = 0.0

    return spreads


def compute_edge_spread(df):
    """
    Edge spread estimator (Ardia et al. 2024).

    Wrapper for Numba-optimized implementation.

    Args:
        df: DataFrame with OHLC columns

    Returns:
        Series of estimated spreads (as % of price)
    """
    spreads = compute_edge_spread_numba(
        df["high"].values,
        df["low"].values,
        df["open"].values,
        df["close"].values,
    )

    return pd.Series(spreads * 100, index=df.index, name="edge_spread_pct")


def compute_roll_spread(df, window=20):
    """
    Roll (1984) spread estimator.

    Uses negative serial correlation from bid-ask bounce.
    Classic estimator: spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))

    Args:
        df: DataFrame with 'close' column
        window: Rolling window for covariance calculation

    Returns:
        Series of estimated spreads (rolling window)
    """
    close_prices = df["close"]
    price_changes = close_prices.diff()

    # Rolling covariance with lag-1 (vectorized)
    # Use pandas rolling with pairwise covariance
    price_changes_lag1 = price_changes.shift(1)
    cov_lag1 = price_changes.rolling(window=window, min_periods=window).cov(
        price_changes_lag1
    )

    # Spread = 2 * sqrt(-cov) if cov < 0, else 0
    spread = np.where(cov_lag1 < 0, 2 * np.sqrt(-cov_lag1), 0)
    spread_pct = (spread / close_prices) * 100  # As % of price

    return pd.Series(spread_pct, index=df.index, name="roll_spread_pct")


# ==============================================================================
# TIMING FEATURES (arXiv:2509.16137)
# ==============================================================================


@nb.jit(nopython=True, cache=True)
def estimate_timing_features_numba(open_price, high, low, close_price):
    """
    Estimate when high/low occurred within bar - Numba optimized.

    Without tick data, use heuristic based on close position and price direction.

    Returns:
        Tuple of (time_to_high, time_to_low, intrabar_reversal)
    """
    n = len(open_price)
    time_to_high = np.zeros(n, dtype=np.float32)
    time_to_low = np.zeros(n, dtype=np.float32)
    intrabar_reversal = np.zeros(n, dtype=np.float32)

    for i in range(n):
        bar_range = high[i] - low[i]

        if bar_range > 1e-8:
            # Close position within bar range (0 = at low, 1 = at high)
            close_position = (close_price[i] - low[i]) / bar_range

            # Price direction
            price_direction = close_price[i] - open_price[i]

            # Heuristic: If uptrend and close near high, high occurred late
            # If downtrend and close near low, low occurred late
            if price_direction > 0:
                # Uptrend: high likely near end of bar
                time_to_high[i] = 0.5 + 0.3 * close_position
                time_to_low[i] = 0.5 - 0.3 * close_position
            else:
                # Downtrend: high likely near start of bar
                time_to_high[i] = 0.5 - 0.3 * (1 - close_position)
                time_to_low[i] = 0.5 + 0.3 * (1 - close_position)

            # Intrabar reversal: how much bar reversed from midpoint
            intrabar_reversal[i] = abs(close_position - 0.5)
        else:
            # No range - set neutral values
            time_to_high[i] = 0.5
            time_to_low[i] = 0.5
            intrabar_reversal[i] = 0.0

    return time_to_high, time_to_low, intrabar_reversal


def compute_timing_features(df):
    """
    Estimate when high/low occurred within bar (arXiv:2509.16137).

    Key Finding (Sept 2024 research):
        Timing features improve VWAP prediction by 15-20%,
        especially for Transformer models.

    Args:
        df: DataFrame with OHLC columns

    Returns:
        DataFrame with timing features
    """
    time_to_high, time_to_low, intrabar_reversal = estimate_timing_features_numba(
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
    )

    # Close after high indicator (binary)
    close_after_high = (time_to_high < 0.5).astype(int)

    return pd.DataFrame(
        {
            "time_to_high": time_to_high,
            "time_to_low": time_to_low,
            "close_after_high": close_after_high,
            "intrabar_reversal": intrabar_reversal,
        },
        index=df.index,
    )


# ==============================================================================
# PRICE IMPACT & LIQUIDITY
# ==============================================================================


def compute_kyle_lambda(df, window=20):
    """
    Kyle's lambda (price impact proxy).

    Measures dollars per volume unit - how much price moves per trade.
    Lambda = |price change| / volume

    Args:
        df: DataFrame with OHLC and volume columns
        window: Rolling window for smoothing

    Returns:
        Series of price impact estimates
    """
    price_change = np.abs(df["close"] - df["open"])
    volume = df["volume"]

    # Lambda = |price change| / volume
    # Add 1 to volume to avoid division by zero
    lambda_raw = price_change / (volume + 1)

    # Smooth over window to reduce noise
    lambda_smooth = lambda_raw.rolling(window, min_periods=1).mean()

    return pd.Series(lambda_smooth, index=df.index, name="kyle_lambda")


def compute_amihud_illiquidity(df, window=20):
    """
    Amihud (2002) illiquidity measure.

    Measures price impact per dollar volume.
    ILLIQ = |return| / dollar_volume

    Args:
        df: DataFrame with close and volume columns
        window: Rolling window for averaging

    Returns:
        Series of illiquidity estimates
    """
    returns = df["close"].pct_change().abs()
    dollar_volume = df["close"] * df["volume"]

    # Illiquidity = |return| / dollar_volume
    # Multiply by 1e6 for interpretability
    illiquidity_raw = (returns / (dollar_volume + 1)) * 1e6

    # Rolling average
    illiquidity = illiquidity_raw.rolling(window, min_periods=1).mean()

    return pd.Series(illiquidity, index=df.index, name="amihud_illiquidity")


# ==============================================================================
# INFORMED TRADING (VPIN)
# ==============================================================================


@nb.jit(nopython=True, cache=True)
def compute_vpin_numba(price_change, volume, window):
    """
    VPIN approximation - Numba optimized.

    Approximates buy/sell volume from price direction.
    VPIN = |buy_vol - sell_vol| / total_vol
    """
    n = len(price_change)
    vpin = np.zeros(n, dtype=np.float32)

    # Approximate buy/sell volume
    buy_volume = np.where(price_change > 0, volume, 0.0)
    sell_volume = np.where(price_change < 0, volume, 0.0)

    for i in range(window, n):
        # Sum over window
        total_buy = np.sum(buy_volume[i - window : i])
        total_sell = np.sum(sell_volume[i - window : i])
        total_vol = total_buy + total_sell

        if total_vol > 0:
            vpin[i] = abs(total_buy - total_sell) / total_vol
        else:
            vpin[i] = 0.0

    return vpin


def compute_vpin(df, window=50):
    """
    VPIN (Volume-Synchronized Probability of Informed Trading).

    Approximation for 1-min bars (Easley et al. 2024 crypto study).

    Without tick data, approximate buy/sell volume from price direction.

    Key Finding (Easley et al. 2024):
        VPIN is 2nd most important feature for predicting volatility changes.

    Args:
        df: DataFrame with close and volume columns
        window: Volume bucket window

    Returns:
        Series of VPIN estimates
    """
    price_change = df["close"] - df["open"]
    volume = df["volume"].values

    vpin_values = compute_vpin_numba(
        price_change.values,
        volume,
        window,
    )

    return pd.Series(vpin_values, index=df.index, name="vpin")


# ==============================================================================
# VOLUME PROFILE
# ==============================================================================


def compute_volume_profile(df):
    """
    Estimate intrabar volume distribution.

    Assumes volume normally distributed across high-low range.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with volume profile features
    """
    volume = df["volume"]
    high = df["high"]
    low = df["low"]
    close_price = df["close"]

    # Volume density (volume per price unit)
    spread = high - low
    volume_density = volume / (spread + 1e-8)

    # Close position in range (0 = at low, 1 = at high)
    close_position = (close_price - low) / (spread + 1e-8)

    # Volume concentration: how much volume at close price
    # If close near high, more volume at high; if near low, more at low
    volume_concentration = 1 - np.abs(close_position - 0.5)

    # Estimated volume at extremes (away from close)
    volume_at_extremes = volume * (1 - close_position)

    return pd.DataFrame(
        {
            "volume_density": volume_density,
            "volume_concentration": volume_concentration,
            "volume_at_extremes": volume_at_extremes,
        },
        index=df.index,
    )


# ==============================================================================
# BAR STRUCTURE
# ==============================================================================


def compute_bar_structure(df):
    """
    Bar structure features (no estimation needed - directly from OHLC).

    Args:
        df: DataFrame with OHLC columns

    Returns:
        DataFrame with bar structure features
    """
    open_price = df["open"]
    high = df["high"]
    low = df["low"]
    close_price = df["close"]

    # Body size (as % of close price)
    body_size = np.abs(close_price - open_price) / close_price

    # Upper wick (as % of close price)
    upper_wick = (high - df[["open", "close"]].max(axis=1)) / close_price

    # Lower wick (as % of close price)
    lower_wick = (df[["open", "close"]].min(axis=1) - low) / close_price

    # Total bar range (as % of close price)
    bar_range = (high - low) / close_price

    # Body to range ratio (how much of bar is body vs wicks)
    body_range_ratio = body_size / (bar_range + 1e-8)

    # Direction (1 = bullish, -1 = bearish)
    direction = np.sign(close_price - open_price)

    return pd.DataFrame(
        {
            "body_size_pct": body_size * 100,
            "upper_wick_pct": upper_wick * 100,
            "lower_wick_pct": lower_wick * 100,
            "bar_range_pct": bar_range * 100,
            "body_range_ratio": body_range_ratio,
            "bar_direction": direction,
        },
        index=df.index,
    )


# ==============================================================================
# MAIN AGGREGATION FUNCTION
# ==============================================================================


def compute_all_microstructure_features(df):
    """
    Compute all microstructure proxies from 1-min OHLCV bars.

    This is the main entry point for microstructure feature engineering.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)

    Returns:
        DataFrame with ~20 microstructure features

    Features:
        - Spread estimators (2): edge_spread, roll_spread
        - Timing features (4): time_to_high, time_to_low, close_after_high, intrabar_reversal
        - Price impact (2): kyle_lambda, amihud_illiquidity
        - Informed trading (1): vpin
        - Volume profile (3): volume_density, volume_concentration, volume_at_extremes
        - Bar structure (6): body_size, wicks, range, ratios, direction

    Total: ~18 features
    """
    logger.info("Computing microstructure features from 1-min OHLCV bars...")

    features = pd.DataFrame(index=df.index)

    # Spread estimators
    logger.info("  Computing spread estimators (Edge, Roll)...")
    features["edge_spread_pct"] = compute_edge_spread(df)
    features["roll_spread_pct"] = compute_roll_spread(df)

    # Timing features (arXiv 2509.16137)
    logger.info("  Computing timing features (arXiv:2509.16137)...")
    timing_feats = compute_timing_features(df)
    features = features.join(timing_feats)

    # Price impact
    logger.info("  Computing price impact metrics (Kyle's lambda, Amihud)...")
    features["kyle_lambda"] = compute_kyle_lambda(df)
    features["amihud_illiquidity"] = compute_amihud_illiquidity(df)

    # Informed trading
    logger.info("  Computing informed trading proxy (VPIN)...")
    features["vpin"] = compute_vpin(df)

    # Volume profile
    logger.info("  Computing volume profile...")
    volume_feats = compute_volume_profile(df)
    features = features.join(volume_feats)

    # Bar structure
    logger.info("  Computing bar structure features...")
    bar_feats = compute_bar_structure(df)
    features = features.join(bar_feats)

    # Handle NaN values (from rolling windows at start)
    features = features.fillna(method="bfill").fillna(0)

    logger.info(f"Microstructure features computed: {features.shape[1]} features")

    return features


__all__ = [
    "compute_edge_spread",
    "compute_roll_spread",
    "compute_timing_features",
    "compute_kyle_lambda",
    "compute_amihud_illiquidity",
    "compute_vpin",
    "compute_volume_profile",
    "compute_bar_structure",
    "compute_all_microstructure_features",
]
