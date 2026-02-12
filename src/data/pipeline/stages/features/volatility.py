"""
Volatility indicator features for feature engineering.

This module provides functions to calculate volatility-based technical
indicators including ATR, Bollinger Bands, Keltner Channels, and various
historical volatility measures.
"""

import logging

import numpy as np
import pandas as pd

from .constants import get_annualization_factor
from .numba_functions import calculate_atr_numba, calculate_ema_numba

logger = logging.getLogger(__name__)


def add_atr(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add Average True Range features.

    Calculates ATR for multiple periods with both absolute and percentage values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of ATR periods. Default: [7, 14, 21]

    Returns
    -------
    pd.DataFrame
        DataFrame with ATR features added
    """
    if periods is None:
        periods = [7, 14, 21]

    logger.info(f"Adding ATR features with periods: {periods}")

    for period in periods:
        atr = calculate_atr_numba(df["high"].values, df["low"].values, df["close"].values, period)
        # ANTI-LOOKAHEAD: shift(1) ensures ATR at bar[t] uses data up to bar[t-1]
        df[f"atr_{period}"] = pd.Series(atr).shift(1).values

        # ATR as percentage of close (safe division)
        # Use lagged close to match lagged ATR
        close_lagged = df["close"].shift(1).replace(0, np.nan)
        df[f"atr_pct_{period}"] = (df[f"atr_{period}"] / close_lagged) * 100

        feature_metadata[f"atr_{period}"] = f"Average True Range ({period}, lagged)"
        feature_metadata[f"atr_pct_{period}"] = f"ATR as % of price ({period}, lagged)"

    return df


def add_bollinger_bands(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, std_mult: float = 2.0
) -> pd.DataFrame:
    """
    Add Bollinger Bands features.

    Calculates Bollinger Bands with configurable period and standard deviation,
    band width (normalized), and price position within the bands.

    All features are made stationary using z-scores and normalized values
    to avoid dependency on absolute price levels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Bollinger Band period
    std_mult : float, default 2.0
        Standard deviation multiplier

    Returns
    -------
    pd.DataFrame
        DataFrame with Bollinger Band features added
    """
    logger.info(f"Adding Bollinger Bands with period: {period}")

    # ANTI-LOOKAHEAD: shift(1) ensures BB at bar[t] uses data up to bar[t-1]
    bb_middle = df["close"].rolling(window=period).mean().shift(1)
    bb_std = df["close"].rolling(window=period).std().shift(1)
    bb_std_safe = bb_std.replace(0, np.nan)
    close_lagged = df["close"].shift(1)

    bb_upper = bb_middle + (std_mult * bb_std)
    bb_lower = bb_middle - (std_mult * bb_std)
    band_range_safe = (bb_upper - bb_lower).replace(0, np.nan)

    # Batch concat to avoid fragmentation
    new_cols = {
        "bb_middle": bb_middle,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_width": (bb_upper - bb_lower) / bb_std_safe,
        "bb_position": (close_lagged - bb_lower) / band_range_safe,
        "close_bb_zscore": (close_lagged - bb_middle) / bb_std_safe,
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    feature_metadata["bb_middle"] = f"Bollinger Band middle ({period},{std_mult}, lagged)"
    feature_metadata["bb_upper"] = f"Bollinger Band upper ({period},{std_mult}, lagged)"
    feature_metadata["bb_lower"] = f"Bollinger Band lower ({period},{std_mult}, lagged)"
    feature_metadata["bb_width"] = "Bollinger Band width (normalized by std, lagged)"
    feature_metadata["bb_position"] = "Price position in Bollinger Bands [0,1] (lagged)"
    feature_metadata["close_bb_zscore"] = "Close price z-score relative to BB middle (lagged)"

    return df


def add_keltner_channels(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, atr_mult: float = 2.0
) -> pd.DataFrame:
    """
    Add Keltner Channels features.

    Calculates Keltner Channels with configurable period and ATR multiplier
    and price position within the channels.

    All features use safe division and stationary representations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Keltner Channel period
    atr_mult : float, default 2.0
        ATR multiplier

    Returns
    -------
    pd.DataFrame
        DataFrame with Keltner Channel features added
    """
    logger.info(f"Adding Keltner Channels with period: {period}")

    ema_raw = calculate_ema_numba(df["close"].values, period)
    atr_raw = calculate_atr_numba(df["high"].values, df["low"].values, df["close"].values, period)

    # ANTI-LOOKAHEAD: shift(1) ensures KC at bar[t] uses data up to bar[t-1]
    ema = pd.Series(ema_raw).shift(1).values
    atr = pd.Series(atr_raw).shift(1).values
    close_lagged = df["close"].shift(1).values

    kc_upper = ema + (atr_mult * atr)
    kc_lower = ema - (atr_mult * atr)
    channel_range = kc_upper - kc_lower
    channel_range_safe = np.where(channel_range == 0, np.nan, channel_range)
    atr_safe = np.where(atr == 0, np.nan, atr)

    # Batch concat to avoid fragmentation
    new_cols = {
        "kc_middle": ema,
        "kc_upper": kc_upper,
        "kc_lower": kc_lower,
        "kc_position": (close_lagged - kc_lower) / channel_range_safe,
        "close_kc_atr_dev": (close_lagged - ema) / atr_safe,
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    feature_metadata["kc_middle"] = f"Keltner Channel middle ({period},{atr_mult}, lagged)"
    feature_metadata["kc_upper"] = f"Keltner Channel upper ({period},{atr_mult}, lagged)"
    feature_metadata["kc_lower"] = f"Keltner Channel lower ({period},{atr_mult}, lagged)"
    feature_metadata["kc_position"] = "Price position in Keltner Channels [0,1] (lagged)"
    feature_metadata["close_kc_atr_dev"] = "Close deviation from KC middle in ATR units (lagged)"

    return df


def add_historical_volatility(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    periods: list[int] | None = None,
    timeframe: str = "5min",
) -> pd.DataFrame:
    """
    Add historical volatility features.

    Calculates annualized historical volatility for multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of volatility periods. Default: [10, 20, 60]
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with historical volatility features added
    """
    if periods is None:
        periods = [10, 20, 60]

    logger.info(f"Adding historical volatility with periods: {periods}")

    log_returns = np.log(df["close"] / df["close"].shift(1))
    annualization_factor = get_annualization_factor(timeframe)

    for period in periods:
        # ANTI-LOOKAHEAD: shift(1) ensures hvol at bar[t] uses data up to bar[t-1]
        df[f"hvol_{period}"] = (
            log_returns.rolling(window=period).std() * annualization_factor
        ).shift(1)

        feature_metadata[f"hvol_{period}"] = f"Historical volatility ({period}, lagged)"

    return df


def add_parkinson_volatility(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Add Parkinson volatility.

    Parkinson volatility uses high-low range to estimate volatility,
    which is more efficient than close-to-close volatility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'high' and 'low' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Parkinson volatility period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Parkinson volatility added
    """
    logger.info(f"Adding Parkinson volatility with period: {period}")

    hl_ratio = np.log(df["high"] / df["low"])
    annualization_factor = get_annualization_factor(timeframe)
    # ANTI-LOOKAHEAD: shift(1) ensures parkinson_vol at bar[t] uses data up to bar[t-1]
    parkinson_raw = (
        np.sqrt((1 / (4 * np.log(2))) * (hl_ratio**2).rolling(window=period).mean())
        * annualization_factor
    )
    df["parkinson_vol"] = parkinson_raw.shift(1)

    feature_metadata["parkinson_vol"] = f"Parkinson volatility ({period}, lagged)"

    return df


def add_garman_klass_volatility(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Add Garman-Klass volatility.

    Garman-Klass volatility uses OHLC data for more efficient volatility estimation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Garman-Klass volatility period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Garman-Klass volatility added
    """
    logger.info(f"Adding Garman-Klass volatility with period: {period}")

    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    annualization_factor = get_annualization_factor(timeframe)

    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    # ANTI-LOOKAHEAD: shift(1) ensures gk_vol at bar[t] uses data up to bar[t-1]
    # np.maximum(..., 0) prevents sqrt of negative values from numerical precision issues
    df["gk_vol"] = (
        np.sqrt(np.maximum(gk.rolling(window=period).mean(), 0)) * annualization_factor
    ).shift(1)

    feature_metadata["gk_vol"] = f"Garman-Klass volatility ({period}, lagged)"

    return df


def add_higher_moments(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add rolling skewness and kurtosis of returns.

    Higher moments measure distribution shape:
    - Skewness: asymmetry (negative = fat left tail, crash risk)
    - Kurtosis: tail fatness (high = more extreme moves)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of rolling periods. Default: [20, 60]

    Returns
    -------
    pd.DataFrame
        DataFrame with skewness and kurtosis features added
    """
    if periods is None:
        periods = [20, 60]

    logger.info(f"Adding higher moments (skewness/kurtosis) with periods: {periods}")

    returns = df["close"].pct_change()

    for period in periods:
        # Skewness - ANTI-LOOKAHEAD: shift(1)
        skew_col = f"return_skew_{period}"
        df[skew_col] = returns.rolling(period).skew().shift(1)
        feature_metadata[skew_col] = f"Return skewness ({period}-period, lagged)"

        # Kurtosis (excess) - ANTI-LOOKAHEAD: shift(1)
        kurt_col = f"return_kurt_{period}"
        df[kurt_col] = returns.rolling(period).kurt().shift(1)
        feature_metadata[kurt_col] = f"Return excess kurtosis ({period}-period, lagged)"

    return df


def add_rogers_satchell_volatility(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Add Rogers-Satchell volatility.

    Rogers-Satchell volatility handles non-zero drift (trending markets) better
    than close-to-close volatility. It uses the relationship between OHLC prices
    and does not assume the mean return is zero.

    Formula:
        RS = sqrt(mean((H-C)(H-O) + (L-C)(L-O)))

    Where H, L, O, C are log prices (high, low, open, close).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Rolling window period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Rogers-Satchell volatility added
    """
    logger.info(f"Adding Rogers-Satchell volatility with period: {period}")

    annualization_factor = get_annualization_factor(timeframe)

    # Log prices for the calculation
    log_high = np.log(df["high"])
    log_low = np.log(df["low"])
    log_open = np.log(df["open"])
    log_close = np.log(df["close"])

    # Rogers-Satchell formula: (H-C)(H-O) + (L-C)(L-O)
    rs_component = (log_high - log_close) * (log_high - log_open) + (log_low - log_close) * (
        log_low - log_open
    )

    # Rolling mean and sqrt for volatility
    # np.maximum(..., 0) prevents sqrt of negative values from numerical precision issues
    rs_vol_raw = (
        np.sqrt(np.maximum(rs_component.rolling(window=period).mean(), 0)) * annualization_factor
    )

    # ANTI-LOOKAHEAD: shift(1) ensures rs_vol at bar[t] uses data up to bar[t-1]
    df["rs_vol"] = rs_vol_raw.shift(1)

    feature_metadata["rs_vol"] = f"Rogers-Satchell volatility ({period}, lagged)"

    return df


def add_yang_zhang_volatility(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Add Yang-Zhang volatility.

    Yang-Zhang volatility is a comprehensive estimator that handles both
    drift (trending markets) and opening jumps (overnight gaps). It combines
    overnight volatility, open-to-close volatility, and Rogers-Satchell volatility.

    Formula:
        YZ = sqrt(vol_overnight^2 + k*vol_open_close^2 + (1-k)*vol_rogers_satchell^2)

    Where k = 0.34 / (1.34 + (n+1)/(n-1)) is an optimal weighting factor.

    This is considered one of the most efficient volatility estimators for
    financial data with overnight gaps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Rolling window period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Yang-Zhang volatility added
    """
    logger.info(f"Adding Yang-Zhang volatility with period: {period}")

    annualization_factor = get_annualization_factor(timeframe)
    n = period

    # Optimal weighting factor k
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Log prices
    log_open = np.log(df["open"])
    log_close = np.log(df["close"])
    log_high = np.log(df["high"])
    log_low = np.log(df["low"])

    # Overnight return (close to open gap)
    log_close_prev = log_close.shift(1)
    overnight_return = log_open - log_close_prev

    # Open-to-close return
    open_close_return = log_close - log_open

    # Variance components
    # 1. Overnight variance
    overnight_mean = overnight_return.rolling(window=period).mean()
    overnight_var = ((overnight_return - overnight_mean) ** 2).rolling(window=period).mean()

    # 2. Open-to-close variance
    oc_mean = open_close_return.rolling(window=period).mean()
    open_close_var = ((open_close_return - oc_mean) ** 2).rolling(window=period).mean()

    # 3. Rogers-Satchell variance (drift-independent intraday variance)
    rs_component = (log_high - log_close) * (log_high - log_open) + (log_low - log_close) * (
        log_low - log_open
    )
    rs_var = rs_component.rolling(window=period).mean()

    # Yang-Zhang volatility: combines all three components
    yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
    # np.maximum(..., 0) prevents sqrt of negative values from numerical precision issues
    yz_vol_raw = np.sqrt(np.maximum(yz_var, 0)) * annualization_factor

    # ANTI-LOOKAHEAD: shift(1) ensures yz_vol at bar[t] uses data up to bar[t-1]
    df["yz_vol"] = yz_vol_raw.shift(1)

    feature_metadata["yz_vol"] = f"Yang-Zhang volatility ({period}, lagged)"

    return df


# =============================================================================
# GARCH VOLATILITY FORECASTING
# =============================================================================

# Try to import arch library for GARCH models
try:
    from arch import arch_model  # type: ignore[import-not-found]

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    arch_model = None


def _fit_garch_rolling(
    returns: np.ndarray,
    window: int,
    p: int = 1,
    q: int = 1,
    forecast_horizon: int = 1,
    refit_interval: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit rolling GARCH(p,q) models and generate volatility forecasts.

    Uses expanding window for more stable estimation, with minimum 100 observations.
    Optimized to fit every `refit_interval` bars instead of every bar for ~10-50x speedup.

    Parameters
    ----------
    returns : np.ndarray
        Array of log returns (scaled to percentage for numerical stability)
    window : int
        Minimum window size for initial estimation
    p : int, default 1
        GARCH lag order for conditional variance
    q : int, default 1
        ARCH lag order for squared residuals
    forecast_horizon : int, default 1
        Number of steps ahead to forecast
    refit_interval : int, default 20
        Fit GARCH model every N bars (Phase 28-3 optimization).
        Set to 1 for original behavior (fit every bar).
        Values 10-50 recommended for balance of speed vs accuracy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (1-step forecasts, h-step forecasts)
    """
    n = len(returns)
    vol_forecast_1 = np.full(n, np.nan)
    vol_forecast_h = np.full(n, np.nan)

    # Minimum observations for stable GARCH estimation
    min_obs = max(100, window)

    # Track last successful fit for forward-fill
    last_vol_1 = np.nan
    last_vol_h = np.nan

    # Use expanding window for more stable estimates
    # Phase 28-3: Fit every refit_interval bars instead of every bar
    for i in range(min_obs, n):
        # Only refit at intervals (or first valid point)
        should_fit = (i == min_obs) or ((i - min_obs) % refit_interval == 0)

        if should_fit:
            try:
                # Use data up to but not including current bar (anti-lookahead)
                sample = returns[:i]

                # Skip if too many NaNs
                valid_sample = sample[~np.isnan(sample)]
                if len(valid_sample) < min_obs:
                    # Forward-fill from last successful fit
                    vol_forecast_1[i] = last_vol_1
                    vol_forecast_h[i] = last_vol_h
                    continue

                # Fit GARCH model with minimal output
                model = arch_model(
                    valid_sample * 100,  # Scale for numerical stability
                    vol="Garch",
                    p=p,
                    q=q,
                    rescale=False,
                )

                # Suppress convergence warnings
                with np.errstate(all="ignore"):
                    result = model.fit(disp="off", show_warning=False)

                # Generate forecasts
                forecast = result.forecast(horizon=max(1, forecast_horizon))

                # 1-step ahead variance forecast (convert back from percentage)
                last_vol_1 = np.sqrt(forecast.variance.values[-1, 0]) / 100
                vol_forecast_1[i] = last_vol_1

                # h-step ahead forecast
                if forecast_horizon > 1:
                    last_vol_h = np.sqrt(forecast.variance.values[-1, -1]) / 100
                else:
                    last_vol_h = last_vol_1
                vol_forecast_h[i] = last_vol_h

            except Exception as e:
                # GARCH estimation can fail for various reasons (non-convergence, etc.)
                # Forward-fill from last successful fit
                logger.warning(f"GARCH forecast failed at index {i}: {e}. Using forward-fill.")
                vol_forecast_1[i] = last_vol_1
                vol_forecast_h[i] = last_vol_h
        else:
            # Between refit intervals: forward-fill from last successful fit
            vol_forecast_1[i] = last_vol_1
            vol_forecast_h[i] = last_vol_h

    return vol_forecast_1, vol_forecast_h


def add_garch_features(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    p: int = 1,
    q: int = 1,
    forecast_horizon: int = 5,
    window: int = 100,
    timeframe: str = "5min",
    refit_interval: int = 20,
) -> pd.DataFrame:
    """
    Add GARCH(p,q) volatility forecast features.

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
    capture volatility clustering - the tendency of large price moves to be
    followed by large moves. This is valuable for:
    - Forecasting future volatility
    - Risk management and position sizing
    - Option pricing and hedging
    - Regime detection (high vs low volatility states)

    Academic Reference:
        Bollerslev, T. (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
        Journal of Econometrics, 31(3), 307-327.

    Features added:
    - garch_vol_forecast: 1-step ahead volatility forecast
    - garch_vol_forecast_{h}: h-step ahead volatility forecast
    - garch_vol_ratio: Ratio of forecast to realized volatility
    - garch_vol_zscore: Z-score of forecast relative to rolling mean

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : dict[str, str]
        Dictionary to store feature descriptions
    p : int, default 1
        GARCH lag order for conditional variance
    q : int, default 1
        ARCH lag order for squared residuals
    forecast_horizon : int, default 5
        Number of steps ahead for longer-term forecast
    window : int, default 100
        Minimum window for GARCH estimation
    timeframe : str, default '5min'
        Bar timeframe for annualization
    refit_interval : int, default 20
        Fit GARCH model every N bars (Phase 28-3 optimization).
        Set to 1 for original behavior (fit every bar).
        Values 10-50 recommended for balance of speed vs accuracy.
        Default 20 provides ~10-20x speedup with minimal accuracy loss.

    Returns
    -------
    pd.DataFrame
        DataFrame with GARCH features added

    Notes
    -----
    - Requires the `arch` library: pip install arch
    - If arch is not available, logs a warning and returns df unchanged
    - Optimized with refit_interval for ~10-20x speedup (Phase 28-3)
    - Anti-lookahead: All forecasts use data strictly before current bar
    """
    if not ARCH_AVAILABLE:
        logger.warning(
            "GARCH features skipped: 'arch' library not installed. "
            "Install with: pip install arch"
        )
        return df

    logger.info(
        f"Adding GARCH({p},{q}) volatility features with forecast horizon: {forecast_horizon}"
    )

    # Calculate log returns
    log_returns = np.log(df["close"] / df["close"].shift(1)).values

    # Get annualization factor
    ann_factor = get_annualization_factor(timeframe)

    # Fit rolling GARCH and get forecasts (optimized with refit_interval)
    vol_1, vol_h = _fit_garch_rolling(
        log_returns,
        window=window,
        p=p,
        q=q,
        forecast_horizon=forecast_horizon,
        refit_interval=refit_interval,
    )

    # 1-step ahead forecast (already anti-lookahead from rolling estimation)
    df["garch_vol_forecast"] = vol_1 * ann_factor
    feature_metadata["garch_vol_forecast"] = (
        f"GARCH({p},{q}) 1-step volatility forecast (annualized, lagged)"
    )

    # h-step ahead forecast
    if forecast_horizon > 1:
        col_h = f"garch_vol_forecast_{forecast_horizon}"
        df[col_h] = vol_h * ann_factor
        feature_metadata[col_h] = (
            f"GARCH({p},{q}) {forecast_horizon}-step volatility forecast (annualized, lagged)"
        )

    # Ratio of forecast to realized volatility
    # Use rolling realized vol from previous bars
    realized_vol = (
        np.log(df["close"] / df["close"].shift(1))
        .rolling(window=20)
        .std()
        .shift(1)  # ANTI-LOOKAHEAD
        * ann_factor
    )
    df["garch_vol_ratio"] = df["garch_vol_forecast"] / realized_vol.replace(0, np.nan)
    feature_metadata["garch_vol_ratio"] = "GARCH forecast / realized volatility ratio (lagged)"

    # Z-score of forecast relative to its rolling mean
    forecast_mean = df["garch_vol_forecast"].rolling(window=50).mean()
    forecast_std = df["garch_vol_forecast"].rolling(window=50).std()
    df["garch_vol_zscore"] = (df["garch_vol_forecast"] - forecast_mean) / forecast_std.replace(
        0, np.nan
    )
    feature_metadata["garch_vol_zscore"] = "GARCH forecast z-score (50-period rolling, lagged)"

    return df


__all__ = [
    "add_atr",
    "add_bollinger_bands",
    "add_keltner_channels",
    "add_historical_volatility",
    "add_parkinson_volatility",
    "add_garman_klass_volatility",
    "add_higher_moments",
    "add_rogers_satchell_volatility",
    "add_yang_zhang_volatility",
    "add_garch_features",
    "ARCH_AVAILABLE",
]
