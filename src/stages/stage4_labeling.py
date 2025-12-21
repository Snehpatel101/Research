"""
Stage 4: Triple-Barrier Labeling with Numba Optimization
Generates initial labels using triple barrier method with ATR-based dynamic barriers

CRITICAL FIX (2024-12): ASYMMETRIC BARRIERS TO CORRECT LONG BIAS
Previous symmetric barriers (k_up = k_down) in a historically bullish market
produced 87-91% long signals. New asymmetric barriers (k_up > k_down) make the
lower barrier easier to hit, targeting ~50/50 long/short distribution.
"""
import pandas as pd
import numpy as np
import numba as nb
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Configure logging - use NullHandler to avoid duplicate logs when imported as module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# BARRIER CONFIGURATION - IMPORTED FROM CENTRAL CONFIG
# =============================================================================
# All barrier parameters are now defined in src/config.py as the single source of truth.
# This ensures consistency across all pipeline stages and eliminates configuration drift.

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BARRIER_PARAMS,
    BARRIER_PARAMS_DEFAULT,
    PERCENTAGE_BARRIER_PARAMS,
    get_barrier_params
)
logger.info("Loaded barrier parameters from config.py (single source of truth)")


@nb.jit(nopython=True, cache=True)
def triple_barrier_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    atr: np.ndarray,
    k_up: float,
    k_down: float,
    max_bars: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized triple barrier labeling.

    Parameters:
    -----------
    close : array of close prices
    high : array of high prices
    low : array of low prices
    open_prices : array of open prices (used to resolve simultaneous barrier hits)
    atr : array of ATR values
    k_up : profit barrier multiplier (e.g., 2.0 means 2*ATR above entry)
    k_down : stop barrier multiplier (e.g., 1.0 means 1*ATR below entry)
    max_bars : maximum bars to hold before timeout

    Returns:
    --------
    labels : +1 (long win), -1 (short loss), 0 (timeout)
    bars_to_hit : number of bars until barrier was hit
    mae : maximum adverse excursion (as % of entry price)
    mfe : maximum favorable excursion (as % of entry price)
    touch_type : 1 (upper), -1 (lower), 0 (timeout)

    Note:
    -----
    When both upper and lower barriers are hit on the same bar, we use distance
    from the bar's open price to determine which barrier was likely hit first.
    This follows Lopez de Prado's methodology and eliminates long bias from
    always checking upper barrier first.
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    bars_to_hit = np.zeros(n, dtype=np.int32)
    mae = np.zeros(n, dtype=np.float32)
    mfe = np.zeros(n, dtype=np.float32)
    touch_type = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        entry_price = close[i]
        entry_atr = atr[i]

        # Skip if ATR is invalid
        if np.isnan(entry_atr) or entry_atr <= 0:
            labels[i] = 0
            bars_to_hit[i] = max_bars
            continue

        # Define barriers
        upper_barrier = entry_price + k_up * entry_atr
        lower_barrier = entry_price - k_down * entry_atr

        # Track excursions
        max_adverse = 0.0
        max_favorable = 0.0

        # Scan forward
        hit = False
        for j in range(1, min(max_bars + 1, n - i)):
            idx = i + j
            bar_high = high[idx]
            bar_low = low[idx]
            bar_open = open_prices[idx]

            # Update excursions
            # For long position perspective
            upside = (bar_high - entry_price) / entry_price
            downside = (bar_low - entry_price) / entry_price

            if upside > max_favorable:
                max_favorable = upside
            if downside < max_adverse:
                max_adverse = downside

            # Check barrier hits
            upper_hit = bar_high >= upper_barrier
            lower_hit = bar_low <= lower_barrier

            if upper_hit and lower_hit:
                # BOTH barriers hit on same bar - determine which was hit first
                # Use distance from bar open as proxy for which barrier hit first
                # The barrier closer to the open price was likely hit first
                dist_to_upper = abs(bar_open - upper_barrier)
                dist_to_lower = abs(bar_open - lower_barrier)

                if dist_to_upper <= dist_to_lower:
                    # Upper barrier was closer to open, likely hit first
                    labels[i] = 1
                    touch_type[i] = 1
                else:
                    # Lower barrier was closer to open, likely hit first
                    labels[i] = -1
                    touch_type[i] = -1
                bars_to_hit[i] = j
                hit = True
                break
            elif upper_hit:
                # Only upper barrier hit (profit for long)
                labels[i] = 1
                bars_to_hit[i] = j
                touch_type[i] = 1
                hit = True
                break
            elif lower_hit:
                # Only lower barrier hit (stop for long)
                labels[i] = -1
                bars_to_hit[i] = j
                touch_type[i] = -1
                hit = True
                break

        # Timeout case
        if not hit:
            labels[i] = 0
            bars_to_hit[i] = max_bars
            touch_type[i] = 0

        mae[i] = max_adverse
        mfe[i] = max_favorable

    # Last bar always timeout
    labels[n-1] = 0
    bars_to_hit[n-1] = 0

    return labels, bars_to_hit, mae, mfe, touch_type


def apply_triple_barrier(
    df: pd.DataFrame,
    horizon: int,
    k_up: Optional[float] = None,
    k_down: Optional[float] = None,
    max_bars: Optional[int] = None,
    atr_column: str = 'atr_14',
    use_percentage: bool = False,
    pct_up: Optional[float] = None,
    pct_down: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply triple barrier labeling to a dataframe.

    Parameters:
    -----------
    df : DataFrame with OHLCV data and ATR
    horizon : horizon identifier (e.g., 1, 5, 20)
    k_up : profit barrier multiplier (ATR-based). If None, uses BARRIER_PARAMS
    k_down : stop barrier multiplier (ATR-based). If None, uses BARRIER_PARAMS
    max_bars : maximum bars. If None, uses BARRIER_PARAMS
    atr_column : column name for ATR values
    use_percentage : if True, use percentage barriers instead of ATR
    pct_up : percentage for upper barrier (e.g., 0.002 = 0.2%)
    pct_down : percentage for lower barrier

    Returns:
    --------
    df : DataFrame with additional columns:
        - label_h{horizon}
        - bars_to_hit_h{horizon}
        - mae_h{horizon}
        - mfe_h{horizon}
        - touch_type_h{horizon}

    Raises:
    -------
    ValueError : If DataFrame is empty or parameters are invalid
    KeyError : If required columns are missing
    """
    # === PARAMETER VALIDATION ===
    # Validate DataFrame is not empty
    if df.empty:
        raise ValueError("DataFrame is empty - cannot apply triple barrier labeling")

    # Validate horizon
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon} (type: {type(horizon).__name__})")

    # Validate k_up if provided
    if k_up is not None and k_up <= 0:
        raise ValueError(f"k_up must be positive, got {k_up}")

    # Validate k_down if provided
    if k_down is not None and k_down <= 0:
        raise ValueError(f"k_down must be positive, got {k_down}")

    # Validate max_bars if provided
    if max_bars is not None and max_bars <= 0:
        raise ValueError(f"max_bars must be positive, got {max_bars}")

    # Validate percentage parameters if use_percentage is True
    if use_percentage:
        if pct_up is not None and pct_up <= 0:
            raise ValueError(f"pct_up must be positive, got {pct_up}")
        if pct_down is not None and pct_down <= 0:
            raise ValueError(f"pct_down must be positive, got {pct_down}")

    # Validate required OHLC columns
    required_cols = {'close', 'high', 'low', 'open'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required OHLC columns: {sorted(missing)}")

    # Validate ATR column exists
    if atr_column not in df.columns:
        available_cols = [c for c in df.columns if 'atr' in c.lower()]
        raise KeyError(
            f"ATR column '{atr_column}' not found. "
            f"Available ATR-like columns: {available_cols if available_cols else 'none'}"
        )

    # Check for NaN values in critical columns and warn
    critical_cols = ['close', 'high', 'low', 'open', atr_column]
    nan_counts = df[critical_cols].isna().sum()
    if nan_counts.any():
        nan_report = nan_counts[nan_counts > 0].to_dict()
        logger.warning(f"NaN values in critical columns: {nan_report}")

    # Get default parameters from configuration if not provided
    # Note: This function doesn't know the symbol, so it uses BARRIER_PARAMS_DEFAULT
    # For symbol-specific params, call this with explicit k_up/k_down/max_bars
    if horizon in BARRIER_PARAMS_DEFAULT:
        defaults = BARRIER_PARAMS_DEFAULT[horizon]
        if k_up is None:
            k_up = defaults['k_up']
        if k_down is None:
            k_down = defaults['k_down']
        if max_bars is None:
            max_bars = defaults['max_bars']
    else:
        # Fallback for non-standard horizons
        if k_up is None:
            k_up = 1.0
        if k_down is None:
            k_down = 1.0
        if max_bars is None:
            max_bars = max(horizon * 3, 10)

    logger.info(f"Applying triple barrier for horizon {horizon}")
    logger.info(f"  k_up={k_up:.3f}, k_down={k_down:.3f}, max_bars={max_bars}")
    logger.info(f"  Mode: {'Percentage-based' if use_percentage else 'ATR-based'}")

    # Extract arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_prices = df['open'].values
    atr = df[atr_column].values

    # Apply numba function
    labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
        close, high, low, open_prices, atr, k_up, k_down, max_bars
    )

    # Add to dataframe
    df[f'label_h{horizon}'] = labels
    df[f'bars_to_hit_h{horizon}'] = bars_to_hit
    df[f'mae_h{horizon}'] = mae
    df[f'mfe_h{horizon}'] = mfe
    df[f'touch_type_h{horizon}'] = touch_type

    # Log statistics
    label_counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)

    logger.info(f"Label distribution for horizon {horizon}:")
    for label_val in [-1, 0, 1]:
        count = label_counts.get(label_val, 0)
        pct = count / total * 100
        label_name = {-1: "Short/Loss", 0: "Neutral/Timeout", 1: "Long/Win"}[label_val]
        logger.info(f"  {label_name:20s}: {count:6d} ({pct:5.1f}%)")

    avg_bars = bars_to_hit[bars_to_hit > 0].mean()
    logger.info(f"  Avg bars to hit: {avg_bars:.1f}")

    return df


def process_symbol_labeling(
    input_path: Path,
    output_path: Path,
    symbol: str,
    horizons: List[int] = [1, 5, 20],
    barrier_params: Optional[Dict] = None,
    atr_column: str = 'atr_14'
) -> pd.DataFrame:
    """
    Process labeling for a single symbol with multiple horizons.

    Parameters:
    -----------
    input_path : Path to features parquet file
    output_path : Path to save labeled data
    symbol : Symbol name
    horizons : List of horizons to label
    barrier_params : Dict mapping horizon -> {k_up, k_down, max_bars}
                     If None, uses symbol-specific params from config.py
    atr_column : column name for ATR values

    Returns:
    --------
    df : Labeled DataFrame
    """
    # Use symbol-specific parameters from config.py if not provided
    # This ensures MES gets asymmetric barriers and MGC gets symmetric barriers
    if barrier_params is None:
        barrier_params = {}
        for h in horizons:
            barrier_params[h] = get_barrier_params(symbol, h)

    logger.info("=" * 70)
    logger.info(f"Triple-Barrier Labeling: {symbol}")
    logger.info("=" * 70)
    logger.info(f"Using symbol-specific barrier parameters from config.py:")
    for h in horizons:
        if h in barrier_params:
            params = barrier_params[h]
            logger.info(f"  H{h}: k_up={params['k_up']:.2f}, k_down={params['k_down']:.2f}, "
                       f"max_bars={params['max_bars']} - {params.get('description', '')}")

    # Load features
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # Ensure ATR column exists
    if atr_column not in df.columns:
        raise ValueError(f"ATR column '{atr_column}' not found in dataframe")

    # Apply labeling for each horizon using symbol-specific parameters
    for horizon in tqdm(horizons, desc=f"Labeling {symbol}"):
        params = barrier_params[horizon]
        k_up = params['k_up']
        k_down = params['k_down']
        max_bars = params['max_bars']

        df = apply_triple_barrier(
            df, horizon, k_up=k_up, k_down=k_down, max_bars=max_bars, atr_column=atr_column
        )

    # Log final label distribution summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("LABEL DISTRIBUTION SUMMARY")
    logger.info("=" * 70)
    for horizon in horizons:
        label_col = f'label_h{horizon}'
        if label_col in df.columns:
            counts = df[label_col].value_counts().sort_index()
            total = len(df)
            short_pct = counts.get(-1, 0) / total * 100
            neutral_pct = counts.get(0, 0) / total * 100
            long_pct = counts.get(1, 0) / total * 100
            logger.info(f"H{horizon}: Short={short_pct:.1f}% | Neutral={neutral_pct:.1f}% | Long={long_pct:.1f}%")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved labeled data to {output_path}")
    logger.info("")

    return df


def main():
    """Run Stage 4: Initial labeling for all symbols."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import FEATURES_DIR, SYMBOLS, PROJECT_ROOT

    # Output directory
    labels_dir = PROJECT_ROOT / 'data' / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    horizons = [1, 5, 20]

    logger.info("=" * 70)
    logger.info("STAGE 4: TRIPLE-BARRIER LABELING (SYMBOL-SPECIFIC)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("SYMBOL-SPECIFIC BARRIER PARAMETERS from config.py:")
    logger.info("-" * 70)
    for symbol in SYMBOLS:
        logger.info(f"{symbol}:")
        for horizon in horizons:
            params = get_barrier_params(symbol, horizon)
            logger.info(f"  H{horizon:2d}: k_up={params['k_up']:.2f}, "
                       f"k_down={params['k_down']:.2f}, max_bars={params['max_bars']:2d}")
            logger.info(f"       {params.get('description', '')}")
    logger.info("-" * 70)
    logger.info("")

    for symbol in SYMBOLS:
        input_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"
        output_path = labels_dir / f"{symbol}_labels_init.parquet"

        if input_path.exists():
            # Pass None to use symbol-specific params from config.py
            process_symbol_labeling(
                input_path, output_path, symbol, horizons,
                barrier_params=None,  # Uses get_barrier_params(symbol, horizon)
                atr_column='atr_14'
            )
        else:
            logger.warning(f"No features file found for {symbol}: {input_path}")

    logger.info("=" * 70)
    logger.info("STAGE 4 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
