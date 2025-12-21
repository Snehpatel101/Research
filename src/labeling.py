"""
Triple-Barrier Labeling Module for Ensemble Trading System
Implements ATR-based barrier labeling with quality scoring

CRITICAL FIX (2024): Barrier multipliers calibrated for balanced class distribution.
Previous values were too wide, causing 99%+ neutral labels.
New values target ~35% long, ~35% short, ~30% neutral distribution.

DEPRECATION WARNING (2025-12-21):
This module is DEPRECATED and will be removed in a future release.
Use src/stages/stage4_labeling.py instead, which provides:
- Better feature output (includes MFE and touch_type)
- Symbol-specific barrier parameters via get_barrier_params()
- Consolidated configuration from config.py
- Forward compatibility with pipeline infrastructure

Migration path:
  OLD: from labeling import main, apply_triple_barrier, label_symbol
  NEW: from stages.stage4_labeling import main, apply_triple_barrier, process_symbol_labeling
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
from typing import Tuple, Dict, Optional
from numba import njit, prange

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Issue deprecation warning at module import
warnings.warn(
    "The labeling module is deprecated. Use stages.stage4_labeling instead. "
    "This module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2
)


# =============================================================================
# BARRIER CONFIGURATION - CALIBRATED FOR BALANCED LABELS
# =============================================================================
# Try to import from config, fall back to local defaults if not available

try:
    from config import BARRIER_PARAMS, PERCENTAGE_BARRIER_PARAMS
    logger.info("Loaded BARRIER_PARAMS from config.py")
except ImportError:
    logger.info("Using local BARRIER_PARAMS (config.py not available)")

    # Local fallback - EMPIRICALLY CALIBRATED for 30-35% neutral label rate
    # These parameters were derived from statistical analysis of actual data.
    # Previous values produced only 1-5% neutral labels (too many directional signals).
    #
    # Key insights from empirical calibration:
    # - H1 is NON-VIABLE after transaction costs (excluded from ACTIVE_HORIZONS)
    # - H5 requires k=0.90 for proper neutral rate
    # - H20 requires k=2.00 for proper neutral rate
    # - max_bars for H20 (60) determines required PURGE_BARS to prevent leakage

    BARRIER_PARAMS: Dict[int, Dict] = {
        # Horizon 1: Ultra-short (5 min) - NON-VIABLE after transaction costs
        # Transaction costs (~0.5 ticks) exceed typical H1 profit (1-2 ticks).
        # Kept for labeling completeness only.
        1: {
            'k_up': 0.25,
            'k_down': 0.25,
            'max_bars': 5,
            'description': 'Ultra-short: NON-VIABLE after transaction costs'
        },
        # Horizon 5: Short-term (25 min) - ACTIVE
        # Empirically calibrated: k=0.90 produces ~30-35% neutral labels.
        5: {
            'k_up': 0.90,
            'k_down': 0.90,
            'max_bars': 15,
            'description': 'Short-term: empirically calibrated for 30-35% neutral'
        },
        # Horizon 20: Medium-term (~1.5 hours) - ACTIVE
        # Empirically calibrated: k=2.0 produces ~30-35% neutral labels.
        # CRITICAL: max_bars=60 determines PURGE_BARS requirement.
        20: {
            'k_up': 2.00,
            'k_down': 2.00,
            'max_bars': 60,
            'description': 'Medium-term: empirically calibrated for 30-35% neutral'
        }
    }

    # Alternative: Percentage-based barriers (ATR-independent fallback)
    PERCENTAGE_BARRIER_PARAMS: Dict[int, Dict] = {
        1: {'pct_up': 0.0015, 'pct_down': 0.0015, 'max_bars': 5},   # 0.15%
        5: {'pct_up': 0.0030, 'pct_down': 0.0030, 'max_bars': 15},  # 0.30%
        20: {'pct_up': 0.0075, 'pct_down': 0.0075, 'max_bars': 60}  # 0.75%
    }


@njit(parallel=True)
def apply_triple_barrier_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    atr: np.ndarray,
    k_up: float,
    k_down: float,
    max_bars: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized triple barrier labeling.

    Parameters:
    -----------
    close : array of close prices
    high : array of high prices
    low : array of low prices
    open_prices : array of open prices (used to resolve simultaneous barrier hits)
    atr : array of ATR values
    k_up : profit barrier multiplier
    k_down : stop barrier multiplier
    max_bars : maximum bars to hold before timeout

    Returns:
        labels: -1 (short), 0 (neutral), 1 (long)
        bars_to_hit: number of bars until barrier hit
        mae: maximum adverse excursion
        quality: sample quality score (0-1)

    Note:
    -----
    When both upper and lower barriers are hit on the same bar, we use distance
    from the bar's open price to determine which barrier was likely hit first.
    This follows Lopez de Prado's methodology and eliminates long bias from
    always checking upper barrier first.
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int32)
    bars_to_hit = np.zeros(n, dtype=np.int32)
    mae = np.zeros(n, dtype=np.float64)
    quality = np.zeros(n, dtype=np.float64)

    for i in prange(n - max_bars):
        entry_price = close[i]
        current_atr = atr[i]

        if current_atr <= 0 or np.isnan(current_atr):
            labels[i] = 0
            bars_to_hit[i] = max_bars
            continue

        upper_barrier = entry_price + k_up * current_atr
        lower_barrier = entry_price - k_down * current_atr

        max_adverse = 0.0
        hit_bar = max_bars
        hit_label = 0

        for j in range(1, max_bars + 1):
            idx = i + j
            if idx >= n:
                break

            bar_high = high[idx]
            bar_low = low[idx]
            bar_open = open_prices[idx]

            # Check barrier hits
            upper_hit = bar_high >= upper_barrier
            lower_hit = bar_low <= lower_barrier

            if upper_hit and lower_hit:
                # BOTH barriers hit on same bar - determine which was hit first
                # Use distance from bar open as proxy for which barrier hit first
                # The barrier closer to the open price was likely hit first
                dist_to_upper = abs(bar_open - upper_barrier)
                dist_to_lower = abs(bar_open - lower_barrier)

                hit_bar = j
                if dist_to_upper <= dist_to_lower:
                    # Upper barrier was closer to open, likely hit first
                    hit_label = 1
                else:
                    # Lower barrier was closer to open, likely hit first
                    hit_label = -1
                break
            elif upper_hit:
                # Only upper barrier hit
                hit_bar = j
                hit_label = 1
                break
            elif lower_hit:
                # Only lower barrier hit
                hit_bar = j
                hit_label = -1
                break

            # Track MAE (only if not hit yet)
            adverse = max(
                (entry_price - bar_low) / entry_price,
                (bar_high - entry_price) / entry_price
            )
            if adverse > max_adverse:
                max_adverse = adverse

        labels[i] = hit_label
        bars_to_hit[i] = hit_bar
        mae[i] = max_adverse

        # Quality score: faster hits with lower MAE are higher quality
        if hit_label != 0:
            speed_score = 1.0 - (hit_bar / max_bars)
            mae_score = 1.0 - min(max_adverse / (k_up * current_atr / entry_price), 1.0)
            quality[i] = 0.5 * speed_score + 0.5 * mae_score
        else:
            quality[i] = 0.3  # Neutral samples get moderate quality

    return labels, bars_to_hit, mae, quality


def apply_triple_barrier(
    df: pd.DataFrame,
    horizon: int,
    barrier_params: Optional[Tuple[float, float, int]] = None,
    atr_column: str = 'atr_14'
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to dataframe.

    Args:
        df: DataFrame with OHLC and ATR
        horizon: Lookahead horizon (1, 5, or 20 bars)
        barrier_params: Optional (k_up, k_down, max_bars) tuple.
                       If None, uses calibrated BARRIER_PARAMS.
        atr_column: Name of ATR column to use
    """
    # Use calibrated defaults if not provided
    if barrier_params is None:
        if horizon in BARRIER_PARAMS:
            params = BARRIER_PARAMS[horizon]
            k_up = params['k_up']
            k_down = params['k_down']
            max_bars = params['max_bars']
        else:
            # Fallback for non-standard horizons
            k_up = 0.5
            k_down = 0.5
            max_bars = max(horizon * 3, 10)
    else:
        k_up, k_down, max_bars = barrier_params

    logger.info(f"Applying triple-barrier labeling for horizon {horizon}")
    logger.info(f"  k_up={k_up:.3f}, k_down={k_down:.3f}, max_bars={max_bars}")
    logger.info(f"  Symmetric barriers: {k_up == k_down}")

    # Get ATR column - prefer specified, fallback to horizon-based, then atr_14
    if atr_column in df.columns:
        atr_col = atr_column
    else:
        atr_col = f'atr_{max(7, min(horizon * 2, 21))}'
        if atr_col not in df.columns:
            atr_col = 'atr_14'
    logger.info(f"  Using ATR column: {atr_col}")

    labels, bars_to_hit, mae, quality = apply_triple_barrier_numba(
        df['close'].values.astype(np.float64),
        df['high'].values.astype(np.float64),
        df['low'].values.astype(np.float64),
        df['open'].values.astype(np.float64),
        df[atr_col].values.astype(np.float64),
        k_up,
        k_down,
        max_bars
    )

    df[f'label_h{horizon}'] = labels
    df[f'bars_to_hit_h{horizon}'] = bars_to_hit
    df[f'mae_h{horizon}'] = mae
    df[f'quality_h{horizon}'] = quality

    # Compute sample weights based on quality
    df[f'sample_weight_h{horizon}'] = df[f'quality_h{horizon}'] ** 0.5

    # Log label distribution with balance check
    label_counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    logger.info(f"  Label distribution for h{horizon}:")
    for label in [-1, 0, 1]:
        count = label_counts.get(label, 0)
        pct = count / total * 100
        label_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}.get(label, str(label))
        logger.info(f"    {label_name}: {count:,} ({pct:.1f}%)")

    # Check for imbalance warning
    short_pct = label_counts.get(-1, 0) / total * 100
    long_pct = label_counts.get(1, 0) / total * 100
    neutral_pct = label_counts.get(0, 0) / total * 100

    if neutral_pct > 80:
        logger.warning(f"  WARNING: High neutral rate ({neutral_pct:.1f}%). Consider tightening barriers.")
    if abs(short_pct - long_pct) > 10:
        logger.warning(f"  WARNING: Asymmetric labels (short={short_pct:.1f}%, long={long_pct:.1f}%). Check barrier symmetry.")

    return df


def label_symbol(
    input_path: Path,
    output_path: Path,
    symbol: str,
    barrier_params: Optional[Dict[int, Tuple[float, float, int]]] = None,
    horizons: Optional[list] = None,
    atr_column: str = 'atr_14'
) -> pd.DataFrame:
    """
    Label a single symbol's data.

    Args:
        input_path: Path to feature parquet file
        output_path: Path to save labeled data
        symbol: Symbol name
        barrier_params: Optional dict mapping horizon -> (k_up, k_down, max_bars).
                       If None, uses calibrated BARRIER_PARAMS.
        horizons: List of horizons to process. If None, uses keys from barrier_params
                 or default [1, 5, 20].
        atr_column: ATR column to use for barrier calculation
    """
    logger.info("="*60)
    logger.info(f"Labeling {symbol}")
    logger.info("="*60)

    # Determine horizons to process
    if horizons is None:
        if barrier_params is not None:
            horizons = list(barrier_params.keys())
        else:
            horizons = [1, 5, 20]

    # Log parameters being used
    logger.info("Using CALIBRATED barrier parameters:")
    for h in horizons:
        if barrier_params and h in barrier_params:
            k_up, k_down, max_bars = barrier_params[h]
        elif h in BARRIER_PARAMS:
            k_up = BARRIER_PARAMS[h]['k_up']
            k_down = BARRIER_PARAMS[h]['k_down']
            max_bars = BARRIER_PARAMS[h]['max_bars']
        else:
            k_up, k_down, max_bars = 0.5, 0.5, max(h * 3, 10)
        logger.info(f"  H{h}: k_up={k_up:.3f}, k_down={k_down:.3f}, max_bars={max_bars}")

    # Load feature data
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # Apply labeling for each horizon
    for horizon in horizons:
        params = barrier_params.get(horizon) if barrier_params else None
        df = apply_triple_barrier(df, horizon, barrier_params=params, atr_column=atr_column)

    # Print summary
    logger.info("")
    logger.info("="*60)
    logger.info("LABEL DISTRIBUTION SUMMARY")
    logger.info("="*60)
    for h in horizons:
        label_col = f'label_h{h}'
        if label_col in df.columns:
            counts = df[label_col].value_counts().sort_index()
            total = len(df)
            short_pct = counts.get(-1, 0) / total * 100
            neutral_pct = counts.get(0, 0) / total * 100
            long_pct = counts.get(1, 0) / total * 100
            logger.info(f"H{h}: Short={short_pct:.1f}% | Neutral={neutral_pct:.1f}% | Long={long_pct:.1f}%")

    # Save labeled data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved labeled data to {output_path}")

    return df


def main():
    """Run labeling for all symbols using calibrated barrier parameters."""
    from config import FEATURES_DIR, FINAL_DATA_DIR, SYMBOLS, LOOKBACK_HORIZONS

    logger.info("="*70)
    logger.info("TRIPLE-BARRIER LABELING (CALIBRATED FOR BALANCED CLASSES)")
    logger.info("="*70)
    logger.info("")
    logger.info("Using calibrated BARRIER_PARAMS:")
    for horizon, params in BARRIER_PARAMS.items():
        logger.info(f"  H{horizon}: k_up={params['k_up']:.3f}, k_down={params['k_down']:.3f}, max_bars={params['max_bars']}")
        logger.info(f"       {params.get('description', '')}")
    logger.info("")
    logger.info("Target distribution: ~35% short, ~30% neutral, ~35% long")
    logger.info("-"*70)
    logger.info("")

    for symbol in SYMBOLS:
        input_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"
        output_path = FINAL_DATA_DIR / f"{symbol}_labeled.parquet"

        if input_path.exists():
            # Use None to trigger calibrated defaults from BARRIER_PARAMS
            label_symbol(
                input_path, output_path, symbol,
                barrier_params=None,  # Uses BARRIER_PARAMS
                horizons=LOOKBACK_HORIZONS,
                atr_column='atr_14'
            )
        else:
            logger.warning(f"No feature data found for {symbol}")


if __name__ == "__main__":
    main()
