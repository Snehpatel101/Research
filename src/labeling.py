"""
Triple-Barrier Labeling Module for Ensemble Trading System
Implements ATR-based barrier labeling with quality scoring

CRITICAL FIX (2024): Barrier multipliers calibrated for balanced class distribution.
Previous values were too wide, causing 99%+ neutral labels.
New values target ~35% long, ~35% short, ~30% neutral distribution.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
from numba import njit, prange

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# BARRIER CONFIGURATION - CALIBRATED FOR BALANCED LABELS
# =============================================================================
# Try to import from config, fall back to local defaults if not available

try:
    from config import BARRIER_PARAMS, PERCENTAGE_BARRIER_PARAMS
    logger.info("Loaded BARRIER_PARAMS from config.py")
except ImportError:
    logger.info("Using local BARRIER_PARAMS (config.py not available)")

    # Local fallback - calibrated for balanced class distribution
    # Key insight: barriers must be narrow enough relative to max_bars for price
    # to realistically hit them. Previous k=1.0-2.0 with max_bars=horizon was
    # far too wide, resulting in 99%+ timeouts (neutral labels).
    #
    # These parameters are calibrated for 5-minute bars on futures (MES, MGC).
    # Adjust based on your instrument's volatility characteristics.

    BARRIER_PARAMS: Dict[int, Dict] = {
        # Horizon 1 (5 minutes): Ultra-short term
        # Barrier: 0.3 ATR, Window: 5 bars (25 min)
        # Rationale: Price can move ~0.3 ATR per bar on average
        1: {
            'k_up': 0.3,
            'k_down': 0.3,
            'max_bars': 5,
            'description': 'Ultra-short: tight barriers for quick signals'
        },
        # Horizon 5 (25 minutes): Short-term
        # Barrier: 0.5 ATR, Window: 15 bars (75 min)
        # Rationale: sqrt(5) * 0.3 ~= 0.67, using 0.5 for more signals
        5: {
            'k_up': 0.5,
            'k_down': 0.5,
            'max_bars': 15,
            'description': 'Short-term: moderate barriers, extended window'
        },
        # Horizon 20 (~1.5 hours): Medium-term
        # Barrier: 0.75 ATR, Window: 60 bars (5 hours)
        # Rationale: sqrt(20) * 0.3 ~= 1.34, using 0.75 for more signals
        20: {
            'k_up': 0.75,
            'k_down': 0.75,
            'max_bars': 60,
            'description': 'Medium-term: wider barriers, long window'
        }
    }

    # Alternative: Percentage-based barriers (instrument-agnostic)
    PERCENTAGE_BARRIER_PARAMS: Dict[int, Dict] = {
        1: {'pct_up': 0.0015, 'pct_down': 0.0015, 'max_bars': 5},   # 0.15%
        5: {'pct_up': 0.0025, 'pct_down': 0.0025, 'max_bars': 15},  # 0.25%
        20: {'pct_up': 0.0050, 'pct_down': 0.0050, 'max_bars': 60}  # 0.50%
    }


@njit(parallel=True)
def apply_triple_barrier_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    k_up: float,
    k_down: float,
    max_bars: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized triple barrier labeling.

    Returns:
        labels: -1 (short), 0 (neutral), 1 (long)
        bars_to_hit: number of bars until barrier hit
        mae: maximum adverse excursion
        quality: sample quality score (0-1)
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

            # Check high for upper barrier hit
            if high[idx] >= upper_barrier:
                hit_bar = j
                hit_label = 1
                break

            # Check low for lower barrier hit
            if low[idx] <= lower_barrier:
                hit_bar = j
                hit_label = -1
                break

            # Track MAE
            if hit_label == 0:  # Not hit yet
                adverse = max(
                    (entry_price - low[idx]) / entry_price,
                    (high[idx] - entry_price) / entry_price
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
