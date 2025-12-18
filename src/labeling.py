"""
Triple-Barrier Labeling Module for Ensemble Trading System
Implements ATR-based barrier labeling with quality scoring
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict
from numba import njit, prange

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    barrier_params: Tuple[float, float, int]
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to dataframe.

    Args:
        df: DataFrame with OHLC and ATR
        horizon: Lookahead horizon (1, 5, or 20 bars)
        barrier_params: (k_up, k_down, max_bars) tuple
    """
    k_up, k_down, max_bars = barrier_params

    logger.info(f"Applying triple-barrier labeling for horizon {horizon}")
    logger.info(f"  k_up={k_up}, k_down={k_down}, max_bars={max_bars}")

    # Get ATR column
    atr_col = f'atr_{max(7, min(horizon * 2, 21))}'
    if atr_col not in df.columns:
        atr_col = 'atr_14'

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

    # Log label distribution
    label_counts = pd.Series(labels).value_counts().sort_index()
    logger.info(f"  Label distribution for h{horizon}:")
    for label, count in label_counts.items():
        label_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}.get(label, str(label))
        logger.info(f"    {label_name}: {count:,} ({count/len(labels)*100:.1f}%)")

    return df


def label_symbol(
    input_path: Path,
    output_path: Path,
    symbol: str,
    barrier_params: Dict[int, Tuple[float, float, int]]
) -> pd.DataFrame:
    """Label a single symbol's data."""
    logger.info(f"="*60)
    logger.info(f"Labeling {symbol}")
    logger.info(f"="*60)

    # Load feature data
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # Apply labeling for each horizon
    for horizon, params in barrier_params.items():
        df = apply_triple_barrier(df, horizon, params)

    # Save labeled data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved labeled data to {output_path}")

    return df


def main():
    """Run labeling for all symbols."""
    from config import FEATURES_DIR, FINAL_DATA_DIR, SYMBOLS, LOOKBACK_HORIZONS

    # Default barrier parameters: (k_up, k_down, max_bars)
    barrier_params = {
        1: (1.0, 1.0, 1),
        5: (1.5, 1.0, 5),
        20: (2.0, 1.5, 20)
    }

    for symbol in SYMBOLS:
        input_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"
        output_path = FINAL_DATA_DIR / f"{symbol}_labeled.parquet"

        if input_path.exists():
            label_symbol(input_path, output_path, symbol, barrier_params)
        else:
            logger.warning(f"No feature data found for {symbol}")


if __name__ == "__main__":
    main()
