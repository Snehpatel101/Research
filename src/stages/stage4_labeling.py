"""
Stage 4: Triple-Barrier Labeling with Numba Optimization
Generates initial labels using triple barrier method with ATR-based dynamic barriers
"""
import pandas as pd
import numpy as np
import numba as nb
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True)
def triple_barrier_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
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

            # Update excursions
            # For long position perspective
            upside = (high[idx] - entry_price) / entry_price
            downside = (low[idx] - entry_price) / entry_price

            if upside > max_favorable:
                max_favorable = upside
            if downside < max_adverse:
                max_adverse = downside

            # Check upper barrier (profit for long)
            if high[idx] >= upper_barrier:
                labels[i] = 1
                bars_to_hit[i] = j
                touch_type[i] = 1
                hit = True
                break

            # Check lower barrier (stop for long)
            if low[idx] <= lower_barrier:
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
    k_up: float = 2.0,
    k_down: float = 1.0,
    max_bars: int = None,
    atr_column: str = 'atr_14'
) -> pd.DataFrame:
    """
    Apply triple barrier labeling to a dataframe.

    Parameters:
    -----------
    df : DataFrame with OHLCV data and ATR
    horizon : horizon identifier (e.g., 1, 5, 20)
    k_up : profit barrier multiplier
    k_down : stop barrier multiplier
    max_bars : maximum bars (defaults to horizon * 3)
    atr_column : column name for ATR values

    Returns:
    --------
    df : DataFrame with additional columns:
        - label_h{horizon}
        - bars_to_hit_h{horizon}
        - mae_h{horizon}
        - mfe_h{horizon}
        - touch_type_h{horizon}
    """
    if max_bars is None:
        max_bars = horizon * 3

    logger.info(f"Applying triple barrier for horizon {horizon}")
    logger.info(f"  k_up={k_up:.2f}, k_down={k_down:.2f}, max_bars={max_bars}")

    # Extract arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr = df[atr_column].values

    # Apply numba function
    labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
        close, high, low, atr, k_up, k_down, max_bars
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
    default_params: Dict = None
) -> pd.DataFrame:
    """
    Process labeling for a single symbol with multiple horizons.

    Parameters:
    -----------
    input_path : Path to features parquet file
    output_path : Path to save labeled data
    symbol : Symbol name
    horizons : List of horizons to label
    default_params : Default parameters dict with keys:
        - k_up: float (default 2.0)
        - k_down: float (default 1.0)
        - atr_column: str (default 'atr_14')

    Returns:
    --------
    df : Labeled DataFrame
    """
    if default_params is None:
        default_params = {
            'k_up': 2.0,
            'k_down': 1.0,
            'atr_column': 'atr_14'
        }

    logger.info("=" * 70)
    logger.info(f"Triple-Barrier Labeling: {symbol}")
    logger.info("=" * 70)

    # Load features
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # Ensure ATR column exists
    atr_col = default_params.get('atr_column', 'atr_14')
    if atr_col not in df.columns:
        raise ValueError(f"ATR column '{atr_col}' not found in dataframe")

    # Apply labeling for each horizon
    for horizon in tqdm(horizons, desc=f"Labeling {symbol}"):
        k_up = default_params.get('k_up', 2.0)
        k_down = default_params.get('k_down', 1.0)
        max_bars = horizon * 3  # Default: 3x the horizon

        df = apply_triple_barrier(
            df, horizon, k_up, k_down, max_bars, atr_col
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved labeled data to {output_path}")
    logger.info("")

    return df


def main():
    """Run Stage 4: Initial labeling for all symbols."""
    import sys
    sys.path.insert(0, '/home/user/Research/src')

    from config import FEATURES_DIR, SYMBOLS

    # Output directory
    labels_dir = Path('/home/user/Research/data/labels')
    labels_dir.mkdir(parents=True, exist_ok=True)

    horizons = [1, 5, 20]

    # Default initial parameters (will be optimized in Stage 5)
    default_params = {
        'k_up': 2.0,
        'k_down': 1.0,
        'atr_column': 'atr_14'
    }

    logger.info("=" * 70)
    logger.info("STAGE 4: TRIPLE-BARRIER LABELING")
    logger.info("=" * 70)
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Default parameters: {default_params}")
    logger.info("")

    for symbol in SYMBOLS:
        input_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"
        output_path = labels_dir / f"{symbol}_labels_init.parquet"

        if input_path.exists():
            process_symbol_labeling(
                input_path, output_path, symbol, horizons, default_params
            )
        else:
            logger.warning(f"No features file found for {symbol}: {input_path}")

    logger.info("=" * 70)
    logger.info("STAGE 4 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
