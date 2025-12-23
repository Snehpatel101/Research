"""
Stage 4: Labeling with Pluggable Strategies

This module provides the stage 4 labeling pipeline with support for multiple
labeling strategies. The default is triple-barrier labeling with ATR-based
dynamic barriers, but other strategies can be used via the labeling module.

REFACTORED (2025-12): Strategy pattern for pluggable labelers
The labeling logic has been moved to src/stages/labeling/ with:
- Base LabelingStrategy ABC
- TripleBarrierLabeler (default)
- DirectionalLabeler
- ThresholdLabeler
- RegressionLabeler

For backward compatibility, this module re-exports the core functions
from the labeling module.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import from the labeling module
from .labeling import (
    LabelingResult,
    LabelingStrategy,
    LabelingType,
    ThresholdLabeler,
    TripleBarrierLabeler,
    get_labeler,
    triple_barrier_numba,
)

# =============================================================================
# BARRIER CONFIGURATION - IMPORTED FROM CENTRAL CONFIG
# =============================================================================
from src.config import (
    BARRIER_PARAMS,
    BARRIER_PARAMS_DEFAULT,
    PERCENTAGE_BARRIER_PARAMS,
    get_barrier_params,
)
logger.info("Loaded barrier parameters from config.py (single source of truth)")


# Re-export for backward compatibility
__all__ = [
    # Legacy API
    'triple_barrier_numba',
    'apply_triple_barrier',
    'process_symbol_labeling',
    'main',
    # New API
    'LabelingType',
    'LabelingStrategy',
    'LabelingResult',
    'TripleBarrierLabeler',
    'ThresholdLabeler',
    'get_labeler',
]


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

    This is a wrapper around the TripleBarrierLabeler for backward compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data and ATR
    horizon : int
        Horizon identifier (e.g., 1, 5, 20)
    k_up : float, optional
        Profit barrier multiplier (ATR-based). If None, uses BARRIER_PARAMS
    k_down : float, optional
        Stop barrier multiplier (ATR-based). If None, uses BARRIER_PARAMS
    max_bars : int, optional
        Maximum bars. If None, uses BARRIER_PARAMS
    atr_column : str
        Column name for ATR values
    use_percentage : bool
        If True, use percentage barriers instead of ATR
    pct_up : float, optional
        Percentage for upper barrier (e.g., 0.002 = 0.2%)
    pct_down : float, optional
        Percentage for lower barrier

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - label_h{horizon}: Label values (-1, 0, 1, or -99 for invalid)
        - bars_to_hit_h{horizon}: Bars until barrier hit
        - mae_h{horizon}: Maximum adverse excursion
        - mfe_h{horizon}: Maximum favorable excursion
        - touch_type_h{horizon}: Which barrier was touched (1, -1, or 0)

    Notes
    -----
    CRITICAL FIX (2025-12-21): The last max_bars samples are marked with
    label=-99 (invalid) because there isn't enough future data to properly
    evaluate barrier hits. These samples MUST be filtered out before model
    training to prevent edge case leakage.
    """
    # === PARAMETER VALIDATION ===
    if df.empty:
        raise ValueError("DataFrame is empty - cannot apply triple barrier labeling")

    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(
            f"horizon must be a positive integer, got {horizon} "
            f"(type: {type(horizon).__name__})"
        )

    if k_up is not None and k_up <= 0:
        raise ValueError(f"k_up must be positive, got {k_up}")
    if k_down is not None and k_down <= 0:
        raise ValueError(f"k_down must be positive, got {k_down}")
    if max_bars is not None and max_bars <= 0:
        raise ValueError(f"max_bars must be positive, got {max_bars}")

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

    if use_percentage:
        # Use ThresholdLabeler for percentage-based barriers
        pct_up = pct_up or PERCENTAGE_BARRIER_PARAMS.get(horizon, {}).get('pct_up', 0.005)
        pct_down = pct_down or PERCENTAGE_BARRIER_PARAMS.get(horizon, {}).get('pct_down', 0.005)
        max_bars = max_bars or PERCENTAGE_BARRIER_PARAMS.get(horizon, {}).get('max_bars', 20)

        labeler = ThresholdLabeler(pct_up=pct_up, pct_down=pct_down, max_bars=max_bars)
        result = labeler.compute_labels(df, horizon)

        # Add labels to dataframe with same column naming as triple-barrier
        df = df.copy()
        df[f'label_h{horizon}'] = result.labels
        df[f'bars_to_hit_h{horizon}'] = result.metadata.get('bars_to_hit', 0)
        df[f'mae_h{horizon}'] = result.metadata.get('max_loss', 0)
        df[f'mfe_h{horizon}'] = result.metadata.get('max_gain', 0)
        df[f'touch_type_h{horizon}'] = 0

        return df

    # Validate ATR column exists
    if atr_column not in df.columns:
        available_cols = [c for c in df.columns if 'atr' in c.lower()]
        raise KeyError(
            f"ATR column '{atr_column}' not found. "
            f"Available ATR-like columns: {available_cols if available_cols else 'none'}"
        )

    # Use TripleBarrierLabeler
    labeler = TripleBarrierLabeler(atr_column=atr_column)

    result = labeler.compute_labels(
        df, horizon, k_up=k_up, k_down=k_down, max_bars=max_bars
    )

    # Add labels to dataframe
    df = labeler.add_labels_to_dataframe(df, result)

    return df


def process_symbol_labeling(
    input_path: Path,
    output_path: Path,
    symbol: str,
    horizons: List[int] = [1, 5, 20],
    barrier_params: Optional[Dict] = None,
    atr_column: str = 'atr_14',
    labeling_strategy: LabelingType | str = LabelingType.TRIPLE_BARRIER,
    strategy_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Process labeling for a single symbol with multiple horizons.

    Parameters
    ----------
    input_path : Path
        Path to features parquet file
    output_path : Path
        Path to save labeled data
    symbol : str
        Symbol name
    horizons : list[int]
        List of horizons to label (default: [1, 5, 20])
    barrier_params : dict, optional
        Dict mapping horizon -> {k_up, k_down, max_bars}
        If None, uses symbol-specific params from config.py
    atr_column : str
        Column name for ATR values
    labeling_strategy : LabelingType or str
        Type of labeling strategy to use (default: TRIPLE_BARRIER)
    strategy_config : dict, optional
        Additional configuration for the labeling strategy

    Returns
    -------
    pd.DataFrame
        Labeled DataFrame
    """
    # Determine if using triple-barrier
    is_triple_barrier = (
        labeling_strategy == LabelingType.TRIPLE_BARRIER or
        labeling_strategy == 'triple_barrier'
    )

    # Use symbol-specific parameters from config.py if not provided
    if barrier_params is None and is_triple_barrier:
        barrier_params = {}
        for h in horizons:
            barrier_params[h] = get_barrier_params(symbol, h)

    logger.info("=" * 70)
    if is_triple_barrier:
        logger.info(f"Triple-Barrier Labeling: {symbol}")
    else:
        logger.info(f"{labeling_strategy} Labeling: {symbol}")
    logger.info("=" * 70)

    if is_triple_barrier:
        logger.info("Using symbol-specific barrier parameters from config.py:")
        for h in horizons:
            if h in barrier_params:
                params = barrier_params[h]
                logger.info(
                    f"  H{h}: k_up={params['k_up']:.2f}, k_down={params['k_down']:.2f}, "
                    f"max_bars={params['max_bars']} - {params.get('description', '')}"
                )

    # Load features
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # Create labeler and apply labels
    config = strategy_config or {}

    if is_triple_barrier:
        # Ensure ATR column exists
        if atr_column not in df.columns:
            raise ValueError(f"ATR column '{atr_column}' not found in dataframe")

        labeler = TripleBarrierLabeler(atr_column=atr_column, **config)

        # Apply labeling for each horizon
        for horizon in tqdm(horizons, desc=f"Labeling {symbol}"):
            params = barrier_params[horizon]
            result = labeler.compute_labels(
                df, horizon,
                k_up=params['k_up'],
                k_down=params['k_down'],
                max_bars=params['max_bars']
            )
            df = labeler.add_labels_to_dataframe(df, result)
    else:
        # Use factory for other strategies
        labeler = get_labeler(labeling_strategy, **config)

        # Apply labeling for each horizon
        for horizon in tqdm(horizons, desc=f"Labeling {symbol}"):
            result = labeler.compute_labels(df, horizon)
            df = labeler.add_labels_to_dataframe(df, result)

    # Log final label distribution summary
    _log_label_summary(df, horizons)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved labeled data to {output_path}")
    logger.info("")

    return df


def _log_label_summary(df: pd.DataFrame, horizons: List[int]) -> None:
    """Log label distribution summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("LABEL DISTRIBUTION SUMMARY (excluding invalid samples)")
    logger.info("=" * 70)

    for horizon in horizons:
        label_col = f'label_h{horizon}'
        if label_col not in df.columns:
            continue

        # Exclude invalid labels (-99)
        valid_labels = df[df[label_col] != -99][label_col]
        counts = valid_labels.value_counts().sort_index()
        total_valid = len(valid_labels)
        total_all = len(df)
        num_invalid = total_all - total_valid

        if total_valid == 0:
            logger.warning(f"H{horizon}: No valid labels")
            continue

        short_pct = counts.get(-1, 0) / total_valid * 100
        neutral_pct = counts.get(0, 0) / total_valid * 100
        long_pct = counts.get(1, 0) / total_valid * 100

        logger.info(
            f"H{horizon}: Short={short_pct:.1f}% | Neutral={neutral_pct:.1f}% | "
            f"Long={long_pct:.1f}% (valid: {total_valid:,}, invalid: {num_invalid})"
        )


def main() -> None:
    """Run Stage 4: Initial labeling for all symbols using dynamic horizons."""
    from src.config import (
        FEATURES_DIR,
        HORIZONS,
        LOOKBACK_HORIZONS,
        PROJECT_ROOT,
        SYMBOLS,
        validate_horizons,
    )

    # Output directory
    labels_dir = PROJECT_ROOT / 'data' / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Use LOOKBACK_HORIZONS for labeling (includes H1)
    # HORIZONS excludes H1 which is not viable after transaction costs
    horizons = LOOKBACK_HORIZONS  # [1, 5, 20] - label all, filter later

    # Validate horizons
    try:
        validate_horizons(horizons)
    except ValueError as e:
        logger.error(f"Horizon validation failed: {e}")
        raise

    logger.info("=" * 70)
    logger.info("STAGE 4: TRIPLE-BARRIER LABELING (DYNAMIC HORIZONS)")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Labeling horizons: {horizons}")
    logger.info(f"Active horizons (for training): {HORIZONS}")
    logger.info("")
    logger.info("SYMBOL-SPECIFIC BARRIER PARAMETERS from config.py:")
    logger.info("-" * 70)
    for symbol in SYMBOLS:
        logger.info(f"{symbol}:")
        for horizon in horizons:
            params = get_barrier_params(symbol, horizon)
            logger.info(
                f"  H{horizon:2d}: k_up={params['k_up']:.2f}, "
                f"k_down={params['k_down']:.2f}, max_bars={params['max_bars']:2d}"
            )
            logger.info(f"       {params.get('description', '')}")
    logger.info("-" * 70)
    logger.info("")

    for symbol in SYMBOLS:
        input_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"
        output_path = labels_dir / f"{symbol}_labels_init.parquet"

        if input_path.exists():
            process_symbol_labeling(
                input_path, output_path, symbol, horizons,
                barrier_params=None,
                atr_column='atr_14'
            )
        else:
            logger.warning(f"No features file found for {symbol}: {input_path}")

    logger.info("=" * 70)
    logger.info("STAGE 4 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
