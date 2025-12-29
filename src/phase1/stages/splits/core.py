"""
Core splitting logic with purging and embargo.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

INVALID_LABEL_SENTINEL = -99


def validate_no_overlap(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray
) -> bool:
    """Validate that there is no overlap between splits."""
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if train_val_overlap:
        logger.error(f"Train/Val overlap: {len(train_val_overlap)} samples")
        return False
    if train_test_overlap:
        logger.error(f"Train/Test overlap: {len(train_test_overlap)} samples")
        return False
    if val_test_overlap:
        logger.error(f"Val/Test overlap: {len(val_test_overlap)} samples")
        return False

    logger.info("No overlap between splits - validation passed")
    return True


def validate_per_symbol_distribution(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    min_symbol_pct: float = 20.0
) -> None:
    """
    Validate that each split has adequate representation from each symbol.

    Args:
        df: DataFrame with 'symbol' column
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        min_symbol_pct: Minimum percentage required for each symbol in each split

    Raises:
        ValueError: If any symbol is underrepresented in any split
    """
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    symbols = df['symbol'].unique()

    for split_name, indices in splits.items():
        split_df = df.iloc[indices]
        split_size = len(indices)

        logger.info(f"\n{split_name.upper()} split symbol distribution:")

        for symbol in symbols:
            symbol_count = (split_df['symbol'] == symbol).sum()
            symbol_pct = symbol_count / split_size * 100

            logger.info(f"  {symbol}: {symbol_count:,} ({symbol_pct:.1f}%)")

            if symbol_pct < min_symbol_pct:
                raise ValueError(
                    f"Symbol {symbol} underrepresented in {split_name} split: "
                    f"{symbol_pct:.1f}% < {min_symbol_pct}% minimum"
                )


def validate_label_distribution(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    horizons: List[int] = None
) -> Dict:
    """
    Validate label distribution across splits, excluding invalid labels.

    Args:
        df: DataFrame with label columns
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        horizons: List of label horizons to check

    Returns:
        Dict: Distribution statistics per label column per split
    """
    if horizons is None:
        from src.common.horizon_config import ACTIVE_HORIZONS
        horizons = ACTIVE_HORIZONS

    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    distribution = {}

    for horizon in horizons:
        label_col = f'label_h{horizon}'
        if label_col not in df.columns:
            continue

        distribution[label_col] = {}
        logger.info(f"\nLabel distribution for horizon {horizon}:")

        for split_name, indices in splits.items():
            split_labels = df.iloc[indices][label_col]

            valid_labels = split_labels[split_labels != INVALID_LABEL_SENTINEL]
            n_invalid = (split_labels == INVALID_LABEL_SENTINEL).sum()
            split_size = len(split_labels)
            invalid_pct = n_invalid / split_size * 100 if split_size > 0 else 0

            if n_invalid > 0:
                logger.info(
                    f"{split_name} split: {n_invalid} invalid samples ({invalid_pct:.1f}%) "
                    f"excluded (edge case: insufficient horizon at dataset end)"
                )

            if invalid_pct > 10:
                logger.warning(
                    f"{split_name} split has {invalid_pct:.1f}% invalid labels in {label_col} - "
                    f"consider reducing max_bars or increasing dataset size"
                )

            counts = valid_labels.value_counts().sort_index()
            total_valid = len(valid_labels)

            distribution[label_col][split_name] = {
                'counts': counts.to_dict(),
                'total_valid': int(total_valid),
                'n_invalid': int(n_invalid),
                'invalid_pct': float(invalid_pct)
            }

            if total_valid > 0:
                dist_str = ", ".join([
                    f"{label}: {count/total_valid*100:.1f}%"
                    for label, count in counts.items()
                ])
                invalid_info = f" (excluded {n_invalid} invalid)" if n_invalid > 0 else ""
                logger.info(f"  {split_name}: {dist_str}{invalid_info}")
            else:
                logger.warning(f"  {split_name}: no valid labels (all {n_invalid} are invalid)")

    return distribution


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge_bars: int = 60,
    embargo_bars: int = 1440,
    datetime_col: str = 'datetime'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create chronological train/val/test splits with purging and embargo.

    Args:
        df: DataFrame sorted by datetime
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        purge_bars: Number of bars to remove at split boundaries
        embargo_bars: Number of bars buffer between splits
        datetime_col: Name of datetime column

    Returns:
        train_indices, val_indices, test_indices, metadata dict

    Raises:
        ValueError: If parameters are invalid or splits would be empty
        KeyError: If datetime column is missing
    """
    if df.empty:
        raise ValueError("DataFrame is empty - cannot create splits")

    n = len(df)

    if datetime_col not in df.columns:
        raise KeyError(
            f"Datetime column '{datetime_col}' not found. "
            f"Available columns: {list(df.columns)[:10]}..."
        )

    if train_ratio <= 0:
        raise ValueError(f"train_ratio must be positive, got {train_ratio}")
    if val_ratio <= 0:
        raise ValueError(f"val_ratio must be positive, got {val_ratio}")
    if test_ratio <= 0:
        raise ValueError(f"test_ratio must be positive, got {test_ratio}")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio:.4f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    if purge_bars < 0:
        raise ValueError(f"purge_bars must be non-negative, got {purge_bars}")
    if embargo_bars < 0:
        raise ValueError(f"embargo_bars must be non-negative, got {embargo_bars}")

    min_required_samples = (purge_bars * 2) + (embargo_bars * 2) + 3
    if n < min_required_samples:
        raise ValueError(
            f"Dataset too small ({n} samples) for purge_bars={purge_bars} and embargo_bars={embargo_bars}. "
            f"Need at least {min_required_samples} samples. "
            f"Reduce purge_bars/embargo_bars or increase data size."
        )

    if not df[datetime_col].is_monotonic_increasing:
        logger.warning(f"DataFrame not sorted by {datetime_col}, sorting now...")
        df = df.copy().sort_values(datetime_col).reset_index(drop=True)

    n = len(df)
    logger.info(f"Total samples: {n:,}")
    logger.info(f"Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    logger.info(f"Purge bars: {purge_bars}, Embargo bars: {embargo_bars}")

    train_end_raw = int(n * train_ratio)
    val_end_raw = int(n * (train_ratio + val_ratio))

    train_end = train_end_raw - purge_bars
    val_start = train_end_raw + embargo_bars
    val_end = val_end_raw - purge_bars
    test_start = val_end_raw + embargo_bars

    if train_end <= 0:
        raise ValueError(
            f"Training set eliminated by purging. "
            f"train_end_raw={train_end_raw}, purge_bars={purge_bars}, result={train_end}. "
            f"Reduce purge_bars (current: {purge_bars}) or increase training data. "
            f"Minimum train_end_raw needed: {purge_bars + 1}"
        )

    if val_start >= val_end:
        val_samples_available = val_end_raw - train_end_raw
        samples_needed = embargo_bars + purge_bars
        raise ValueError(
            f"Validation set is empty after purge/embargo. "
            f"val_start={val_start}, val_end={val_end}. "
            f"Available samples between train/val boundary: {val_samples_available}. "
            f"Samples consumed by embargo+purge: {samples_needed}. "
            f"Reduce embargo_bars (current: {embargo_bars}) or purge_bars (current: {purge_bars}), "
            f"or increase validation data ratio."
        )

    if test_start >= n:
        test_samples_available = n - val_end_raw
        raise ValueError(
            f"Test set is empty after embargo. "
            f"test_start={test_start}, n={n}. "
            f"Available samples after val boundary: {test_samples_available}. "
            f"Embargo consumes: {embargo_bars} samples. "
            f"Reduce embargo_bars (current: {embargo_bars}) or increase test data ratio."
        )

    train_indices = np.arange(0, train_end)
    val_indices = np.arange(val_start, val_end)
    test_indices = np.arange(test_start, n)

    train_dates = df.iloc[train_indices][datetime_col]
    val_dates = df.iloc[val_indices][datetime_col]
    test_dates = df.iloc[test_indices][datetime_col]

    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_indices):,} samples ({len(train_indices)/n:.1%})")
    logger.info(f"  Val:   {len(val_indices):,} samples ({len(val_indices)/n:.1%})")
    logger.info(f"  Test:  {len(test_indices):,} samples ({len(test_indices)/n:.1%})")
    logger.info(f"  Lost to purge/embargo: {n - len(train_indices) - len(val_indices) - len(test_indices):,} samples")

    logger.info(f"\nDate ranges:")
    logger.info(f"  Train: {train_dates.min()} to {train_dates.max()}")
    logger.info(f"  Val:   {val_dates.min()} to {val_dates.max()}")
    logger.info(f"  Test:  {test_dates.min()} to {test_dates.max()}")

    if not validate_no_overlap(train_indices, val_indices, test_indices):
        raise RuntimeError("Split validation failed: overlapping indices detected")

    metadata = {
        'total_samples': n,
        'train_samples': int(len(train_indices)),
        'val_samples': int(len(val_indices)),
        'test_samples': int(len(test_indices)),
        'train_ratio': float(train_ratio),
        'val_ratio': float(val_ratio),
        'test_ratio': float(test_ratio),
        'purge_bars': int(purge_bars),
        'embargo_bars': int(embargo_bars),
        'train_date_start': str(train_dates.min()),
        'train_date_end': str(train_dates.max()),
        'val_date_start': str(val_dates.min()),
        'val_date_end': str(val_dates.max()),
        'test_date_start': str(test_dates.min()),
        'test_date_end': str(test_dates.max()),
        'created_at': datetime.now().isoformat(),
        'validation_passed': True
    }

    return train_indices, val_indices, test_indices, metadata
