"""
Stage 7: Time-Based Splitting with Purging and Embargo
Implements chronological train/val/test splits with leakage prevention
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

# Configure logging - use NullHandler to avoid duplicate logs when imported as module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Sentinel value for invalid labels (e.g., labels near end of dataset where
# triple barrier cannot complete, or labels filtered out by quality checks)
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

    logger.info("✓ No overlap between splits - validation passed")
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

    Invalid labels (INVALID_LABEL_SENTINEL = -99) are excluded from distribution
    statistics but tracked separately. A warning is issued if more than 10% of
    labels in any split are invalid.

    Args:
        df: DataFrame with label columns
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        horizons: List of label horizons to check (default: [5, 20])

    Returns:
        Dict: Distribution statistics per label column per split, including:
            - counts: valid label value counts
            - total_valid: number of valid labels
            - n_invalid: number of invalid labels
            - invalid_pct: percentage of invalid labels
    """
    if horizons is None:
        horizons = [5, 20]

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

            # Filter out invalid labels
            valid_labels = split_labels[split_labels != INVALID_LABEL_SENTINEL]

            # Count invalid labels
            n_invalid = (split_labels == INVALID_LABEL_SENTINEL).sum()
            invalid_pct = n_invalid / len(split_labels) * 100 if len(split_labels) > 0 else 0

            # Warn if high invalid rate (>10%)
            if invalid_pct > 10:
                logger.warning(
                    f"{split_name} split has {invalid_pct:.1f}% invalid labels in {label_col}"
                )

            # Calculate distribution on valid labels only
            counts = valid_labels.value_counts().sort_index()
            total_valid = len(valid_labels)

            # Store distribution data
            distribution[label_col][split_name] = {
                'counts': counts.to_dict(),
                'total_valid': int(total_valid),
                'n_invalid': int(n_invalid),
                'invalid_pct': float(invalid_pct)
            }

            # Log distribution (valid labels only)
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
    embargo_bars: int = 1440,  # 5 days for 5-min data (288 bars/day * 5)
    datetime_col: str = 'datetime'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create chronological train/val/test splits with purging and embargo.

    Args:
        df: DataFrame sorted by datetime
        train_ratio: Ratio for training set (default 0.70)
        val_ratio: Ratio for validation set (default 0.15)
        test_ratio: Ratio for test set (default 0.15)
        purge_bars: Number of bars to remove at split boundaries
        embargo_bars: Number of bars buffer between splits
        datetime_col: Name of datetime column

    Returns:
        train_indices, val_indices, test_indices, metadata dict

    Raises:
        ValueError: If parameters are invalid or splits would be empty
        KeyError: If datetime column is missing
    """
    # === PARAMETER VALIDATION ===
    # Validate DataFrame is not empty
    if df.empty:
        raise ValueError("DataFrame is empty - cannot create splits")

    n = len(df)

    # Validate datetime column exists
    if datetime_col not in df.columns:
        raise KeyError(
            f"Datetime column '{datetime_col}' not found. "
            f"Available columns: {list(df.columns)[:10]}..."
        )

    # Validate ratios are positive
    if train_ratio <= 0:
        raise ValueError(f"train_ratio must be positive, got {train_ratio}")
    if val_ratio <= 0:
        raise ValueError(f"val_ratio must be positive, got {val_ratio}")
    if test_ratio <= 0:
        raise ValueError(f"test_ratio must be positive, got {test_ratio}")

    # Validate ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio:.4f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    # Validate purge_bars and embargo_bars are non-negative
    if purge_bars < 0:
        raise ValueError(f"purge_bars must be non-negative, got {purge_bars}")
    if embargo_bars < 0:
        raise ValueError(f"embargo_bars must be non-negative, got {embargo_bars}")

    # Validate dataset is large enough for purging and embargo
    min_required_samples = (purge_bars * 2) + (embargo_bars * 2) + 3  # At least 1 sample per split
    if n < min_required_samples:
        raise ValueError(
            f"Dataset too small ({n} samples) for purge_bars={purge_bars} and embargo_bars={embargo_bars}. "
            f"Need at least {min_required_samples} samples. "
            f"Reduce purge_bars/embargo_bars or increase data size."
        )

    # Sort by datetime if not already sorted
    if not df[datetime_col].is_monotonic_increasing:
        logger.warning(f"DataFrame not sorted by {datetime_col}, sorting now...")
        df = df.copy().sort_values(datetime_col).reset_index(drop=True)

    n = len(df)
    logger.info(f"Total samples: {n:,}")
    logger.info(f"Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    logger.info(f"Purge bars: {purge_bars}, Embargo bars: {embargo_bars}")

    # Calculate raw split points
    train_end_raw = int(n * train_ratio)
    val_end_raw = int(n * (train_ratio + val_ratio))

    # Apply purging: remove N bars before each split boundary
    train_end = train_end_raw - purge_bars
    val_start = train_end_raw + embargo_bars
    val_end = val_end_raw - purge_bars
    test_start = val_end_raw + embargo_bars

    # Validate split indices with actionable error messages
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

    # Create index arrays
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(val_start, val_end)
    test_indices = np.arange(test_start, n)

    # Get date ranges
    train_dates = df.iloc[train_indices][datetime_col]
    val_dates = df.iloc[val_indices][datetime_col]
    test_dates = df.iloc[test_indices][datetime_col]

    # Log split information
    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_indices):,} samples ({len(train_indices)/n:.1%})")
    logger.info(f"  Val:   {len(val_indices):,} samples ({len(val_indices)/n:.1%})")
    logger.info(f"  Test:  {len(test_indices):,} samples ({len(test_indices)/n:.1%})")
    logger.info(f"  Lost to purge/embargo: {n - len(train_indices) - len(val_indices) - len(test_indices):,} samples")

    logger.info(f"\nDate ranges:")
    logger.info(f"  Train: {train_dates.min()} to {train_dates.max()}")
    logger.info(f"  Val:   {val_dates.min()} to {val_dates.max()}")
    logger.info(f"  Test:  {test_dates.min()} to {test_dates.max()}")

    # Validate no overlap
    if not validate_no_overlap(train_indices, val_indices, test_indices):
        raise RuntimeError("Split validation failed: overlapping indices detected")

    # Create metadata
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


def create_splits(
    data_path: Path,
    output_dir: Path,
    run_id: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge_bars: int = 60,
    embargo_bars: int = 1440,  # 5 days for 5-min data (288 bars/day * 5)
    datetime_col: str = 'datetime'
) -> Dict:
    """
    Main function to create and save train/val/test splits.

    Args:
        data_path: Path to combined labeled parquet file
        output_dir: Base directory for splits
        run_id: Optional run identifier for organizing outputs
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        purge_bars: Bars to purge at boundaries
        embargo_bars: Embargo buffer bars
        datetime_col: Datetime column name

    Returns:
        Dictionary with split metadata and paths
    """
    logger.info("="*70)
    logger.info("STAGE 7: TIME-BASED SPLITTING")
    logger.info("="*70)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Create run-specific directory
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    split_dir = output_dir / run_id
    split_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {split_dir}")

    # Create splits
    train_idx, val_idx, test_idx, metadata = create_chronological_splits(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        datetime_col=datetime_col
    )

    # Save indices as .npy files
    # Filenames must match what downstream stages expect (stage7_5_scaling, baseline_backtest)
    train_path = split_dir / "train_indices.npy"
    val_path = split_dir / "val_indices.npy"
    test_path = split_dir / "test_indices.npy"
    config_path = split_dir / "split_config.json"

    np.save(train_path, train_idx)
    np.save(val_path, val_idx)
    np.save(test_path, test_idx)

    logger.info(f"\nSaved split indices:")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Val:   {val_path}")
    logger.info(f"  Test:  {test_path}")

    # Add paths to metadata
    metadata['run_id'] = run_id
    metadata['data_path'] = str(data_path)
    metadata['split_dir'] = str(split_dir)
    metadata['train_path'] = str(train_path)
    metadata['val_path'] = str(val_path)
    metadata['test_path'] = str(test_path)

    # Save metadata
    with open(config_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Config: {config_path}")

    logger.info("\n" + "="*70)
    logger.info("STAGE 7 COMPLETE")
    logger.info("="*70)

    return metadata


def main():
    """Run splits creation for the default configuration with auto-scaled purge/embargo."""
    from src.config import (
        FINAL_DATA_DIR,
        SPLITS_DIR,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        PURGE_BARS,
        EMBARGO_BARS,
        HORIZONS,
        auto_scale_purge_embargo,
    )

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Auto-scale purge and embargo based on horizons
    # This ensures sufficient buffer for the configured horizon labels
    purge_bars, embargo_bars = auto_scale_purge_embargo(HORIZONS)

    logger.info(f"Auto-scaled purge/embargo for horizons {HORIZONS}:")
    logger.info(f"  Purge bars: {purge_bars} (was {PURGE_BARS})")
    logger.info(f"  Embargo bars: {embargo_bars} (was {EMBARGO_BARS})")

    metadata = create_splits(
        data_path=data_path,
        output_dir=SPLITS_DIR,
        run_id=None,  # Auto-generate timestamp
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars
    )

    logger.info(f"\nRun ID: {metadata['run_id']}")
    logger.info(f"Split directory: {metadata['split_dir']}")


if __name__ == "__main__":
    main()
