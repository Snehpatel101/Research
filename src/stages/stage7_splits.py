"""
Stage 7: Time-Based Splitting with Purging and Embargo
Implements chronological train/val/test splits with leakage prevention
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

# Configure logging - use NullHandler to avoid duplicate logs when imported as module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge_bars: int = 60,
    embargo_bars: int = 288,
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
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Sort by datetime if not already sorted
    if not df[datetime_col].is_monotonic_increasing:
        logger.warning(f"DataFrame not sorted by {datetime_col}, sorting now...")
        df = df.sort_values(datetime_col).reset_index(drop=True)

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

    # Validate indices
    if train_end <= 0:
        raise ValueError(f"Train set too small after purging: {train_end}")
    if val_start >= val_end:
        raise ValueError(f"Validation set eliminated by purge/embargo: {val_start} >= {val_end}")
    if test_start >= n:
        raise ValueError(f"Test set eliminated by embargo: {test_start} >= {n}")

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
    embargo_bars: int = 288,
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
    train_path = split_dir / "train.npy"
    val_path = split_dir / "val.npy"
    test_path = split_dir / "test.npy"
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
    """Run splits creation for the default configuration."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import FINAL_DATA_DIR, SPLITS_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, PURGE_BARS, EMBARGO_BARS

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    metadata = create_splits(
        data_path=data_path,
        output_dir=SPLITS_DIR,
        run_id=None,  # Auto-generate timestamp
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        purge_bars=PURGE_BARS,
        embargo_bars=EMBARGO_BARS
    )

    logger.info(f"\nRun ID: {metadata['run_id']}")
    logger.info(f"Split directory: {metadata['split_dir']}")


if __name__ == "__main__":
    main()
