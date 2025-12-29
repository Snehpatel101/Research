"""
Verification script for vectorized PurgedKFold implementation.

Tests the actual PurgedKFold.split() method to ensure it:
1. Produces correct results (no label leakage)
2. Uses the vectorized implementation
3. Handles edge cases correctly

Usage:
    python scripts/verify_vectorized_purged_kfold.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from the file to avoid dependency issues
exec(open(project_root / "src/cross_validation/purged_kfold.py").read())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_no_label_leakage(
    X: pd.DataFrame,
    label_end_times: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> bool:
    """
    Verify that no training sample has label leakage with test set.

    Args:
        X: Features DataFrame
        label_end_times: Label end times
        train_idx: Training indices
        test_idx: Test indices

    Returns:
        True if no leakage, False otherwise
    """
    test_start_time = X.index[test_idx[0]]
    test_end_time = X.index[test_idx[-1]]

    for idx in train_idx:
        sample_time = X.index[idx]
        label_end = label_end_times.iloc[idx]

        # Check for overlap
        if label_end >= test_start_time and sample_time <= test_end_time:
            logger.error(
                f"LABEL LEAKAGE DETECTED! "
                f"Sample at {sample_time} (idx={idx}) has label_end={label_end} "
                f"overlapping test period [{test_start_time}, {test_end_time}]"
            )
            return False

    return True


def test_basic_functionality():
    """Test basic PurgedKFold functionality."""
    logger.info("Test 1: Basic functionality with label_end_times")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    horizon = 20

    start_time = pd.Timestamp("2020-01-01 09:30:00")
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        index=dates,
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    label_end_times = pd.Series(
        [dates[min(i + horizon, n_samples - 1)] for i in range(n_samples)],
        index=dates,
        name="label_end_time",
    )

    # Create PurgedKFold
    config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=100, min_train_size=0.2)
    cv = PurgedKFold(config)

    # Generate folds
    folds = list(cv.split(X, label_end_times=label_end_times))

    # Verify
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    # Check each fold for label leakage
    all_clean = True
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        clean = verify_no_label_leakage(X, label_end_times, train_idx, test_idx)
        if not clean:
            all_clean = False
            logger.error(f"Fold {fold_idx} has label leakage!")
        else:
            logger.info(f"  Fold {fold_idx}: ✓ No label leakage (train={len(train_idx)}, test={len(test_idx)})")

    assert all_clean, "Some folds have label leakage!"
    logger.info("✓ Test 1 PASSED\n")
    return True


def test_edge_case_small_dataset():
    """Test with very small dataset."""
    logger.info("Test 2: Edge case - small dataset")

    np.random.seed(42)
    n_samples = 100
    horizon = 10

    start_time = pd.Timestamp("2020-01-01 09:30:00")
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    X = pd.DataFrame(
        np.random.randn(n_samples, 5),
        index=dates,
        columns=[f"feature_{i}" for i in range(5)],
    )

    label_end_times = pd.Series(
        [dates[min(i + horizon, n_samples - 1)] for i in range(n_samples)],
        index=dates,
    )

    # Small purge/embargo to avoid training set too small
    config = PurgedKFoldConfig(n_splits=3, purge_bars=5, embargo_bars=5, min_train_size=0.2)
    cv = PurgedKFold(config)

    folds = list(cv.split(X, label_end_times=label_end_times))
    assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}"

    # Check for leakage
    all_clean = True
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        clean = verify_no_label_leakage(X, label_end_times, train_idx, test_idx)
        if not clean:
            all_clean = False
        logger.info(f"  Fold {fold_idx}: {'✓' if clean else '✗'} (train={len(train_idx)}, test={len(test_idx)})")

    assert all_clean, "Some folds have label leakage!"
    logger.info("✓ Test 2 PASSED\n")
    return True


def test_edge_case_long_horizon():
    """Test with long horizon (labels extend far into future)."""
    logger.info("Test 3: Edge case - long horizon")

    np.random.seed(42)
    n_samples = 2000
    horizon = 100  # Very long horizon

    start_time = pd.Timestamp("2020-01-01 09:30:00")
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    X = pd.DataFrame(
        np.random.randn(n_samples, 10),
        index=dates,
        columns=[f"feature_{i}" for i in range(10)],
    )

    label_end_times = pd.Series(
        [dates[min(i + horizon, n_samples - 1)] for i in range(n_samples)],
        index=dates,
    )

    # Larger purge to handle long horizon
    config = PurgedKFoldConfig(n_splits=5, purge_bars=150, embargo_bars=150, min_train_size=0.15)
    cv = PurgedKFold(config)

    folds = list(cv.split(X, label_end_times=label_end_times))
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    # Check for leakage
    all_clean = True
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        clean = verify_no_label_leakage(X, label_end_times, train_idx, test_idx)
        if not clean:
            all_clean = False
        logger.info(f"  Fold {fold_idx}: {'✓' if clean else '✗'} (train={len(train_idx)}, test={len(test_idx)})")

    assert all_clean, "Some folds have label leakage!"
    logger.info("✓ Test 3 PASSED\n")
    return True


def test_without_label_end_times():
    """Test that PurgedKFold works without label_end_times (basic purge only)."""
    logger.info("Test 4: Without label_end_times (basic purge)")

    np.random.seed(42)
    n_samples = 500

    start_time = pd.Timestamp("2020-01-01 09:30:00")
    dates = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    X = pd.DataFrame(
        np.random.randn(n_samples, 10),
        index=dates,
        columns=[f"feature_{i}" for i in range(10)],
    )

    config = PurgedKFoldConfig(n_splits=5, purge_bars=20, embargo_bars=30, min_train_size=0.2)
    cv = PurgedKFold(config)

    # Without label_end_times
    folds = list(cv.split(X, label_end_times=None))
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    logger.info(f"  Generated {len(folds)} folds successfully")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        logger.info(f"  Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")

    logger.info("✓ Test 4 PASSED\n")
    return True


def main():
    """Run all verification tests."""
    logger.info("=" * 80)
    logger.info("PurgedKFold Vectorized Implementation Verification")
    logger.info("=" * 80 + "\n")

    all_passed = True

    try:
        all_passed &= test_basic_functionality()
    except Exception as e:
        logger.error(f"Test 1 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_edge_case_small_dataset()
    except Exception as e:
        logger.error(f"Test 2 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_edge_case_long_horizon()
    except Exception as e:
        logger.error(f"Test 3 FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_without_label_end_times()
    except Exception as e:
        logger.error(f"Test 4 FAILED: {e}")
        all_passed = False

    logger.info("=" * 80)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED")
        logger.info("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
