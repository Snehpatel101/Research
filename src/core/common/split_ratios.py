"""
Default train/val/test split ratios - single source of truth.

CFG-010: This module provides the canonical default split ratios used throughout
the codebase. All files that need default ratios should import from here.

Usage:
    from src.core.common.split_ratios import DEFAULT_SPLIT_RATIOS

    train_ratio = DEFAULT_SPLIT_RATIOS["train"]
    val_ratio = DEFAULT_SPLIT_RATIOS["val"]
    test_ratio = DEFAULT_SPLIT_RATIOS["test"]

    # Or unpack all at once
    from src.core.common.split_ratios import (
        DEFAULT_TRAIN_RATIO,
        DEFAULT_VAL_RATIO,
        DEFAULT_TEST_RATIO,
    )
"""

from __future__ import annotations

from typing import Final

# Default split ratios for train/val/test (70/15/15)
# These values are standard for ML model evaluation:
# - 70% train: Sufficient data for model learning
# - 15% val: Adequate for hyperparameter tuning and early stopping
# - 15% test: Reserved for final unbiased evaluation
DEFAULT_TRAIN_RATIO: Final[float] = 0.70
DEFAULT_VAL_RATIO: Final[float] = 0.15
DEFAULT_TEST_RATIO: Final[float] = 0.15

# Dictionary form for convenient access
DEFAULT_SPLIT_RATIOS: Final[dict[str, float]] = {
    "train": DEFAULT_TRAIN_RATIO,
    "val": DEFAULT_VAL_RATIO,
    "test": DEFAULT_TEST_RATIO,
}


def validate_split_ratios(
    train: float = DEFAULT_TRAIN_RATIO,
    val: float = DEFAULT_VAL_RATIO,
    test: float = DEFAULT_TEST_RATIO,
    tolerance: float = 0.01,
) -> list[str]:
    """
    Validate that split ratios are valid and sum to 1.0.

    Parameters
    ----------
    train : float
        Training set ratio
    val : float
        Validation set ratio
    test : float
        Test set ratio
    tolerance : float
        Tolerance for sum check (default 0.01)

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    if train <= 0:
        errors.append(f"train_ratio must be positive, got {train}")
    if val <= 0:
        errors.append(f"val_ratio must be positive, got {val}")
    if test <= 0:
        errors.append(f"test_ratio must be positive, got {test}")

    total = train + val + test
    if abs(total - 1.0) > tolerance:
        errors.append(f"Split ratios must sum to 1.0 (+/- {tolerance}), got {total}")

    return errors


__all__ = [
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_SPLIT_RATIOS",
    "validate_split_ratios",
]
