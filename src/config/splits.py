"""
Train/validation/test split configuration.

This module contains configuration for data splitting, including:
- Split ratios (train/val/test)
- Purge and embargo bars for leakage prevention
"""

# =============================================================================
# SPLIT RATIOS
# =============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# PURGE AND EMBARGO CONFIGURATION - CRITICAL FOR LEAKAGE PREVENTION
# =============================================================================
# PURGE_BARS: Number of bars to remove at split boundaries to prevent look-ahead bias.
# CRITICAL: Must equal max(max_bars) across all horizons to fully prevent leakage.
# H20 uses max_bars=60, therefore PURGE_BARS must be at least 60.
# Previous value of 20 was INSUFFICIENT and allowed label leakage from future data.
PURGE_BARS = 60  # = max_bars for H20 (CRITICAL: prevents leakage)

# EMBARGO_BARS: Buffer between splits to account for serial correlation in features.
# Serial correlation in financial features can persist for multiple days.
# 1440 bars = 5 days for 5-min data (288 bars/day * 5 days).
# Previous value of 288 (1 day) was insufficient for capturing feature decay patterns.
EMBARGO_BARS = 1440  # ~5 days for 5-min data (CRITICAL: ensures feature decorrelation)


def validate_splits_config() -> list[str]:
    """
    Validate split configuration values.

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    import numpy as np

    errors = []

    # Validate split ratios sum to 1.0
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not np.isclose(total_ratio, 1.0):
        errors.append(
            f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
            f"(train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO})"
        )

    # Validate individual split ratios are positive
    if TRAIN_RATIO <= 0:
        errors.append(f"TRAIN_RATIO must be positive, got {TRAIN_RATIO}")
    if VAL_RATIO <= 0:
        errors.append(f"VAL_RATIO must be positive, got {VAL_RATIO}")
    if TEST_RATIO <= 0:
        errors.append(f"TEST_RATIO must be positive, got {TEST_RATIO}")

    # Validate purge and embargo bars are non-negative
    if PURGE_BARS < 0:
        errors.append(f"PURGE_BARS must be non-negative, got {PURGE_BARS}")
    if EMBARGO_BARS < 0:
        errors.append(f"EMBARGO_BARS must be non-negative, got {EMBARGO_BARS}")

    return errors


def get_splits_config() -> dict:
    """
    Get split configuration as a dictionary.

    Returns
    -------
    dict
        Dictionary with split configuration values
    """
    return {
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'purge_bars': PURGE_BARS,
        'embargo_bars': EMBARGO_BARS,
    }
