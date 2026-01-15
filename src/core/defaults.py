"""
Centralized defaults registry - single source of truth for all default values.

CONFIG-002: This module consolidates defaults from:
- src/models/config/trainer_config.py
- src/phase1/config/runtime.py
- src/phase1/pipeline_config.py
- config/pipeline/training.yaml

All default values should reference this module.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GlobalDefaults:
    """
    Immutable defaults for the entire ML pipeline.

    All default values used across the codebase should be defined here.
    This ensures consistency and makes it easy to find/change defaults.
    """

    # ==========================================================================
    # RANDOM SEED
    # ==========================================================================
    RANDOM_SEED: int = 42

    # ==========================================================================
    # DATA SPLIT RATIOS
    # ==========================================================================
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # ==========================================================================
    # TRAINING HYPERPARAMETERS
    # ==========================================================================
    DEFAULT_BATCH_SIZE: int = 256
    DEFAULT_MAX_EPOCHS: int = 100
    DEFAULT_LEARNING_RATE: float = 0.001
    DEFAULT_EARLY_STOPPING_PATIENCE: int = 10
    DEFAULT_MIN_DELTA: float = 0.0001

    # ==========================================================================
    # SEQUENCE MODEL PARAMETERS
    # ==========================================================================
    DEFAULT_SEQUENCE_LENGTH: int = 60
    DEFAULT_HIDDEN_SIZE: int = 128
    DEFAULT_NUM_LAYERS: int = 2
    DEFAULT_DROPOUT: float = 0.1

    # ==========================================================================
    # TIMEFRAME DEFAULTS
    # ==========================================================================
    DEFAULT_TARGET_TIMEFRAME: str = "1min"  # Canonical source (CLAUDE.md)
    DEFAULT_BAR_RESOLUTION: str = "1min"

    # ==========================================================================
    # LABELING DEFAULTS
    # ==========================================================================
    DEFAULT_HORIZON: int = 20
    DEFAULT_LABEL_HORIZONS: tuple = (5, 10, 15, 20)

    # ==========================================================================
    # PURGE/EMBARGO (for leakage prevention)
    # ==========================================================================
    DEFAULT_PURGE_BARS: int = 60
    DEFAULT_EMBARGO_BARS: int = 1440

    # ==========================================================================
    # CROSS-VALIDATION
    # ==========================================================================
    DEFAULT_CV_N_SPLITS: int = 5
    DEFAULT_N_JOBS: int = 1


# Singleton instance
DEFAULTS = GlobalDefaults()


def get_default(key: str) -> Any:
    """Get a default value by key name."""
    if hasattr(DEFAULTS, key):
        return getattr(DEFAULTS, key)
    raise KeyError(f"Unknown default key: {key}")


def as_dict() -> dict[str, Any]:
    """Get all defaults as a dictionary."""
    return {
        field: getattr(DEFAULTS, field)
        for field in GlobalDefaults.__dataclass_fields__
    }


__all__ = [
    "GlobalDefaults",
    "DEFAULTS",
    "get_default",
    "as_dict",
]
