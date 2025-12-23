"""
Core constants and path definitions for the ensemble trading pipeline.

This module contains fundamental constants including:
- Random seed for reproducibility
- Project directory paths
- Symbol configurations
- Timeframe configurations
"""

from pathlib import Path

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across the entire pipeline.

    This function sets seeds for:
    - Python's random module
    - NumPy's random module
    - PyTorch (if available)
    - TensorFlow (if available)

    Call this at the start of any pipeline run to ensure reproducible results.

    Parameters
    ----------
    seed : int
        Random seed value (default: RANDOM_SEED = 42)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Try to set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # For full determinism (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Try to set TensorFlow seeds if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# =============================================================================
# PROJECT PATHS
# =============================================================================
# Base paths - computed relative to this file's location
# This file is at src/config/constants.py, so PROJECT_ROOT is two levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"
SPLITS_DIR = DATA_DIR / "splits"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base"
ENSEMBLE_MODELS_DIR = MODELS_DIR / "ensemble"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_directories() -> None:
    """
    Create all required directories if they don't exist.

    This function should be called explicitly when needed, rather than
    at module import time, to avoid side effects during imports.
    """
    for dir_path in [
        RAW_DATA_DIR, CLEAN_DATA_DIR, FEATURES_DIR, PROCESSED_DATA_DIR,
        FINAL_DATA_DIR, SPLITS_DIR, BASE_MODELS_DIR, ENSEMBLE_MODELS_DIR,
        RESULTS_DIR, REPORTS_DIR, LOGS_DIR
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRADING SYMBOLS
# =============================================================================
# MES (S&P 500 futures) and MGC (Gold futures) for cross-asset analysis
SYMBOLS = ['MES', 'MGC']


# =============================================================================
# MULTI-TIMEFRAME (MTF) CONFIGURATION
# =============================================================================
# Supported timeframes for OHLCV resampling.
# Input data is assumed to be 1-minute bars, which can be resampled to any of these.
SUPPORTED_TIMEFRAMES = ['1min', '5min', '10min', '15min', '20min', '30min', '45min', '60min']

# Default target timeframe for resampling.
# This is the primary resolution used for feature calculation and model training.
TARGET_TIMEFRAME = '5min'

# Legacy alias for backward compatibility
BAR_RESOLUTION = TARGET_TIMEFRAME


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """
    Parse a timeframe string to minutes.

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '5min', '1h', '1d')

    Returns
    -------
    int
        Number of minutes

    Raises
    ------
    ValueError
        If timeframe format is invalid
    """
    timeframe = timeframe.lower().strip()

    if 'min' in timeframe:
        try:
            return int(timeframe.replace('min', ''))
        except ValueError:
            raise ValueError(f"Invalid timeframe format: '{timeframe}'. Expected format like '5min'")
    elif 'h' in timeframe:
        try:
            return int(timeframe.replace('h', '')) * 60
        except ValueError:
            raise ValueError(f"Invalid timeframe format: '{timeframe}'. Expected format like '1h'")
    elif 'd' in timeframe:
        try:
            return int(timeframe.replace('d', '')) * 60 * 24
        except ValueError:
            raise ValueError(f"Invalid timeframe format: '{timeframe}'. Expected format like '1d'")
    else:
        raise ValueError(
            f"Invalid timeframe format: '{timeframe}'. "
            f"Expected formats: '5min', '1h', '1d'"
        )


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate that a timeframe is supported.

    Parameters
    ----------
    timeframe : str
        Timeframe string to validate

    Returns
    -------
    bool
        True if timeframe is supported

    Raises
    ------
    ValueError
        If timeframe is not supported
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: '{timeframe}'. "
            f"Supported timeframes are: {SUPPORTED_TIMEFRAMES}"
        )
    return True


def get_timeframe_metadata(timeframe: str) -> dict:
    """
    Get metadata for a given timeframe.

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '5min', '15min')

    Returns
    -------
    dict
        Metadata including minutes, bars_per_hour, bars_per_day, description
    """
    validate_timeframe(timeframe)
    minutes = parse_timeframe_to_minutes(timeframe)

    # Trading day = 6.5 hours for US markets (9:30 AM - 4:00 PM)
    # For futures (MES, MGC), trading is nearly 24 hours but we use standard day
    trading_minutes_per_day = 390  # 6.5 hours * 60

    return {
        'timeframe': timeframe,
        'minutes': minutes,
        'bars_per_hour': 60 // minutes if minutes <= 60 else 60 / minutes,
        'bars_per_trading_day': trading_minutes_per_day // minutes,
        'bars_per_24h': (24 * 60) // minutes,
        'description': f'{minutes}-minute OHLCV bars',
        'pandas_freq': f'{minutes}min' if minutes < 60 else f'{minutes // 60}h',
    }
