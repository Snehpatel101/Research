"""
Runtime defaults and helper utilities for Phase 1 configuration.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.common.horizon_config import HORIZONS, auto_scale_purge_embargo
from src.phase1.config.barriers_config import BARRIER_PARAMS, BARRIER_PARAMS_DEFAULT
from src.phase1.config.features import parse_timeframe_to_minutes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"


def detect_available_symbols(raw_dir: Path = None) -> list[str]:
    """
    Auto-detect available symbols from raw data directory.

    Looks for parquet files matching pattern: {SYMBOL}_1m.parquet

    Returns
    -------
    List[str]
        List of detected symbol names, sorted alphabetically
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR

    if not raw_dir.exists():
        return []

    symbols = []
    for f in raw_dir.glob("*_1m.parquet"):
        # Extract symbol from filename (e.g., "MES_1m.parquet" -> "MES")
        symbol = f.stem.replace("_1m", "")
        if symbol:
            symbols.append(symbol)

    return sorted(symbols)


# Auto-detect symbols from available data (empty list if no data)
SYMBOLS = detect_available_symbols()
TARGET_TIMEFRAME = "1min"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

PURGE_BARS, EMBARGO_BARS = auto_scale_purge_embargo(HORIZONS)


def set_global_seeds(seed: int) -> None:
    """Set random seeds across common RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def validate_config() -> None:
    """Validate core configuration settings and raise on issues."""
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    max_max_bars = 0
    for horizons in BARRIER_PARAMS.values():
        for params in horizons.values():
            max_max_bars = max(max_max_bars, params.get("max_bars", 0))
    for params in BARRIER_PARAMS_DEFAULT.values():
        max_max_bars = max(max_max_bars, params.get("max_bars", 0))

    if PURGE_BARS < max_max_bars:
        raise ValueError(
            f"PURGE_BARS ({PURGE_BARS}) < max_bars ({max_max_bars}) - label leakage risk"
        )


def get_timeframe_metadata(timeframe: str) -> dict[str, float | str]:
    """Return basic metadata for a timeframe string."""
    minutes = parse_timeframe_to_minutes(timeframe)
    bars_per_hour = 60 / minutes
    if minutes < 60:
        description = f"{minutes:g} minute bars"
    else:
        hours = minutes / 60
        description = f"{hours:g} hour bars"

    return {
        "timeframe": timeframe,
        "minutes": minutes,
        "bars_per_hour": bars_per_hour,
        "description": description,
    }


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "RUNS_DIR",
    "CONFIG_DIR",
    "SYMBOLS",
    "TARGET_TIMEFRAME",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",
    "RANDOM_SEED",
    "PURGE_BARS",
    "EMBARGO_BARS",
    "set_global_seeds",
    "validate_config",
    "get_timeframe_metadata",
    "detect_available_symbols",
]
