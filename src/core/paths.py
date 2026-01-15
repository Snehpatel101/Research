"""
Unified paths module - single source of truth for all path constants.

CONFIG-001: This module consolidates paths from:
- src/models/config/paths.py
- src/phase1/config/runtime.py

All path definitions should reference this module.
"""

from pathlib import Path

# =============================================================================
# PROJECT ROOT AND DATA DIRECTORIES
# =============================================================================

# 2 levels up from src/core/paths.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "runs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# =============================================================================
# CONFIG DIRECTORIES
# =============================================================================

CONFIG_ROOT = PROJECT_ROOT / "config"
CONFIG_MODELS_DIR = CONFIG_ROOT / "models"
CONFIG_PIPELINE_DIR = CONFIG_ROOT / "pipeline"

# For backward compatibility with models/config/paths.py
CONFIG_DIR = CONFIG_MODELS_DIR  # Alias for models config directory

# =============================================================================
# SPECIFIC CONFIG FILES
# =============================================================================

TRAINING_CONFIG_PATH = CONFIG_PIPELINE_DIR / "training.yaml"
CV_CONFIG_PATH = CONFIG_PIPELINE_DIR / "cv.yaml"

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "RUNS_DIR",
    "EXPERIMENTS_DIR",
    "CONFIG_ROOT",
    "CONFIG_MODELS_DIR",
    "CONFIG_PIPELINE_DIR",
    "CONFIG_DIR",
    "TRAINING_CONFIG_PATH",
    "CV_CONFIG_PATH",
]
