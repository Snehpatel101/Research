#!/usr/bin/env python3
"""
Cross-Validation Runner CLI.

This is a backward-compatible wrapper that delegates to the unified CLI.
For full documentation, run: python -m src.cli cv --help

Run purged k-fold cross-validation for Phase 3.
Generates out-of-fold predictions for ensemble stacking in Phase 4.

Usage:
    python scripts/run_cv.py --models xgboost,lightgbm --horizons 5,10,20
    python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 100
    python scripts/run_cv.py --models all --horizons all --no-feature-selection

Requirements:
    - Phase 1 data in data/splits/scaled/
    - Phase 2 models registered in ModelRegistry
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.commands.evaluate import run_cv


def main() -> int:
    """Main entry point - delegates to CLI cv command."""
    run_cv()
    return 0


if __name__ == "__main__":
    sys.exit(main())
