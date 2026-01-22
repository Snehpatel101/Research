#!/usr/bin/env python3
"""
Walk-Forward Evaluation CLI.

This is a backward-compatible wrapper that delegates to the unified CLI.
For full documentation, run: python -m src.cli walk-forward --help

Runs walk-forward analysis on trained models to evaluate performance
across expanding or rolling time windows. More realistic than k-fold
for trading applications as it respects temporal ordering.

Usage:
    # Basic walk-forward with 5 windows
    python scripts/run_walk_forward.py --models xgboost --horizons 20

    # Rolling window with custom settings
    python scripts/run_walk_forward.py --models xgboost,lightgbm \
        --window-type rolling --n-windows 10 --min-train-pct 0.3

    # All models, all horizons
    python scripts/run_walk_forward.py --models all --horizons all
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.commands.evaluate import run_walk_forward


def main() -> int:
    """Main entry point - delegates to CLI walk-forward command."""
    run_walk_forward()
    return 0


if __name__ == "__main__":
    sys.exit(main())
