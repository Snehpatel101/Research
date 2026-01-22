#!/usr/bin/env python3
"""
CPCV and PBO Evaluation CLI.

This is a backward-compatible wrapper that delegates to the unified CLI.
For full documentation, run: python -m src.cli cpcv-pbo --help

Runs Combinatorially Purged Cross-Validation (CPCV) and computes
Probability of Backtest Overfitting (PBO) for model selection gating.

Usage:
    # Basic CPCV with PBO computation
    python scripts/run_cpcv_pbo.py --models xgboost,lightgbm --horizons 20

    # More groups for higher resolution
    python scripts/run_cpcv_pbo.py --models all --n-groups 8 --n-test-groups 2

    # Strict PBO thresholds
    python scripts/run_cpcv_pbo.py --models xgboost --pbo-warn 0.4 --pbo-block 0.7
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.commands.evaluate import run_cpcv_pbo  # noqa: E402


def main() -> int:
    """Main entry point - delegates to CLI cpcv-pbo command."""
    run_cpcv_pbo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
