#!/usr/bin/env python3
"""
Train heterogeneous stacking ensembles from the Model Factory.

This is a backward-compatible wrapper that delegates to the unified CLI.
For full documentation, run: python -m src.cli train ensemble --help

Examples:
    # Standard 3-base heterogeneous ensemble
    python scripts/train_ensemble.py --base-models catboost,tcn,patchtst \
        --meta-learner logistic --horizon 20

    # 4-base ensemble with custom meta-learner
    python scripts/train_ensemble.py --base-models xgboost,lightgbm,lstm,transformer \
        --meta-learner ridge_meta --horizon 20

    # Minimal 2-base ensemble for quick testing
    python scripts/train_ensemble.py --base-models xgboost,lstm \
        --meta-learner logistic --horizon 20 --n-folds 3

    # List available base models and meta-learners
    python scripts/train_ensemble.py --list-models

Usage:
    python scripts/train_ensemble.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.commands.train import train_app  # noqa: E402


def main() -> int:
    """Main entry point - delegates to CLI train ensemble command."""
    # Run the train_app with "ensemble" as the subcommand
    # Insert "ensemble" as the first argument
    sys.argv.insert(1, "ensemble")

    train_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
