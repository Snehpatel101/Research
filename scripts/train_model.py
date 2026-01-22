#!/usr/bin/env python3
"""
Train any registered model from the Model Factory.

This is a backward-compatible wrapper that delegates to the unified CLI.
For full documentation, run: python -m src.cli train model --help

Examples:
    # Train XGBoost with defaults
    python scripts/train_model.py --model xgboost --horizon 20

    # Train a voting ensemble from scratch
    python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost --horizon 20

    # Train LSTM with custom config
    python scripts/train_model.py --model lstm --horizon 20 --hidden-size 256 --seq-len 60

    # Train TCN with longer sequences
    python scripts/train_model.py --model tcn --horizon 20 --seq-len 120

    # Train with custom data path
    python scripts/train_model.py --model xgboost --horizon 20 --data-dir data/splits/scaled

    # List available models
    python scripts/train_model.py --list-models

Usage:
    python scripts/train_model.py --help
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.commands.train import train_app


def main() -> int:
    """Main entry point - delegates to CLI train model command."""
    # Run the train_app with "model" as the subcommand
    # We need to inject "model" as the first argument if not present
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        # User didn't specify a subcommand, default to "model"
        sys.argv.insert(1, "model")
    elif sys.argv[1] not in ["model", "ensemble"]:
        # User specified flags directly, insert "model" subcommand
        sys.argv.insert(1, "model")

    train_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
