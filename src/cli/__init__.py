"""
Unified CLI for ML Factory.

Typer-based command line over MLFactory (raw OHLCV in, models and deploy artifacts out):
- ml run: full pipeline (features, labels, training, ensemble, backtest, deploy)
- ml data: features + labels to parquet
- ml status: checkpoint progress of a run
- ml models: registered models
- ml cv / walk-forward / cpcv-pbo: standalone evaluation of tabular models

Usage:
    python -m src.cli --help
    python -m src.cli run --help
"""

from .unified_cli import app, main

__all__ = ["main", "app"]
