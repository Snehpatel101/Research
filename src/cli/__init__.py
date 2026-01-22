"""
Unified CLI for ML pipeline.

Provides a Typer-based command-line interface for the complete ML workflow:
- ml run: Full pipeline (data + training + evaluation)
- ml data: Data pipeline only
- ml train model: Train model(s)
- ml train ensemble: Train ensemble
- ml cv: Cross-validation
- ml walk-forward: Walk-forward evaluation
- ml cpcv-pbo: CPCV/PBO evaluation
- ml status: Show pipeline status
- ml resume: Resume from checkpoint

Usage:
    python -m src.cli --help
    python -m src.cli run --help
"""

from .unified_cli import main, app

__all__ = ["main", "app"]
