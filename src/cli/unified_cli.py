"""
Unified CLI for ML Factory.

One pipeline (MLFactory) behind every command; raw OHLCV bars in, no intermediate
stage directories:

- ml run: features, labels, training, ensemble, backtest, deploy (`--resume` a run)
- ml data: features + labels to parquet only
- ml status: checkpoint progress of a run
- ml models: registered models
- ml cv: purged k-fold cross-validation (tabular models)
- ml walk-forward: walk-forward evaluation (tabular models)
- ml cpcv-pbo: CPCV backtest paths + PBO overfitting gate

Usage:
    python -m src.cli --help
    python -m src.cli run --help
"""

from __future__ import annotations

import typer

from src.cli.commands.evaluate import run_cpcv_pbo, run_cv, run_walk_forward
from src.cli.commands.pipeline import list_models, run_data, run_pipeline, show_status
from src.cli.utils import console

app = typer.Typer(
    name="ml",
    help="ML Factory - train and evaluate ML models on raw OHLCV data",
    no_args_is_help=True,
    add_completion=False,
)

app.command("run", help="Run the full pipeline (features, labels, training, ensemble, deploy)")(
    run_pipeline
)
app.command("data", help="Compute features and labels to parquet (no training)")(run_data)
app.command("status", help="Show checkpoint progress of a run")(show_status)
app.command("models", help="List registered models or show one model's configuration")(list_models)
app.command("cv", help="Purged k-fold cross-validation of tabular models")(run_cv)
app.command("walk-forward", help="Walk-forward evaluation of tabular models")(run_walk_forward)
app.command("cpcv-pbo", help="CPCV backtest paths and PBO overfitting gate")(run_cpcv_pbo)


@app.command("version")
def show_version():
    """Show version information."""
    console.print("[bold]ML Factory CLI[/bold]")
    console.print("Version: 1.0.0")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
