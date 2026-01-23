"""
Unified CLI for ML Pipeline.

This is the main entry point that provides a Typer-based CLI with all subcommands:
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
    python -m src.cli train model --help
"""

from __future__ import annotations

import typer

from src.cli.commands.evaluate import evaluate_app
from src.cli.commands.pipeline import pipeline_app
from src.cli.commands.train import train_app

# Create the main CLI app
app = typer.Typer(
    name="ml",
    help="Unified ML Pipeline - Train and evaluate ML models on OHLCV data",
    no_args_is_help=True,
    add_completion=False,
)

# =============================================================================
# REGISTER PIPELINE COMMANDS (run, data, status, resume)
# =============================================================================

# Register individual commands from pipeline_app directly on main app
app.command("run", help="Run full pipeline (data + training + evaluation)")(
    pipeline_app.registered_commands[0].callback  # type: ignore[type-var]
)
app.command("data", help="Run data pipeline only")(pipeline_app.registered_commands[1].callback)  # type: ignore[type-var]
app.command("status", help="Show pipeline status")(pipeline_app.registered_commands[2].callback)  # type: ignore[type-var]
app.command("resume", help="Resume pipeline from checkpoint")(
    pipeline_app.registered_commands[3].callback  # type: ignore[type-var]
)

# =============================================================================
# REGISTER TRAIN COMMANDS (train, train-ensemble)
# =============================================================================

# Add train as a sub-app with its commands
app.add_typer(train_app, name="train", help="Model training commands")

# =============================================================================
# REGISTER EVALUATE COMMANDS (cv, walk-forward, cpcv-pbo)
# =============================================================================

# Register evaluation commands directly on main app for convenience
app.command("cv", help="Run cross-validation")(evaluate_app.registered_commands[0].callback)  # type: ignore[type-var]
app.command("walk-forward", help="Run walk-forward evaluation")(
    evaluate_app.registered_commands[1].callback  # type: ignore[type-var]
)
app.command("cpcv-pbo", help="Run CPCV/PBO evaluation")(
    evaluate_app.registered_commands[2].callback  # type: ignore[type-var]
)


# =============================================================================
# VERSION COMMAND
# =============================================================================


@app.command("version")
def show_version():
    """Show version information."""
    from rich.console import Console

    console = Console()
    console.print("[bold]ML Pipeline CLI[/bold]")
    console.print("Version: 1.0.0")
    console.print("Python: 3.12+")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
