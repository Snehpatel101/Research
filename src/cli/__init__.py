"""
CLI Module - Typer-based command-line interface for pipeline management.

This module provides the main CLI application and aggregates commands from submodules.
"""
import typer

from .preset_commands import presets_command
from .run_commands import models_command, rerun_command, run_command
from .status_commands import (
    clean_command,
    compare_command,
    list_runs_command,
    status_command,
    validate_command,
)
from .utils import console, show_error, show_info, show_success, show_warning

# Create main app
app = typer.Typer(
    name="pipeline",
    help="Phase 1 Data Preparation Pipeline CLI",
    add_completion=False
)

# Register commands
app.command(name="run")(run_command)
app.command(name="rerun")(rerun_command)
app.command(name="status")(status_command)
app.command(name="validate")(validate_command)
app.command(name="list-runs")(list_runs_command)
app.command(name="compare")(compare_command)
app.command(name="clean")(clean_command)
app.command(name="presets")(presets_command)
app.command(name="models")(models_command)


def main() -> None:
    """Main entry point for the CLI."""
    app()


__all__ = [
    "app",
    "main",
    "console",
    "show_error",
    "show_success",
    "show_info",
    "show_warning",
]
