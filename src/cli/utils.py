"""
CLI Utilities - Shared functions for CLI commands.

This module provides shared utilities for all CLI commands including:
- Display helpers (errors, warnings, success messages)
- Argument parsing helpers (parse_model_list, parse_horizon_list)
- Logging setup
- ExperimentConfig construction from CLI arguments (one config class for every command)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from src.config.experiment import ExperimentConfig

console = Console()

# =============================================================================
# DEFAULTS
# =============================================================================

# Root of experiment output; each run writes into ``<output_dir>/<run_id>/``
DEFAULT_OUTPUT_DIR = Path("experiments")

# Default horizons for evaluation commands
DEFAULT_HORIZONS = [5, 10, 15, 20]


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def show_error(message: str) -> None:
    """Display error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def show_info(message: str) -> None:
    """Display info message."""
    console.print(f"[bold blue]Info:[/bold blue] {message}")


def show_warning(message: str) -> None:
    """Display warning message."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for CLI commands.

    Args:
        verbose: If True, sets log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# =============================================================================
# ARGUMENT PARSING HELPERS
# =============================================================================


def parse_model_list(model_arg: str) -> list[str]:
    """
    Parse model argument into list of model names.

    Args:
        model_arg: Comma-separated model names or 'all' for all registered models.

    Returns:
        List of validated model names.

    Raises:
        SystemExit: If any model name is invalid.
    """
    # Import here to avoid circular imports
    import src.models  # noqa: F401 - ensures models are registered
    from src.models.registry import ModelRegistry

    if model_arg.lower() == "all":
        return ModelRegistry.list_all()

    models = [m.strip().lower() for m in model_arg.split(",") if m.strip()]

    # Validate models exist
    available = ModelRegistry.list_all()
    invalid = [m for m in models if m not in available]
    if invalid:
        show_error(f"Unknown models: {invalid}")
        show_info(f"Available models: {available}")
        sys.exit(1)

    return models


def parse_horizon_list(horizon_arg: str) -> list[int]:
    """
    Parse horizon argument into list of integers.

    Args:
        horizon_arg: Comma-separated horizons or 'all' for default horizons.

    Returns:
        List of horizon integers.

    Raises:
        SystemExit: If horizon format is invalid.
    """
    if horizon_arg.lower() == "all":
        return DEFAULT_HORIZONS.copy()

    try:
        return [int(h.strip()) for h in horizon_arg.split(",") if h.strip()]
    except ValueError as e:
        show_error(f"Invalid horizon format: {e}")
        show_info("Horizons should be comma-separated integers (e.g., '5,10,15,20')")
        sys.exit(1)


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================


def build_experiment_config(
    *,
    data_path: Path,
    output_dir: Path,
    symbol: str,
    horizons: list[int],
    models: list[str],
    bar_timeframe: str | None = None,
    mtf: bool = True,
    training_mode: str = "standard",
    build_ensemble: bool = False,
    meta_learner: str = "ridge_meta",
    n_trials: int = 0,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    n_splits: int | None = None,
    purge_bars: int | None = None,
    embargo_bars: int | None = None,
    run_backtest: bool = False,
    deploy: bool = True,
    name: str | None = None,
) -> ExperimentConfig:
    """
    Build the ExperimentConfig every CLI command runs on.

    ``None`` for purge/embargo/epochs/batch size/splits keeps the config
    defaults (purge and embargo are derived from the label span and the bar
    timeframe).
    """
    from src.config.experiment import (
        BundlingSection,
        DataSection,
        EvaluationSection,
        ExperimentConfig,
        TrainingSection,
    )
    from src.config.training import OptunaConfig

    data = DataSection(symbol=symbol, data_path=data_path, bar_timeframe=bar_timeframe)
    data.mtf.enabled = mtf

    training = TrainingSection(
        models=models,
        horizons=horizons,
        training_mode=training_mode,
        build_ensemble=build_ensemble,
        meta_learner=meta_learner,
        optuna=OptunaConfig(n_trials=n_trials),
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    if max_epochs is not None:
        training.max_epochs = max_epochs
    if batch_size is not None:
        training.batch_size = batch_size
    if n_splits is not None:
        training.n_splits = n_splits

    return ExperimentConfig(
        name=name or f"{symbol}_pipeline",
        output_dir=output_dir,
        data=data,
        training=training,
        evaluation=EvaluationSection(run_backtest=run_backtest),
        bundling=BundlingSection(create_bundle=deploy, deploy_artifact=deploy),
    )
