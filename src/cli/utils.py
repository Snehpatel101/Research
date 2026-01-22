"""
CLI Utilities - Shared functions for CLI commands.

This module provides shared utilities for all CLI commands including:
- Display helpers (errors, warnings, success messages)
- Argument parsing helpers (parse_model_list, parse_horizon_list)
- Logging setup
- Project path resolution
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console

console = Console()

# =============================================================================
# PROJECT ROOT AND PATHS
# =============================================================================

# Get project root from file location (src/cli/utils.py -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Default paths used across CLI commands
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "splits" / "scaled"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_STACKING_OUTPUT_DIR = PROJECT_ROOT / "data" / "stacking"
DEFAULT_WALK_FORWARD_OUTPUT_DIR = PROJECT_ROOT / "data" / "walk_forward"
DEFAULT_CPCV_OUTPUT_DIR = PROJECT_ROOT / "data" / "cpcv_pbo"

# Default horizons for evaluation commands
DEFAULT_HORIZONS = [5, 10, 15, 20]


def get_project_root(project_root: str | None = None) -> Path:
    """
    Get the project root path.

    Args:
        project_root: Optional path to project root. If None, derives from this file's location.

    Returns:
        Path to project root directory.
    """
    if project_root is None:
        return PROJECT_ROOT
    return Path(project_root)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def show_error(message: str) -> None:
    """Display error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def show_success(message: str) -> None:
    """Display success message."""
    console.print(f"[bold green]Success:[/bold green] {message}")


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


def validate_data_dir(data_dir: Path) -> bool:
    """
    Validate that a data directory exists.

    Args:
        data_dir: Path to data directory.

    Returns:
        True if valid, False otherwise.
    """
    if not data_dir.exists():
        show_error(f"Data directory not found: {data_dir}")
        show_info("Run the data pipeline first to generate scaled data")
        return False
    return True


# =============================================================================
# ID GENERATION
# =============================================================================


def generate_run_id() -> str:
    """
    Generate unique run ID for output directories.

    Format: {timestamp_with_ms}_{random_suffix}
    Example: 20251228_143025_789456_a3f9

    Returns:
        Unique run ID string.
    """
    import secrets
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = secrets.token_hex(2)  # 2 bytes = 4 hex chars
    return f"{timestamp}_{random_suffix}"
