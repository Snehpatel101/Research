"""
CLI Utilities - Shared functions for CLI commands.
"""
from pathlib import Path

from rich.console import Console

console = Console()


def show_error(message: str) -> None:
    """Display error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def show_success(message: str) -> None:
    """Display success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def show_info(message: str) -> None:
    """Display info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def show_warning(message: str) -> None:
    """Display warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def get_project_root(project_root: str | None = None) -> Path:
    """
    Get the project root path.

    Args:
        project_root: Optional path to project root. If None, derives from this file's location.

    Returns:
        Path to project root directory.
    """
    if project_root is None:
        # src/cli/utils.py -> src/cli -> src -> project_root
        return Path(__file__).parent.parent.parent.resolve()
    return Path(project_root)
