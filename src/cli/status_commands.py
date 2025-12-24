"""
CLI Status Commands - status, list, validate, compare, and clean commands.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .utils import console, show_error, show_success, show_info, show_warning, get_project_root

# Lazy imports to avoid circular dependencies
_pipeline_config = None
_manifest = None


def _get_pipeline_config():
    """Lazy import pipeline_config module."""
    global _pipeline_config
    if _pipeline_config is None:
        from .. import pipeline_config
        _pipeline_config = pipeline_config
    return _pipeline_config


def _get_manifest():
    """Lazy import manifest module."""
    global _manifest
    if _manifest is None:
        from .. import manifest
        _manifest = manifest
    return _manifest


def status_command(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to check"
    ),
    project_root: Optional[str] = typer.Option(
        None,
        "--project-root",
        help="Project root directory"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information"
    )
) -> None:
    """
    Check the status of a pipeline run.

    Examples:
        pipeline status 20241218_120000
        pipeline status phase1_v1 --verbose
    """
    pipeline_config = _get_pipeline_config()
    manifest_mod = _get_manifest()

    console.print(f"\n[bold cyan]Pipeline Run Status: {run_id}[/bold cyan]\n")

    project_path = get_project_root(project_root)
    run_dir = project_path / "runs" / run_id

    # Check if run exists
    if not run_dir.exists():
        show_error(f"Run not found: {run_id}")
        raise typer.Exit(1)

    # Load configuration
    try:
        config = pipeline_config.PipelineConfig.load_from_run_id(run_id, project_path)
    except FileNotFoundError:
        show_error("Configuration not found for this run")
        raise typer.Exit(1)

    # Display basic info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Key", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Run ID", config.run_id)
    info_table.add_row("Description", config.description)
    info_table.add_row("Symbols", ", ".join(config.symbols))
    info_table.add_row("Date Range", f"{config.start_date or 'N/A'} to {config.end_date or 'N/A'}")

    console.print(info_table)
    console.print()

    # Load and display pipeline state
    _display_pipeline_state(run_dir)

    # Load manifest
    _display_manifest_info(run_id, project_path, manifest_mod, verbose)

    # Show paths
    if verbose:
        console.print(f"[bold]Paths:[/bold]")
        console.print(f"  Run directory: {run_dir}")
        console.print(f"  Config: {run_dir / 'config' / 'config.json'}")
        console.print(f"  Logs: {run_dir / 'logs' / 'pipeline.log'}")
        console.print(f"  Manifest: {run_dir / 'artifacts' / 'manifest.json'}")


def _display_pipeline_state(run_dir: Path) -> None:
    """Display pipeline stage information from state file."""
    state_path = run_dir / "artifacts" / "pipeline_state.json"
    if not state_path.exists():
        show_warning("No pipeline state found. Run may not have started.")
        return

    with open(state_path, 'r') as f:
        state = json.load(f)

    completed_stages = set(state.get('completed_stages', []))
    stage_results = state.get('stage_results', {})

    # Create stages table
    stages_table = Table(title="Pipeline Stages", show_header=True)
    stages_table.add_column("Stage", style="cyan")
    stages_table.add_column("Status", style="white")
    stages_table.add_column("Duration", style="green")
    stages_table.add_column("Artifacts", style="yellow")

    try:
        from src.pipeline.stage_registry import get_stage_definitions
        stage_defs = get_stage_definitions()
        all_stages = [stage["name"] for stage in stage_defs]
    except Exception:
        # Fallback for legacy runs if stage registry is unavailable
        all_stages = list(stage_results.keys())

    for stage in all_stages:
        if stage in stage_results:
            result = stage_results[stage]
            status = result['status']

            # Status with emoji
            if status == 'completed':
                status_str = "[green]✓ Completed[/green]"
            elif status == 'failed':
                status_str = "[red]✗ Failed[/red]"
            elif status == 'in_progress':
                status_str = "[yellow]⋯ In Progress[/yellow]"
            else:
                status_str = f"[dim]{status}[/dim]"

            duration = f"{result.get('duration_seconds', 0):.2f}s"
            artifacts = str(len(result.get('artifacts', [])))

            stages_table.add_row(
                stage.replace('_', ' ').title(),
                status_str,
                duration,
                artifacts
            )
        else:
            stages_table.add_row(
                stage.replace('_', ' ').title(),
                "[dim]Pending[/dim]",
                "-",
                "-"
            )

    console.print(stages_table)
    console.print()

    # Overall progress
    total_stages = len(all_stages)
    if total_stages == 0:
        show_warning("No stages available to display.")
        return

    completed = len([stage for stage in all_stages if stage in completed_stages])
    progress_pct = (completed / total_stages) * 100

    console.print(f"[bold]Progress:[/bold] {completed}/{total_stages} stages ({progress_pct:.0f}%)")
    console.print()


def _display_manifest_info(run_id: str, project_path: Path, manifest_mod, verbose: bool) -> None:
    """Display manifest and artifact information."""
    try:
        manifest = manifest_mod.ArtifactManifest.load(run_id, project_path)
        summary = manifest.get_summary()

        console.print(f"[bold]Artifacts:[/bold]")
        console.print(f"  Total: {summary['total_artifacts']}")
        console.print(f"  Size: {summary['total_size_mb']:.2f} MB")
        console.print()

        if verbose:
            # Show artifact details
            artifacts_table = Table(title="Artifacts", show_header=True)
            artifacts_table.add_column("Name", style="cyan")
            artifacts_table.add_column("Stage", style="yellow")
            artifacts_table.add_column("Size", style="green")
            artifacts_table.add_column("Exists", style="white")

            for name, artifact in manifest.artifacts.items():
                size_mb = artifact.get('size_bytes', 0) / (1024 * 1024)
                exists = "✓" if artifact.get('exists', False) else "✗"

                artifacts_table.add_row(
                    name,
                    artifact.get('stage', 'unknown'),
                    f"{size_mb:.2f} MB",
                    exists
                )

            console.print(artifacts_table)
            console.print()

    except FileNotFoundError:
        show_warning("No manifest found for this run")


def validate_command(
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Run ID to validate (validates new config if not provided)"
    ),
    symbols: str = typer.Option(
        "MES,MGC",
        "--symbols",
        help="Comma-separated list of symbols"
    ),
    project_root: Optional[str] = typer.Option(
        None,
        "--project-root",
        help="Project root directory"
    )
) -> None:
    """
    Validate pipeline configuration and data integrity.

    Examples:
        pipeline validate --symbols MES,MGC,MNQ
        pipeline validate --run-id 20241218_120000
    """
    pipeline_config = _get_pipeline_config()
    manifest_mod = _get_manifest()

    console.print("\n[bold cyan]Pipeline Validation[/bold cyan]\n")

    project_root_path = get_project_root(project_root)

    # Load or create configuration
    if run_id:
        try:
            config = pipeline_config.PipelineConfig.load_from_run_id(run_id, project_root_path)
            show_info(f"Validating configuration for run: {run_id}")
        except FileNotFoundError:
            show_error(f"Run not found: {run_id}")
            raise typer.Exit(1)
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        config = pipeline_config.create_default_config(
            symbols=symbol_list,
            project_root=project_root_path
        )
        show_info("Validating new configuration")

    console.print()

    # Validate configuration
    issues = config.validate()

    if issues:
        show_error("Configuration validation failed:")
        for issue in issues:
            console.print(f"  • [red]{issue}[/red]")
        console.print()
        raise typer.Exit(1)
    else:
        show_success("Configuration is valid")
        console.print()

    # If run_id provided, validate artifacts
    if run_id:
        try:
            manifest = manifest_mod.ArtifactManifest.load(run_id, project_root_path)
            show_info("Validating artifacts...")

            verification = manifest.verify_all_artifacts()
            valid_count = sum(verification.values())
            total_count = len(verification)

            if valid_count == total_count:
                show_success(f"All {total_count} artifacts are valid")
            else:
                show_warning(f"Only {valid_count}/{total_count} artifacts are valid")

                # Show failed artifacts
                failed = [name for name, valid in verification.items() if not valid]
                if failed:
                    console.print("\n[bold]Failed Artifacts:[/bold]")
                    for name in failed:
                        console.print(f"  • [red]{name}[/red]")

        except FileNotFoundError:
            show_warning("No manifest found for this run")

    console.print()
    show_success("Validation complete")


def list_runs_command(
    project_root: Optional[str] = typer.Option(
        None,
        "--project-root",
        help="Project root directory"
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Number of runs to show"
    )
) -> None:
    """
    List all pipeline runs.

    Examples:
        pipeline list-runs
        pipeline list-runs --limit 20
    """
    pipeline_config = _get_pipeline_config()

    console.print("\n[bold cyan]Pipeline Runs[/bold cyan]\n")

    project_root_path = get_project_root(project_root)
    runs_dir = project_root_path / "runs"

    if not runs_dir.exists():
        show_warning("No runs directory found")
        return

    # Get all run directories
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:limit]

    if not run_dirs:
        show_warning("No runs found")
        return

    # Create table
    table = Table(show_header=True)
    table.add_column("Run ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Symbols", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Created", style="dim")

    for run_dir in run_dirs:
        run_id = run_dir.name

        # Try to load config
        try:
            config = pipeline_config.PipelineConfig.load_from_run_id(run_id, project_root_path)
            description = config.description
            symbols = ", ".join(config.symbols)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
            # Handle specific exceptions that can occur when loading config:
            # - FileNotFoundError: config file doesn't exist
            # - JSONDecodeError: config file is malformed
            # - KeyError/TypeError: config missing required fields
            description = "N/A"
            symbols = "N/A"

        # Check status
        state_path = run_dir / "artifacts" / "pipeline_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            completed = len(state.get('completed_stages', []))
            # Get total from stage registry
            try:
                from src.pipeline.stage_registry import get_stage_definitions
                total = len(get_stage_definitions())
            except Exception:
                total = 12  # Fallback to known count
            status = f"{completed}/{total} stages"
        else:
            status = "Not started"

        # Get creation time
        created = datetime.fromtimestamp(run_dir.stat().st_ctime)
        created_str = created.strftime("%Y-%m-%d %H:%M")

        table.add_row(run_id, description, symbols, status, created_str)

    console.print(table)
    console.print(f"\n[dim]Showing {len(run_dirs)} most recent runs[/dim]")


def compare_command(
    run1: str = typer.Argument(..., help="First run ID"),
    run2: str = typer.Argument(..., help="Second run ID"),
    project_root: Optional[str] = typer.Option(
        None,
        "--project-root",
        help="Project root directory"
    )
) -> None:
    """
    Compare artifacts between two runs.

    Examples:
        pipeline compare 20241218_120000 20241218_130000
    """
    manifest_mod = _get_manifest()

    console.print(f"\n[bold cyan]Comparing Runs: {run1} vs {run2}[/bold cyan]\n")

    project_root_path = get_project_root(project_root)

    try:
        comparison = manifest_mod.compare_runs(run1, run2, project_root_path)

        # Display results
        console.print(f"[bold]Added Artifacts:[/bold] {len(comparison['added'])}")
        if comparison['added']:
            for artifact in comparison['added'][:10]:
                console.print(f"  + [green]{artifact}[/green]")
            if len(comparison['added']) > 10:
                console.print(f"  [dim]... and {len(comparison['added']) - 10} more[/dim]")
        console.print()

        console.print(f"[bold]Removed Artifacts:[/bold] {len(comparison['removed'])}")
        if comparison['removed']:
            for artifact in comparison['removed'][:10]:
                console.print(f"  - [red]{artifact}[/red]")
            if len(comparison['removed']) > 10:
                console.print(f"  [dim]... and {len(comparison['removed']) - 10} more[/dim]")
        console.print()

        console.print(f"[bold]Modified Artifacts:[/bold] {len(comparison['modified'])}")
        if comparison['modified']:
            for artifact in comparison['modified'][:10]:
                console.print(f"  ~ [yellow]{artifact}[/yellow]")
            if len(comparison['modified']) > 10:
                console.print(f"  [dim]... and {len(comparison['modified']) - 10} more[/dim]")
        console.print()

        console.print(f"[bold]Unchanged Artifacts:[/bold] {len(comparison['unchanged'])}")

    except Exception as e:
        show_error(f"Comparison failed: {e}")
        raise typer.Exit(1)


def clean_command(
    run_id: str = typer.Argument(..., help="Run ID to clean"),
    project_root: Optional[str] = typer.Option(
        None,
        "--project-root",
        help="Project root directory"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation"
    )
) -> None:
    """
    Delete a pipeline run and all its artifacts.

    Examples:
        pipeline clean 20241218_120000
        pipeline clean phase1_v1 --force
    """
    project_root_path = get_project_root(project_root)
    run_dir = project_root_path / "runs" / run_id

    if not run_dir.exists():
        show_error(f"Run not found: {run_id}")
        raise typer.Exit(1)

    console.print(f"\n[bold yellow]⚠ Warning:[/bold yellow] This will delete run: {run_id}")
    console.print(f"Location: {run_dir}\n")

    if not force:
        if not typer.confirm("Are you sure you want to delete this run?"):
            show_warning("Deletion cancelled")
            return

    try:
        shutil.rmtree(run_dir)
        show_success(f"Deleted run: {run_id}")
    except Exception as e:
        show_error(f"Failed to delete run: {e}")
        raise typer.Exit(1)
