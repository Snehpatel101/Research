#!/usr/bin/env python3
"""
Phase 1 Pipeline CLI
Typer-based command-line interface for pipeline management
"""
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_config import PipelineConfig, create_default_config
from pipeline_runner import PipelineRunner, StageStatus
from manifest import ArtifactManifest, compare_runs

app = typer.Typer(
    name="pipeline",
    help="Phase 1 Data Preparation Pipeline CLI",
    add_completion=False
)
console = Console()


def show_error(message: str):
    """Display error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def show_success(message: str):
    """Display success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def show_info(message: str):
    """Display info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def show_warning(message: str):
    """Display warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


@app.command()
def run(
    symbols: str = typer.Option(
        "MES,MGC",
        "--symbols",
        "-s",
        help="Comma-separated list of symbols to process"
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD)"
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD)"
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Custom run ID (defaults to YYYYMMDD_HHMMSS)"
    ),
    description: str = typer.Option(
        "Phase 1 pipeline run",
        "--description",
        "-d",
        help="Description of this run"
    ),
    train_ratio: float = typer.Option(
        0.70,
        "--train-ratio",
        help="Training set ratio"
    ),
    val_ratio: float = typer.Option(
        0.15,
        "--val-ratio",
        help="Validation set ratio"
    ),
    test_ratio: float = typer.Option(
        0.15,
        "--test-ratio",
        help="Test set ratio"
    ),
    purge_bars: int = typer.Option(
        20,
        "--purge-bars",
        help="Number of bars to purge at split boundaries"
    ),
    embargo_bars: int = typer.Option(
        288,
        "--embargo-bars",
        help="Embargo period in bars (~1 day for 5-min data)"
    ),
    horizons: str = typer.Option(
        "1,5,20",
        "--horizons",
        help="Comma-separated label horizons"
    ),
    synthetic: bool = typer.Option(
        False,
        "--synthetic",
        help="Generate synthetic data"
    ),
    project_root: str = typer.Option(
        "str(Path(__file__).parent.parent.resolve())",
        "--project-root",
        help="Project root directory"
    )
):
    """
    Run the complete Phase 1 pipeline.

    Examples:
        pipeline run --symbols MES,MGC --start 2020-01-01 --end 2024-12-31
        pipeline run --run-id phase1_v1 --synthetic
        pipeline run --symbols MES,MGC,MNQ --horizons 1,5,10,20
    """
    console.print("\n[bold cyan]Phase 1 Pipeline - Data Preparation[/bold cyan]\n")

    # Parse symbols and horizons
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    horizon_list = [int(h.strip()) for h in horizons.split(",")]

    # Create configuration
    try:
        config = create_default_config(
            symbols=symbol_list,
            start_date=start,
            end_date=end,
            run_id=run_id,
            description=description,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            label_horizons=horizon_list,
            use_synthetic_data=synthetic,
            project_root=Path(project_root)
        )
    except ValueError as e:
        show_error(f"Configuration error: {e}")
        raise typer.Exit(1)

    # Display configuration
    table = Table(title="Pipeline Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Run ID", config.run_id)
    table.add_row("Description", config.description)
    table.add_row("Symbols", ", ".join(config.symbols))
    table.add_row("Date Range", f"{config.start_date or 'N/A'} to {config.end_date or 'N/A'}")
    table.add_row("Label Horizons", ", ".join(map(str, config.label_horizons)))
    table.add_row("Train/Val/Test", f"{config.train_ratio:.0%} / {config.val_ratio:.0%} / {config.test_ratio:.0%}")
    table.add_row("Purge/Embargo", f"{config.purge_bars} / {config.embargo_bars} bars")
    table.add_row("Synthetic Data", "Yes" if config.use_synthetic_data else "No")

    console.print(table)
    console.print()

    # Validate configuration
    issues = config.validate()
    if issues:
        show_error("Configuration validation failed:")
        for issue in issues:
            console.print(f"  • {issue}")
        raise typer.Exit(1)

    show_success("Configuration validated")
    console.print()

    # Confirm execution
    if not typer.confirm("Do you want to proceed with pipeline execution?"):
        show_warning("Pipeline execution cancelled")
        raise typer.Exit(0)

    console.print()

    # Run pipeline
    try:
        runner = PipelineRunner(config)
        success = runner.run()

        console.print()
        if success:
            show_success(f"Pipeline completed successfully! Run ID: {config.run_id}")
            console.print(f"\n[bold]Results:[/bold]")
            console.print(f"  • Logs: {runner.log_file}")
            console.print(f"  • Config: {config.run_config_dir / 'config.json'}")
            console.print(f"  • Manifest: {runner.manifest.manifest_path}")
            console.print(f"\nView results with: [bold cyan]pipeline status --run-id {config.run_id}[/bold cyan]")
        else:
            show_error(f"Pipeline failed. Check logs at {runner.log_file}")
            raise typer.Exit(1)

    except Exception as e:
        show_error(f"Pipeline execution error: {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def rerun(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to resume"
    ),
    from_stage: str = typer.Option(
        None,
        "--from",
        help="Stage to resume from (e.g., 'labels', 'features')"
    ),
    project_root: str = typer.Option(
        "str(Path(__file__).parent.parent.resolve())",
        "--project-root",
        help="Project root directory"
    )
):
    """
    Resume a pipeline run from a specific stage.

    Examples:
        pipeline rerun 20241218_120000 --from labeling
        pipeline rerun phase1_v1 --from create_splits
    """
    console.print(f"\n[bold cyan]Resuming Pipeline Run: {run_id}[/bold cyan]\n")

    # Load configuration
    try:
        config = PipelineConfig.load_from_run_id(run_id, Path(project_root))
        show_success(f"Loaded configuration for run: {run_id}")
    except FileNotFoundError as e:
        show_error(f"Run not found: {e}")
        raise typer.Exit(1)

    # Display configuration
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Symbols: {', '.join(config.symbols)}")
    console.print(f"  Description: {config.description}")

    # Map friendly names to stage names
    stage_map = {
        'data': 'data_generation',
        'cleaning': 'data_cleaning',
        'clean': 'data_cleaning',
        'features': 'feature_engineering',
        'labeling': 'labeling',
        'labels': 'labeling',
        'splits': 'create_splits',
        'report': 'generate_report'
    }

    stage_name = stage_map.get(from_stage, from_stage) if from_stage else None

    if stage_name:
        show_info(f"Resuming from stage: {stage_name}")
    else:
        show_info("Resuming from last successful stage")

    console.print()

    # Run pipeline
    try:
        runner = PipelineRunner(config, resume=True)
        success = runner.run(from_stage=stage_name)

        console.print()
        if success:
            show_success("Pipeline completed successfully!")
        else:
            show_error("Pipeline failed. Check logs for details.")
            raise typer.Exit(1)

    except Exception as e:
        show_error(f"Pipeline execution error: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to check"
    ),
    project_root: str = typer.Option(
        "str(Path(__file__).parent.parent.resolve())",
        "--project-root",
        help="Project root directory"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information"
    )
):
    """
    Check the status of a pipeline run.

    Examples:
        pipeline status 20241218_120000
        pipeline status phase1_v1 --verbose
    """
    console.print(f"\n[bold cyan]Pipeline Run Status: {run_id}[/bold cyan]\n")

    project_path = Path(project_root)
    run_dir = project_path / "runs" / run_id

    # Check if run exists
    if not run_dir.exists():
        show_error(f"Run not found: {run_id}")
        raise typer.Exit(1)

    # Load configuration
    try:
        config = PipelineConfig.load_from_run_id(run_id, project_path)
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

    # Load pipeline state
    state_path = run_dir / "artifacts" / "pipeline_state.json"
    if state_path.exists():
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

        all_stages = [
            "data_generation",
            "data_cleaning",
            "feature_engineering",
            "labeling",
            "create_splits",
            "generate_report"
        ]

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
        completed = len(completed_stages)
        progress_pct = (completed / total_stages) * 100

        console.print(f"[bold]Progress:[/bold] {completed}/{total_stages} stages ({progress_pct:.0f}%)")
        console.print()

    else:
        show_warning("No pipeline state found. Run may not have started.")

    # Load manifest
    try:
        manifest = ArtifactManifest.load(run_id, project_path)
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

    # Show paths
    if verbose:
        console.print(f"[bold]Paths:[/bold]")
        console.print(f"  Run directory: {run_dir}")
        console.print(f"  Config: {run_dir / 'config' / 'config.json'}")
        console.print(f"  Logs: {run_dir / 'logs' / 'pipeline.log'}")
        console.print(f"  Manifest: {run_dir / 'artifacts' / 'manifest.json'}")


@app.command()
def validate(
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
    project_root: str = typer.Option(
        "str(Path(__file__).parent.parent.resolve())",
        "--project-root",
        help="Project root directory"
    )
):
    """
    Validate pipeline configuration and data integrity.

    Examples:
        pipeline validate --symbols MES,MGC,MNQ
        pipeline validate --run-id 20241218_120000
    """
    console.print("\n[bold cyan]Pipeline Validation[/bold cyan]\n")

    # Load or create configuration
    if run_id:
        try:
            config = PipelineConfig.load_from_run_id(run_id, Path(project_root))
            show_info(f"Validating configuration for run: {run_id}")
        except FileNotFoundError:
            show_error(f"Run not found: {run_id}")
            raise typer.Exit(1)
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        config = create_default_config(
            symbols=symbol_list,
            project_root=Path(project_root)
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
            manifest = ArtifactManifest.load(run_id, Path(project_root))
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


@app.command()
def list_runs(
    project_root: str = typer.Option(
        "str(Path(__file__).parent.parent.resolve())",
        "--project-root",
        help="Project root directory"
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Number of runs to show"
    )
):
    """
    List all pipeline runs.

    Examples:
        pipeline list-runs
        pipeline list-runs --limit 20
    """
    console.print("\n[bold cyan]Pipeline Runs[/bold cyan]\n")

    runs_dir = Path(project_root) / "runs"

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
            config = PipelineConfig.load_from_run_id(run_id, Path(project_root))
            description = config.description
            symbols = ", ".join(config.symbols)
        except Exception as e:
            logger.debug(f"Could not load config for run {run_id}: {e}")
            description = "N/A"
            symbols = "N/A"

        # Check status
        state_path = run_dir / "artifacts" / "pipeline_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            completed = len(state.get('completed_stages', []))
            total = 6  # Total stages
            status = f"{completed}/{total} stages"
        else:
            status = "Not started"

        # Get creation time
        created = datetime.fromtimestamp(run_dir.stat().st_ctime)
        created_str = created.strftime("%Y-%m-%d %H:%M")

        table.add_row(run_id, description, symbols, status, created_str)

    console.print(table)
    console.print(f"\n[dim]Showing {len(run_dirs)} most recent runs[/dim]")


@app.command()
def compare(
    run1: str = typer.Argument(..., help="First run ID"),
    run2: str = typer.Argument(..., help="Second run ID"),
    project_root: str = typer.Option(
        "str(Path(__file__).parent.parent.resolve())",
        "--project-root",
        help="Project root directory"
    )
):
    """
    Compare artifacts between two runs.

    Examples:
        pipeline compare 20241218_120000 20241218_130000
    """
    console.print(f"\n[bold cyan]Comparing Runs: {run1} vs {run2}[/bold cyan]\n")

    try:
        comparison = compare_runs(run1, run2, Path(project_root))

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


@app.command()
def clean(
    run_id: str = typer.Argument(..., help="Run ID to clean"),
    project_root: str = typer.Option(
        "str(Path(__file__).parent.parent.resolve())",
        "--project-root",
        help="Project root directory"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation"
    )
):
    """
    Delete a pipeline run and all its artifacts.

    Examples:
        pipeline clean 20241218_120000
        pipeline clean phase1_v1 --force
    """
    run_dir = Path(project_root) / "runs" / run_id

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
        import shutil
        shutil.rmtree(run_dir)
        show_success(f"Deleted run: {run_id}")
    except Exception as e:
        show_error(f"Failed to delete run: {e}")
        raise typer.Exit(1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
