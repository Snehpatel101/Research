"""
Pipeline Commands - run, data, status, resume.

These commands provide the main ML pipeline orchestration using MLPipeline.

Note: This CLI uses the existing pipeline infrastructure:
- src.core.config.PipelineConfig for ML pipeline configuration
- src.orchestrator.MLPipeline for ML pipeline orchestration
- src.data.pipeline.runner.PipelineRunner for data pipeline
- src.data.pipeline.data_config.DataConfig for data pipeline configuration
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.cli.utils import (
    console,
    show_error,
)

pipeline_app = typer.Typer(
    name="pipeline",
    help="Pipeline orchestration commands",
    no_args_is_help=True,
)


def _build_ml_config(
    symbol: str,
    horizons: list[int],
    models: list[str],
    training_mode: str,
    build_ensemble: bool,
    optimize_features: bool,
    data_path: Path,
    output_dir: Path,
    config_path: Path | None,
):
    """Build PipelineConfig from arguments for ML pipeline."""
    from src.core.config import PipelineConfig

    if config_path:
        return PipelineConfig.load(config_path)

    return PipelineConfig(
        symbol=symbol,
        data_path=data_path,
        output_dir=output_dir,
        horizons=horizons,
        models=models,
        training_mode=training_mode,
        build_ensemble=build_ensemble,
        optimize_features=optimize_features,
    )


def _build_data_config(
    symbol: str,
    horizons: list[int],
    timeframe: str,
    start_date: str | None,
    end_date: str | None,
    config_path: Path | None,
):
    """Build DataConfig from arguments for data pipeline."""
    from src.data.pipeline.data_config import DataConfig

    if config_path:
        return DataConfig.load_config(config_path)

    return DataConfig(
        symbols=[symbol],
        label_horizons=horizons,
        target_timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )


@pipeline_app.command("run")
def run_pipeline(
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option("20", "--horizons", "-h", help="Label horizons (comma-separated)"),
    models: str = typer.Option(
        "xgboost", "--models", "-m", help="Models to train (comma-separated)"
    ),
    training_mode: str = typer.Option(
        "standard",
        "--training-mode",
        help="Training mode: standard, walk_forward, regime_aware, meta_labeling",
    ),
    build_ensemble: bool = typer.Option(
        False, "--build-ensemble", help="Build ensemble from base models"
    ),
    optimize_features: bool = typer.Option(
        False, "--optimize-features", help="Run feature optimization"
    ),
    data_path: Path = typer.Option(
        ..., "--data-path", "-d", help="Path to input data file (parquet)"
    ),
    output_dir: Path = typer.Option(
        Path("./experiments"), "--output-dir", "-o", help="Output directory for results"
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to JSON config file"),
):
    """
    Run full ML pipeline (data + training + evaluation).

    This orchestrates the complete pipeline including data preparation,
    model training, and evaluation using MLPipeline from src.orchestrator.

    Example:
        pipeline run --symbol MES --data-path ./data/mes.parquet --output-dir ./exp
    """
    from src.orchestrator import MLPipeline

    # Parse comma-separated arguments
    horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    ml_config = _build_ml_config(
        symbol=symbol,
        horizons=horizon_list,
        models=model_list,
        training_mode=training_mode,
        build_ensemble=build_ensemble,
        optimize_features=optimize_features,
        data_path=data_path,
        output_dir=output_dir,
        config_path=config,
    )

    console.print("\n[bold]Starting full ML pipeline[/bold]")
    console.print(f"Symbol: {ml_config.symbol}")
    console.print(f"Horizons: {ml_config.horizons}")
    console.print(f"Models: {ml_config.models}")
    console.print(f"Data path: {ml_config.data_path}")
    console.print(f"Output dir: {ml_config.output_dir}")
    console.print()

    pipeline = MLPipeline(ml_config)

    try:
        result = pipeline.run()

        console.print("\n" + "=" * 70)
        console.print("[bold green]PIPELINE COMPLETED SUCCESSFULLY[/bold green]")
        console.print("=" * 70)
        console.print(f"Run ID: {result.run_id}")
        console.print(f"Output: {result.output_dir}")
        console.print(f"Duration: {result.duration_seconds:.1f}s")

        if result.best_model:
            console.print(f"\nBest model: {result.best_model}")

        if result.metrics:
            console.print("\nMetrics:")
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    console.print(f"  {k}: {v:.4f}")
                else:
                    console.print(f"  {k}: {v}")

        raise typer.Exit(0)

    except Exception as e:
        show_error(f"Pipeline failed: {e}")
        raise typer.Exit(1) from None


@pipeline_app.command("data")
def run_data(
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option("20", "--horizons", "-h", help="Label horizons (comma-separated)"),
    timeframe: str = typer.Option("5min", "--timeframe", "-t", help="Primary timeframe"),
    start_date: str | None = typer.Option(None, "--start-date", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end-date", help="End date (YYYY-MM-DD)"),
    resume: bool = typer.Option(False, "--resume", help="Resume from last successful stage"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to JSON config file"),
):
    """
    Run data pipeline only.

    Prepares and processes raw data for training without actually training models.
    Uses PipelineRunner from src.data.pipeline for stage-based execution.

    Example:
        pipeline data --symbol MES --timeframe 5min --horizons 20
    """
    from src.data.pipeline.runner import PipelineRunner

    # Parse comma-separated arguments
    horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]

    data_config = _build_data_config(
        symbol=symbol,
        horizons=horizon_list,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config_path=config,
    )

    console.print("\n[bold]Running data pipeline[/bold]")
    console.print(f"Symbol: {data_config.symbols}")
    console.print(f"Horizons: {data_config.label_horizons}")
    console.print(f"Timeframe: {data_config.target_timeframe}")
    console.print(f"Run ID: {data_config.run_id}")
    console.print()

    try:
        runner = PipelineRunner(data_config, resume=resume)
        success = runner.run()

        console.print("\n" + "=" * 70)
        if success:
            console.print("[bold green]DATA PIPELINE COMPLETED[/bold green]")
        else:
            console.print("[bold red]DATA PIPELINE FAILED[/bold red]")
        console.print("=" * 70)
        console.print(f"Run ID: {data_config.run_id}")
        console.print(f"Completed stages: {runner.get_completed_stages()}")

        raise typer.Exit(0 if success else 1)

    except Exception as e:
        show_error(f"Data pipeline failed: {e}")
        raise typer.Exit(1) from None


@pipeline_app.command("status")
def show_status(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to check"),
    project_root: Path = typer.Option(
        Path("."), "--project-root", "-p", help="Project root directory"
    ),
):
    """
    Show pipeline status for a specific run.

    Displays the current state, completed phases, and any errors.
    Reads state from data/runs/{run_id}/artifacts/pipeline_state.json.

    Example:
        pipeline status --run-id 20250101_120000_abc123
    """
    import json

    # Look for pipeline state file in the expected location
    state_path = project_root / "data" / "runs" / run_id / "artifacts" / "pipeline_state.json"

    if not state_path.exists():
        show_error(f"Run ID '{run_id}' not found at {state_path}")
        console.print("\n[dim]Pipeline state file not found. The run may not exist or ")
        console.print("may have been created with a different project root.[/dim]")
        raise typer.Exit(1)

    try:
        with open(state_path) as f:
            state = json.load(f)

        console.print("=" * 70)
        console.print("[bold]PIPELINE STATUS[/bold]")
        console.print("=" * 70)
        console.print(f"Run ID: {state.get('run_id', run_id)}")
        console.print(f"Saved at: {state.get('saved_at', 'Unknown')}")

        completed = state.get("completed_stages", [])
        console.print(f"\nCompleted stages ({len(completed)}):")
        for stage in completed:
            console.print(f"  [green]+ {stage}[/green]")

        # Show stage results if available
        stage_results = state.get("stage_results", {})
        if stage_results:
            console.print("\nStage results:")
            for stage_name, result in stage_results.items():
                status = result.get("status", "unknown")
                duration = result.get("duration_seconds", 0)
                if status == "completed":
                    console.print(f"  [green]{stage_name}[/green]: {duration:.2f}s")
                else:
                    error = result.get("error", "")
                    console.print(f"  [red]{stage_name}[/red]: {status} - {error}")

        raise typer.Exit(0)

    except json.JSONDecodeError as e:
        show_error(f"Failed to parse pipeline state: {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        show_error(str(e))
        raise typer.Exit(1) from None


@pipeline_app.command("resume")
def resume_pipeline(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to resume"),
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option("20", "--horizons", "-h", help="Label horizons (comma-separated)"),
    timeframe: str = typer.Option("5min", "--timeframe", "-t", help="Primary timeframe"),
    start_date: str | None = typer.Option(None, "--start-date", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end-date", help="End date (YYYY-MM-DD)"),
    from_stage: str | None = typer.Option(None, "--from-stage", help="Stage to resume from"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to JSON config file"),
):
    """
    Resume data pipeline from checkpoint.

    Continues a previously interrupted pipeline run from where it left off.
    Uses PipelineRunner with resume=True to load previous state.

    Example:
        pipeline resume --run-id 20250101_120000_abc123 --symbol MES
    """
    from src.data.pipeline.runner import PipelineRunner

    # Parse comma-separated arguments
    horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]

    data_config = _build_data_config(
        symbol=symbol,
        horizons=horizon_list,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config_path=config,
    )

    # Override run_id to resume the specific run
    data_config.run_id = run_id

    console.print("\n[bold]Resuming data pipeline[/bold]")
    console.print(f"Run ID: {data_config.run_id}")
    console.print(f"Symbol: {data_config.symbols}")
    console.print()

    try:
        runner = PipelineRunner(data_config, resume=True)
        success = runner.run(from_stage=from_stage)

        console.print("\n" + "=" * 70)
        if success:
            console.print("[bold green]RESUME COMPLETED[/bold green]")
        else:
            console.print("[bold red]RESUME FAILED[/bold red]")
        console.print("=" * 70)
        console.print(f"Run ID: {data_config.run_id}")
        console.print(f"Completed stages: {runner.get_completed_stages()}")

        raise typer.Exit(0 if success else 1)

    except Exception as e:
        show_error(f"Resume failed: {e}")
        raise typer.Exit(1) from None
