"""
Pipeline Commands - run, data, status, resume.

These commands provide the main ML pipeline orchestration using MLPipeline.
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


def _build_config(
    symbol: str,
    horizons: list[int],
    models: list[str],
    timeframe: str,
    training_mode: str,
    build_ensemble: bool,
    optimize_features: bool,
    evaluation_methods: list[str],
    start_date: str | None,
    end_date: str | None,
    config_path: Path | None,
):
    """Build MLConfig from arguments."""
    from src.ml_pipeline import MLConfig, ModelConfig  # type: ignore[import-not-found]

    if config_path:
        return MLConfig.from_yaml(config_path)

    model_configs = []
    for model_name in models:
        model_configs.append(
            ModelConfig(
                name=model_name,
                timeframe=timeframe,
                optimize_features=optimize_features,
            )
        )

    return MLConfig(
        symbol=symbol,
        horizons=horizons,
        models=model_configs,
        training_mode=training_mode,
        build_ensemble=build_ensemble,
        evaluation_methods=evaluation_methods,
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
    timeframe: str = typer.Option("5min", "--timeframe", "-t", help="Primary timeframe"),
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
    evaluation_methods: str = typer.Option(
        "", "--evaluation-methods", help="Evaluation methods (comma-separated)"
    ),
    start_date: str | None = typer.Option(None, "--start-date", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end-date", help="End date (YYYY-MM-DD)"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
):
    """
    Run full ML pipeline (data + training + evaluation).

    This orchestrates the complete pipeline including data preparation,
    model training, and evaluation.
    """
    from src.ml_pipeline import MLPipeline

    # Parse comma-separated arguments
    horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    eval_methods = [e.strip() for e in evaluation_methods.split(",") if e.strip()]

    ml_config = _build_config(
        symbol=symbol,
        horizons=horizon_list,
        models=model_list,
        timeframe=timeframe,
        training_mode=training_mode,
        build_ensemble=build_ensemble,
        optimize_features=optimize_features,
        evaluation_methods=eval_methods,
        start_date=start_date,
        end_date=end_date,
        config_path=config,
    )

    console.print("\n[bold]Starting full ML pipeline[/bold]")
    console.print(f"Symbol: {ml_config.symbol}")
    console.print(f"Horizons: {ml_config.horizons}")
    console.print(f"Models: {[m.name for m in ml_config.models]}")
    console.print(f"Run ID: {ml_config.run_id}")
    console.print()

    pipeline = MLPipeline(ml_config)

    try:
        results = pipeline.run()

        console.print("\n" + "=" * 70)
        console.print("[bold green]PIPELINE COMPLETED SUCCESSFULLY[/bold green]")
        console.print("=" * 70)
        console.print(f"Run ID: {results['run_id']}")
        console.print(f"Output: {results['output_dir']}")

        if "data" in results:
            console.print(f"\nData paths: {results['data'].get('labeled_data_paths', {})}")

        if "training" in results:
            console.print(f"\nModel paths: {results['training'].get('model_paths', {})}")

        if "evaluation" in results:
            console.print("\nEvaluation complete")

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
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
):
    """
    Run data pipeline only.

    Prepares and processes raw data for training without actually training models.
    """
    from src.ml_pipeline import MLPipeline

    # Parse comma-separated arguments
    horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]

    ml_config = _build_config(
        symbol=symbol,
        horizons=horizon_list,
        models=["xgboost"],  # Placeholder - not used for data pipeline
        timeframe=timeframe,
        training_mode="standard",
        build_ensemble=False,
        optimize_features=False,
        evaluation_methods=[],
        start_date=start_date,
        end_date=end_date,
        config_path=config,
    )

    console.print("\n[bold]Running data pipeline[/bold]")
    console.print(f"Symbol: {ml_config.symbol}")
    console.print(f"Horizons: {ml_config.horizons}")
    console.print(f"Run ID: {ml_config.run_id}")
    console.print()

    pipeline = MLPipeline(ml_config)

    try:
        results = pipeline.run_data()

        console.print("\n" + "=" * 70)
        console.print("[bold green]DATA PIPELINE COMPLETED[/bold green]")
        console.print("=" * 70)
        console.print(f"Labeled data paths: {results.get('labeled_data_paths', {})}")
        console.print(f"Features: {results.get('features_info', {}).get('n_features', 0)} features")

        raise typer.Exit(0)

    except Exception as e:
        show_error(f"Data pipeline failed: {e}")
        raise typer.Exit(1) from None


@pipeline_app.command("status")
def show_status(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to check"),
):
    """
    Show pipeline status for a specific run.

    Displays the current state, completed phases, and any errors.
    """
    from src.ml_pipeline import PipelineState

    try:
        state = PipelineState.load(run_id)
        summary = state.get_summary()

        console.print("=" * 70)
        console.print("[bold]PIPELINE STATUS[/bold]")
        console.print("=" * 70)
        console.print(f"Run ID: {summary['run_id']}")
        console.print(f"Status: {summary['status']}")

        completed = summary.get("completed_phases", [])
        pending = summary.get("pending_phases", [])

        console.print(f"\nCompleted phases: {', '.join(completed) if completed else 'None'}")
        console.print(f"Pending phases: {', '.join(pending) if pending else 'None'}")

        if summary.get("start_time"):
            console.print(f"\nStart time: {summary['start_time']}")
        if summary.get("end_time"):
            console.print(f"End time: {summary['end_time']}")
        if summary.get("total_duration_seconds"):
            console.print(f"Duration: {summary['total_duration_seconds']:.1f} seconds")

        if summary.get("error_message"):
            show_error(summary["error_message"])

        raise typer.Exit(0)

    except FileNotFoundError:
        show_error(f"Run ID '{run_id}' not found")
        raise typer.Exit(1) from None
    except Exception as e:
        show_error(str(e))
        raise typer.Exit(1) from None


@pipeline_app.command("resume")
def resume_pipeline(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to resume"),
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option("20", "--horizons", "-h", help="Label horizons (comma-separated)"),
    models: str = typer.Option(
        "xgboost", "--models", "-m", help="Models to train (comma-separated)"
    ),
    timeframe: str = typer.Option("5min", "--timeframe", "-t", help="Primary timeframe"),
    training_mode: str = typer.Option("standard", "--training-mode", help="Training mode"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
):
    """
    Resume pipeline from checkpoint.

    Continues a previously interrupted pipeline run from where it left off.
    """
    from src.ml_pipeline import MLPipeline

    # Parse comma-separated arguments
    horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    ml_config = _build_config(
        symbol=symbol,
        horizons=horizon_list,
        models=model_list,
        timeframe=timeframe,
        training_mode=training_mode,
        build_ensemble=False,
        optimize_features=False,
        evaluation_methods=[],
        start_date=None,
        end_date=None,
        config_path=config,
    )

    ml_config.run_id = run_id

    console.print("\n[bold]Resuming pipeline[/bold]")
    console.print(f"Run ID: {ml_config.run_id}")
    console.print()

    pipeline = MLPipeline(ml_config)

    try:
        results = pipeline.resume()

        console.print("\n" + "=" * 70)
        console.print("[bold green]RESUME COMPLETED[/bold green]")
        console.print("=" * 70)
        console.print(f"Run ID: {results['run_id']}")

        raise typer.Exit(0)

    except Exception as e:
        show_error(f"Resume failed: {e}")
        raise typer.Exit(1) from None
