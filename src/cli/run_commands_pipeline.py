"""
Pipeline execution commands - run and rerun.

Provides CLI commands for running the Phase 1 data pipeline
and resuming failed/incomplete runs.
"""

import typer
from rich.table import Table

from .run_commands_core import (
    _create_config_from_args,
    _get_pipeline_config,
    _get_pipeline_runner,
    _get_presets_module,
)
from .utils import console, get_project_root, show_error, show_info, show_success, show_warning


def run_command(
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Comma-separated list of symbols (auto-detected from data/raw/ if not specified)"
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        "-p",
        help="Trading preset: scalping, day_trading, swing (overrides defaults)"
    ),
    timeframe: str | None = typer.Option(
        None,
        "--timeframe",
        "-t",
        help="Target timeframe for resampling (e.g., 1min, 5min, 15min)"
    ),
    start: str | None = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD)"
    ),
    end: str | None = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD)"
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Custom run ID (defaults to YYYYMMDD_HHMMSS)"
    ),
    description: str | None = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of this run"
    ),
    train_ratio: float | None = typer.Option(
        None,
        "--train-ratio",
        help="Training set ratio (default: 0.70)"
    ),
    val_ratio: float | None = typer.Option(
        None,
        "--val-ratio",
        help="Validation set ratio (default: 0.15)"
    ),
    test_ratio: float | None = typer.Option(
        None,
        "--test-ratio",
        help="Test set ratio (default: 0.15)"
    ),
    purge_bars: int | None = typer.Option(
        None,
        "--purge-bars",
        help="Number of bars to purge at split boundaries"
    ),
    embargo_bars: int | None = typer.Option(
        None,
        "--embargo-bars",
        help="Embargo period in bars (~1 day for 5-min data)"
    ),
    horizons: str | None = typer.Option(
        None,
        "--horizons",
        help="Comma-separated label horizons (overrides preset)"
    ),
    feature_set: str | None = typer.Option(
        None,
        "--feature-set",
        help="Feature set: core_min, core_full, mtf_plus, boosting_optimal, neural_optimal, etc."
    ),
    # MTF settings
    mtf_mode: str | None = typer.Option(
        None,
        "--mtf-mode",
        help="MTF mode: bars, indicators, or both (default: both)"
    ),
    mtf_timeframes: str | None = typer.Option(
        None,
        "--mtf-timeframes",
        help="Comma-separated MTF timeframes (e.g., '15min,30min,1h,4h,daily')"
    ),
    mtf_disable: bool = typer.Option(
        False,
        "--mtf-disable",
        help="Disable MTF feature generation entirely"
    ),
    # Feature toggles
    enable_wavelets: bool | None = typer.Option(
        None,
        "--enable-wavelets/--disable-wavelets",
        help="Enable/disable wavelet decomposition features"
    ),
    enable_microstructure: bool | None = typer.Option(
        None,
        "--enable-microstructure/--disable-microstructure",
        help="Enable/disable microstructure features (bid-ask, order flow)"
    ),
    enable_volume: bool | None = typer.Option(
        None,
        "--enable-volume/--disable-volume",
        help="Enable/disable volume-based features"
    ),
    enable_volatility: bool | None = typer.Option(
        None,
        "--enable-volatility/--disable-volatility",
        help="Enable/disable volatility features"
    ),
    # Labeling parameters
    k_up: float | None = typer.Option(
        None,
        "--k-up",
        help="Upper barrier multiplier (overrides symbol-specific defaults)"
    ),
    k_down: float | None = typer.Option(
        None,
        "--k-down",
        help="Lower barrier multiplier (overrides symbol-specific defaults)"
    ),
    max_bars: int | None = typer.Option(
        None,
        "--max-bars",
        help="Maximum bars for label timeout (overrides defaults)"
    ),
    # Scaling options
    scaler_type: str | None = typer.Option(
        None,
        "--scaler-type",
        help="Scaler type: robust, standard, minmax, quantile, none (default: robust)"
    ),
    # Model selection (Phase 2+)
    model_type: str | None = typer.Option(
        None,
        "--model-type",
        help="Target model type: xgboost, lightgbm, lstm, transformer, ensemble, etc."
    ),
    base_models: str | None = typer.Option(
        None,
        "--base-models",
        help="Comma-separated base models for ensemble (e.g., 'xgboost,lstm,transformer')"
    ),
    meta_learner: str | None = typer.Option(
        None,
        "--meta-learner",
        help="Meta-learner for ensemble stacking (e.g., logistic, xgboost)"
    ),
    sequence_length: int | None = typer.Option(
        None,
        "--sequence-length",
        help="Sequence length for sequential models (LSTM, Transformer)"
    ),
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Project root directory"
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt"
    ),
) -> None:
    """
    Run the complete Phase 1 pipeline with full configuration.

    Requires real data in data/raw/ directory (e.g., MES_1m.parquet).

    Examples:
        # Basic usage
        pipeline run --symbols MES --start 2020-01-01 --end 2024-12-31

        # With preset
        pipeline run --preset day_trading --symbols MES

        # Custom horizons and barriers
        pipeline run --symbols MES --horizons 5,10,20 --k-up 1.5 --k-down 1.0

        # MTF configuration
        pipeline run --symbols MES --mtf-mode indicators --mtf-timeframes 15min,1h,4h

        # Feature toggles
        pipeline run --symbols MES --enable-wavelets --disable-microstructure

        # Model-aware preparation (Phase 2+)
        pipeline run --symbols MES --model-type xgboost --feature-set boosting_optimal

        # Ensemble preparation
        pipeline run --symbols MES --model-type ensemble --base-models xgboost,lstm
    """
    pipeline_config = _get_pipeline_config()
    pipeline_mod = _get_pipeline_runner()
    presets_mod = _get_presets_module()

    console.print("\n[bold cyan]Phase 1 Pipeline - Data Preparation[/bold cyan]\n")

    # Create configuration from preset or defaults
    try:
        project_root_path = get_project_root(project_root)
        config = _create_config_from_args(
            preset=preset,
            symbols=symbols,
            timeframe=timeframe,
            horizons=horizons,
            feature_set=feature_set,
            start=start,
            end=end,
            run_id=run_id,
            description=description,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            # MTF settings
            mtf_mode=mtf_mode,
            mtf_timeframes=mtf_timeframes,
            mtf_enable=not mtf_disable if mtf_disable else None,
            # Feature toggles
            enable_wavelets=enable_wavelets,
            enable_microstructure=enable_microstructure,
            enable_volume_features=enable_volume,
            enable_volatility_features=enable_volatility,
            # Labeling parameters
            k_up=k_up,
            k_down=k_down,
            max_bars=max_bars,
            # Scaling options
            scaler_type=scaler_type,
            # Model selection (Phase 2+)
            model_type=model_type,
            base_models=base_models,
            meta_learner=meta_learner,
            sequence_length=sequence_length,
            # Common
            project_root_path=project_root_path,
            pipeline_config=pipeline_config,
            presets_mod=presets_mod
        )
    except ValueError as e:
        show_error(f"Configuration error: {e}")
        raise typer.Exit(1)

    # Display configuration
    table = Table(title="Pipeline Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Run ID", config.run_id)
    table.add_row("Preset", preset or "none (defaults)")
    table.add_row("Description", config.description)
    table.add_row("Symbols", ", ".join(config.symbols))
    table.add_row("Timeframe", config.target_timeframe)
    table.add_row("Date Range", f"{config.start_date or 'N/A'} to {config.end_date or 'N/A'}")
    table.add_row("Label Horizons", ", ".join(map(str, config.label_horizons)))
    table.add_row("Feature Set", config.feature_set)
    table.add_row("Train/Val/Test", f"{config.train_ratio:.0%} / {config.val_ratio:.0%} / {config.test_ratio:.0%}")
    table.add_row("Purge/Embargo", f"{config.purge_bars} / {config.embargo_bars} bars")

    # MTF settings
    table.add_row("MTF Mode", config.mtf_mode)
    table.add_row("MTF Timeframes", ", ".join(config.mtf_timeframes) if config.mtf_timeframes else "disabled")

    # Model settings if specified
    if hasattr(config, 'model_config') and config.model_config:
        model_cfg = config.model_config
        if 'model_type' in model_cfg:
            table.add_row("Target Model", model_cfg['model_type'])
        if 'base_models' in model_cfg:
            table.add_row("Base Models", ", ".join(model_cfg['base_models']))
        if 'meta_learner' in model_cfg:
            table.add_row("Meta-Learner", model_cfg['meta_learner'])

    # Scaler type if specified
    if hasattr(config, 'scaler_type') and config.scaler_type:
        table.add_row("Scaler Type", config.scaler_type)

    console.print(table)
    console.print()

    # Validate configuration
    issues = config.validate()
    if issues:
        show_error("Configuration validation failed:")
        for issue in issues:
            console.print(f"  [red]x[/red] {issue}")
        raise typer.Exit(1)

    show_success("Configuration validated")
    console.print()

    # Confirm execution
    if not yes and not typer.confirm("Do you want to proceed with pipeline execution?"):
        show_warning("Pipeline execution cancelled")
        raise typer.Exit(0)

    console.print()

    # Run pipeline
    try:
        runner = pipeline_mod.PipelineRunner(config)
        success = runner.run()

        console.print()
        if success:
            show_success(f"Pipeline completed successfully! Run ID: {config.run_id}")
            console.print("\n[bold]Results:[/bold]")
            console.print(f"  Logs: {runner.log_file}")
            console.print(f"  Config: {config.run_config_dir / 'config.json'}")
            console.print(f"  Manifest: {runner.manifest.manifest_path}")
            console.print(f"\nView results with: [bold cyan]pipeline status --run-id {config.run_id}[/bold cyan]")
        else:
            show_error(f"Pipeline failed. Check logs at {runner.log_file}")
            raise typer.Exit(1)

    except Exception as e:
        show_error(f"Pipeline execution error: {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


def rerun_command(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to resume"
    ),
    from_stage: str | None = typer.Option(
        None,
        "--from",
        help="Stage to resume from (e.g., 'labels', 'features')"
    ),
    project_root: str | None = typer.Option(
        None,
        "--project-root",
        help="Project root directory"
    )
) -> None:
    """
    Resume a pipeline run from a specific stage.

    Examples:
        pipeline rerun 20241218_120000 --from labeling
        pipeline rerun phase1_v1 --from create_splits
    """
    pipeline_config = _get_pipeline_config()
    pipeline_mod = _get_pipeline_runner()

    console.print(f"\n[bold cyan]Resuming Pipeline Run: {run_id}[/bold cyan]\n")

    # Load configuration
    try:
        project_root_path = get_project_root(project_root)
        config = pipeline_config.PipelineConfig.load_from_run_id(run_id, project_root_path)
        show_success(f"Loaded configuration for run: {run_id}")
    except FileNotFoundError as e:
        show_error(f"Run not found: {e}")
        raise typer.Exit(1)

    # Display configuration
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Symbols: {', '.join(config.symbols)}")
    console.print(f"  Description: {config.description}")

    # Map friendly names to stage names
    stage_map = {
        'data': 'data_generation',
        'cleaning': 'data_cleaning',
        'clean': 'data_cleaning',
        'features': 'feature_engineering',
        'labeling': 'initial_labeling',
        'labels': 'initial_labeling',
        'initial_labeling': 'initial_labeling',
        'initial-labeling': 'initial_labeling',
        'final_labels': 'final_labels',
        'final-labels': 'final_labels',
        'final': 'final_labels',
        'ga': 'ga_optimize',
        'optimize': 'ga_optimize',
        'splits': 'create_splits',
        'scaling': 'feature_scaling',
        'scale': 'feature_scaling',
        'datasets': 'build_datasets',
        'validate_scaled': 'validate_scaled',
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
        runner = pipeline_mod.PipelineRunner(config, resume=True)
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
