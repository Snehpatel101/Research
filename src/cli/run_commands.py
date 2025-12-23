"""
CLI Run Commands - run and rerun pipeline commands.
"""
from pathlib import Path
from typing import Optional, List

import typer
from rich.table import Table

from .utils import console, show_error, show_success, show_info, show_warning, get_project_root

# Lazy imports to avoid circular dependencies
_pipeline_config = None
_pipeline_runner = None
_presets_module = None


def _get_pipeline_config():
    """Lazy import pipeline_config module."""
    global _pipeline_config
    if _pipeline_config is None:
        from .. import pipeline_config
        _pipeline_config = pipeline_config
    return _pipeline_config


def _get_pipeline_runner():
    """Lazy import pipeline module."""
    global _pipeline_runner
    if _pipeline_runner is None:
        from .. import pipeline
        _pipeline_runner = pipeline
    return _pipeline_runner


def _get_presets_module():
    """Lazy import presets module."""
    global _presets_module
    if _presets_module is None:
        from .. import presets
        _presets_module = presets
    return _presets_module


def _create_config_from_args(
    preset: Optional[str],
    symbols: Optional[str],
    timeframe: Optional[str],
    horizons: Optional[str],
    start: Optional[str],
    end: Optional[str],
    run_id: Optional[str],
    description: Optional[str],
    train_ratio: Optional[float],
    val_ratio: Optional[float],
    test_ratio: Optional[float],
    purge_bars: Optional[int],
    embargo_bars: Optional[int],
    synthetic: bool,
    project_root_path: Path,
    pipeline_config,
    presets_mod
):
    """
    Create pipeline config from CLI arguments, applying preset if specified.

    Preset values are applied first, then CLI arguments override specific settings.
    This allows users to use a preset as a base and customize individual parameters.

    Parameters
    ----------
    preset : str, optional
        Trading preset name (scalping, day_trading, swing)
    symbols : str, optional
        Comma-separated list of symbols
    timeframe : str, optional
        Target timeframe for resampling
    horizons : str, optional
        Comma-separated label horizons
    Other parameters as documented in run_command

    Returns
    -------
    PipelineConfig
        Configured pipeline configuration object

    Raises
    ------
    ValueError
        If preset is invalid or configuration is invalid
    """
    # Start with base config kwargs
    config_kwargs = {
        'project_root': project_root_path,
        'use_synthetic_data': synthetic,
    }

    # Apply preset if specified
    preset_config = None
    if preset:
        try:
            presets_mod.validate_preset(preset)
            preset_config = presets_mod.get_preset(preset)
            show_info(f"Applying '{preset}' preset")

            # Map preset values to config kwargs
            config_kwargs['target_timeframe'] = preset_config.get('target_timeframe', '5min')
            config_kwargs['label_horizons'] = preset_config.get('horizons', [5, 20])
            config_kwargs['max_bars_ahead'] = preset_config.get('max_bars_ahead', 50)

            # Apply feature config from preset
            if 'feature_config' in preset_config:
                feat_config = preset_config['feature_config']
                if 'sma_periods' in feat_config:
                    config_kwargs['sma_periods'] = feat_config['sma_periods']
                if 'ema_periods' in feat_config:
                    config_kwargs['ema_periods'] = feat_config['ema_periods']
                if 'atr_periods' in feat_config:
                    config_kwargs['atr_periods'] = feat_config['atr_periods']
                if 'rsi_period' in feat_config:
                    config_kwargs['rsi_period'] = feat_config['rsi_period']

            # Set description based on preset if not provided
            if description is None:
                description = f"{preset_config.get('name', preset)} run"

        except ValueError as e:
            raise ValueError(f"Invalid preset: {e}")

    # Apply CLI overrides (these take precedence over preset)
    # Symbols
    if symbols is not None:
        config_kwargs['symbols'] = [s.strip().upper() for s in symbols.split(",")]
    elif 'symbols' not in config_kwargs:
        config_kwargs['symbols'] = ['MES', 'MGC']  # Default

    # Timeframe override
    if timeframe is not None:
        config_kwargs['target_timeframe'] = timeframe

    # Horizons override
    if horizons is not None:
        config_kwargs['label_horizons'] = [int(h.strip()) for h in horizons.split(",")]

    # Date range
    if start is not None:
        config_kwargs['start_date'] = start
    if end is not None:
        config_kwargs['end_date'] = end

    # Run metadata
    if run_id is not None:
        config_kwargs['run_id'] = run_id
    if description is not None:
        config_kwargs['description'] = description

    # Split ratios (only override if explicitly provided)
    if train_ratio is not None:
        config_kwargs['train_ratio'] = train_ratio
    if val_ratio is not None:
        config_kwargs['val_ratio'] = val_ratio
    if test_ratio is not None:
        config_kwargs['test_ratio'] = test_ratio

    # Purge/embargo (only override if explicitly provided)
    if purge_bars is not None:
        config_kwargs['purge_bars'] = purge_bars
        config_kwargs['auto_scale_purge_embargo'] = False  # Disable auto-scaling
    if embargo_bars is not None:
        config_kwargs['embargo_bars'] = embargo_bars
        config_kwargs['auto_scale_purge_embargo'] = False  # Disable auto-scaling

    # Create and return config
    return pipeline_config.create_default_config(**config_kwargs)


def run_command(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Comma-separated list of symbols to process (default: MES,MGC)"
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Trading preset: scalping, day_trading, swing (overrides defaults)"
    ),
    timeframe: Optional[str] = typer.Option(
        None,
        "--timeframe",
        "-t",
        help="Target timeframe for resampling (e.g., 1min, 5min, 15min)"
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
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of this run"
    ),
    train_ratio: Optional[float] = typer.Option(
        None,
        "--train-ratio",
        help="Training set ratio (default: 0.70)"
    ),
    val_ratio: Optional[float] = typer.Option(
        None,
        "--val-ratio",
        help="Validation set ratio (default: 0.15)"
    ),
    test_ratio: Optional[float] = typer.Option(
        None,
        "--test-ratio",
        help="Test set ratio (default: 0.15)"
    ),
    purge_bars: Optional[int] = typer.Option(
        None,
        "--purge-bars",
        help="Number of bars to purge at split boundaries"
    ),
    embargo_bars: Optional[int] = typer.Option(
        None,
        "--embargo-bars",
        help="Embargo period in bars (~1 day for 5-min data)"
    ),
    horizons: Optional[str] = typer.Option(
        None,
        "--horizons",
        help="Comma-separated label horizons (overrides preset)"
    ),
    synthetic: bool = typer.Option(
        False,
        "--synthetic",
        help="Generate synthetic data"
    ),
    project_root: Optional[str] = typer.Option(
        None,
        "--project-root",
        help="Project root directory"
    )
) -> None:
    """
    Run the complete Phase 1 pipeline.

    Examples:
        pipeline run --symbols MES,MGC --start 2020-01-01 --end 2024-12-31
        pipeline run --preset day_trading --symbols MES
        pipeline run --preset scalping --horizons 1,3,5
        pipeline run --run-id phase1_v1 --synthetic
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
            start=start,
            end=end,
            run_id=run_id,
            description=description,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            synthetic=synthetic,
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
        runner = pipeline_mod.PipelineRunner(config)
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


def rerun_command(
    run_id: str = typer.Argument(
        ...,
        help="Run ID to resume"
    ),
    from_stage: str = typer.Option(
        None,
        "--from",
        help="Stage to resume from (e.g., 'labels', 'features')"
    ),
    project_root: Optional[str] = typer.Option(
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
