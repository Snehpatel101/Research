"""
CLI Run Commands - run and rerun pipeline commands.

Provides fully configurable CLI for the Phase 1 data pipeline and
Phase 2+ model training with support for:
- MTF (Multi-Timeframe) settings
- Feature toggles (wavelets, microstructure, etc.)
- Labeling parameters (barriers, horizons)
- Split ratios (train/val/test)
- Scaling options
- Model type selection (for Phase 2+)
- Ensemble configuration
"""
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .utils import console, show_error, show_success, show_info, show_warning, get_project_root

# Lazy imports to avoid circular dependencies
_pipeline_config = None
_pipeline_runner = None
_presets_module = None
_model_config = None


def _get_pipeline_config():
    """Lazy import pipeline_config module."""
    global _pipeline_config
    if _pipeline_config is None:
        from ..phase1 import pipeline_config
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
        from ..phase1 import presets
        _presets_module = presets
    return _presets_module


def _get_model_config():
    """Lazy import model_config module."""
    global _model_config
    if _model_config is None:
        from ..phase1.config import model_config
        _model_config = model_config
    return _model_config


def _create_config_from_args(
    preset: Optional[str],
    symbols: Optional[str],
    timeframe: Optional[str],
    horizons: Optional[str],
    feature_set: Optional[str],
    start: Optional[str],
    end: Optional[str],
    run_id: Optional[str],
    description: Optional[str],
    train_ratio: Optional[float],
    val_ratio: Optional[float],
    test_ratio: Optional[float],
    purge_bars: Optional[int],
    embargo_bars: Optional[int],
    # MTF settings
    mtf_mode: Optional[str],
    mtf_timeframes: Optional[str],
    mtf_enable: Optional[bool],
    # Feature toggles
    enable_wavelets: Optional[bool],
    enable_microstructure: Optional[bool],
    enable_volume_features: Optional[bool],
    enable_volatility_features: Optional[bool],
    # Labeling parameters
    k_up: Optional[float],
    k_down: Optional[float],
    max_bars: Optional[int],
    # Scaling options
    scaler_type: Optional[str],
    # Model selection (Phase 2+)
    model_type: Optional[str],
    base_models: Optional[str],
    meta_learner: Optional[str],
    sequence_length: Optional[int],
    # Common
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
    mtf_mode : str, optional
        MTF mode: 'bars', 'indicators', or 'both'
    mtf_timeframes : str, optional
        Comma-separated MTF timeframes (e.g., '15min,30min,1h,4h')
    mtf_enable : bool, optional
        Enable/disable MTF feature generation
    enable_wavelets : bool, optional
        Enable wavelet decomposition features
    enable_microstructure : bool, optional
        Enable microstructure features (bid-ask, order flow)
    enable_volume_features : bool, optional
        Enable volume-based features
    enable_volatility_features : bool, optional
        Enable volatility features
    k_up : float, optional
        Upper barrier multiplier (overrides symbol-specific defaults)
    k_down : float, optional
        Lower barrier multiplier (overrides symbol-specific defaults)
    max_bars : int, optional
        Maximum bars for label timeout
    scaler_type : str, optional
        Scaler type: 'robust', 'standard', 'minmax', 'quantile', 'none'
    model_type : str, optional
        Target model type for Phase 2 (e.g., 'xgboost', 'lstm', 'ensemble')
    base_models : str, optional
        Comma-separated base models for ensemble
    meta_learner : str, optional
        Meta-learner for ensemble stacking
    sequence_length : int, optional
        Sequence length for sequential models

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
            config_kwargs['label_horizons'] = preset_config.get('horizons', [5, 10, 15, 20])
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
    # Symbols - auto-detect from available data if not specified
    if symbols is not None:
        config_kwargs['symbols'] = [s.strip().upper() for s in symbols.split(",")]
    elif 'symbols' not in config_kwargs:
        from src.phase1.config.runtime import detect_available_symbols
        detected = detect_available_symbols()
        if detected:
            config_kwargs['symbols'] = detected
            show_info(f"Auto-detected symbols from data: {', '.join(detected)}")
        else:
            raise ValueError(
                "No symbols specified and no data files found in data/raw/. "
                "Use --symbols to specify symbols or add {SYMBOL}_1m.parquet files."
            )

    # Timeframe override
    if timeframe is not None:
        config_kwargs['target_timeframe'] = timeframe

    # Horizons override
    if horizons is not None:
        config_kwargs['label_horizons'] = [int(h.strip()) for h in horizons.split(",")]

    if feature_set is not None:
        config_kwargs['feature_set'] = feature_set

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

    # MTF settings
    if mtf_mode is not None:
        config_kwargs['mtf_mode'] = mtf_mode
    if mtf_timeframes is not None:
        config_kwargs['mtf_timeframes'] = [tf.strip() for tf in mtf_timeframes.split(",")]
    if mtf_enable is not None:
        # If explicitly disabled, clear mtf_timeframes
        if not mtf_enable:
            config_kwargs['mtf_timeframes'] = []
            config_kwargs['mtf_mode'] = 'bars'  # Minimal mode when disabled

    # Feature toggles - stored for use by feature engineering stage
    feature_toggles = {}
    if enable_wavelets is not None:
        feature_toggles['wavelets'] = enable_wavelets
    if enable_microstructure is not None:
        feature_toggles['microstructure'] = enable_microstructure
    if enable_volume_features is not None:
        feature_toggles['volume'] = enable_volume_features
    if enable_volatility_features is not None:
        feature_toggles['volatility'] = enable_volatility_features
    if feature_toggles:
        config_kwargs['feature_toggles'] = feature_toggles

    # Labeling parameters - custom barrier overrides
    barrier_overrides = {}
    if k_up is not None:
        barrier_overrides['k_up'] = k_up
    if k_down is not None:
        barrier_overrides['k_down'] = k_down
    if max_bars is not None:
        barrier_overrides['max_bars'] = max_bars
    if barrier_overrides:
        config_kwargs['barrier_overrides'] = barrier_overrides

    # Scaling options
    if scaler_type is not None:
        config_kwargs['scaler_type'] = scaler_type

    # Model selection (Phase 2+ - stored for downstream use)
    model_config_data = {}
    if model_type is not None:
        model_config_data['model_type'] = model_type
    if base_models is not None:
        model_config_data['base_models'] = [m.strip() for m in base_models.split(",")]
    if meta_learner is not None:
        model_config_data['meta_learner'] = meta_learner
    if sequence_length is not None:
        model_config_data['sequence_length'] = sequence_length
    if model_config_data:
        config_kwargs['model_config'] = model_config_data

    # Create and return config
    return pipeline_config.create_default_config(**config_kwargs)


def run_command(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Comma-separated list of symbols (auto-detected from data/raw/ if not specified)"
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
    feature_set: Optional[str] = typer.Option(
        None,
        "--feature-set",
        help="Feature set: core_min, core_full, mtf_plus, boosting_optimal, neural_optimal, etc."
    ),
    # MTF settings
    mtf_mode: Optional[str] = typer.Option(
        None,
        "--mtf-mode",
        help="MTF mode: bars, indicators, or both (default: both)"
    ),
    mtf_timeframes: Optional[str] = typer.Option(
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
    enable_wavelets: Optional[bool] = typer.Option(
        None,
        "--enable-wavelets/--disable-wavelets",
        help="Enable/disable wavelet decomposition features"
    ),
    enable_microstructure: Optional[bool] = typer.Option(
        None,
        "--enable-microstructure/--disable-microstructure",
        help="Enable/disable microstructure features (bid-ask, order flow)"
    ),
    enable_volume: Optional[bool] = typer.Option(
        None,
        "--enable-volume/--disable-volume",
        help="Enable/disable volume-based features"
    ),
    enable_volatility: Optional[bool] = typer.Option(
        None,
        "--enable-volatility/--disable-volatility",
        help="Enable/disable volatility features"
    ),
    # Labeling parameters
    k_up: Optional[float] = typer.Option(
        None,
        "--k-up",
        help="Upper barrier multiplier (overrides symbol-specific defaults)"
    ),
    k_down: Optional[float] = typer.Option(
        None,
        "--k-down",
        help="Lower barrier multiplier (overrides symbol-specific defaults)"
    ),
    max_bars: Optional[int] = typer.Option(
        None,
        "--max-bars",
        help="Maximum bars for label timeout (overrides defaults)"
    ),
    # Scaling options
    scaler_type: Optional[str] = typer.Option(
        None,
        "--scaler-type",
        help="Scaler type: robust, standard, minmax, quantile, none (default: robust)"
    ),
    # Model selection (Phase 2+)
    model_type: Optional[str] = typer.Option(
        None,
        "--model-type",
        help="Target model type: xgboost, lightgbm, lstm, transformer, ensemble, etc."
    ),
    base_models: Optional[str] = typer.Option(
        None,
        "--base-models",
        help="Comma-separated base models for ensemble (e.g., 'xgboost,lstm,transformer')"
    ),
    meta_learner: Optional[str] = typer.Option(
        None,
        "--meta-learner",
        help="Meta-learner for ensemble stacking (e.g., logistic, xgboost)"
    ),
    sequence_length: Optional[int] = typer.Option(
        None,
        "--sequence-length",
        help="Sequence length for sequential models (LSTM, Transformer)"
    ),
    project_root: Optional[str] = typer.Option(
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
    from_stage: Optional[str] = typer.Option(
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


def models_command() -> None:
    """
    List available model types and their data requirements.

    Shows all supported models for Phase 2 training with their
    recommended feature sets, scaling requirements, and sequence lengths.
    """
    model_config = _get_model_config()

    console.print("\n[bold cyan]Available Model Types[/bold cyan]\n")

    # Create table for models
    table = Table(show_header=True, title="Model Data Requirements")
    table.add_column("Model", style="cyan")
    table.add_column("Family", style="yellow")
    table.add_column("Feature Set", style="green")
    table.add_column("Scaling", style="magenta")
    table.add_column("Sequences", style="blue")
    table.add_column("Description")

    for name in model_config.get_all_model_names():
        req = model_config.get_model_requirements(name)
        table.add_row(
            name,
            req.family.value,
            req.feature_set,
            req.scaler_type.value if req.requires_scaling else "none",
            str(req.sequence_length) if req.requires_sequences else "N/A",
            req.description[:50] + "..." if len(req.description) > 50 else req.description
        )

    console.print(table)

    # Show ensembles
    console.print("\n[bold cyan]Pre-defined Ensembles[/bold cyan]\n")

    ensemble_table = Table(show_header=True, title="Ensemble Configurations")
    ensemble_table.add_column("Ensemble", style="cyan")
    ensemble_table.add_column("Base Models", style="green")
    ensemble_table.add_column("Meta-Learner", style="yellow")
    ensemble_table.add_column("Description")

    for name in model_config.get_all_ensemble_names():
        ens = model_config.get_ensemble_config(name)
        ensemble_table.add_row(
            name,
            ", ".join(ens.base_models),
            ens.meta_learner,
            ens.description
        )

    console.print(ensemble_table)

    console.print("\n[dim]Use --model-type with 'pipeline run' to prepare data for specific models.[/dim]")
    console.print("[dim]Example: pipeline run --model-type lstm --feature-set neural_optimal[/dim]\n")
