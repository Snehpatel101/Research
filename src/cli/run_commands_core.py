"""
Core utilities for CLI run commands.

Provides configuration creation, lazy imports, and shared utilities
for pipeline execution commands.
"""
from pathlib import Path

from .utils import show_info

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
    preset: str | None,
    symbols: str | None,
    timeframe: str | None,
    horizons: str | None,
    feature_set: str | None,
    start: str | None,
    end: str | None,
    run_id: str | None,
    description: str | None,
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
    purge_bars: int | None,
    embargo_bars: int | None,
    # MTF settings
    mtf_mode: str | None,
    mtf_timeframes: str | None,
    mtf_enable: bool | None,
    # Feature toggles
    enable_wavelets: bool | None,
    enable_microstructure: bool | None,
    enable_volume_features: bool | None,
    enable_volatility_features: bool | None,
    # Labeling parameters
    k_up: float | None,
    k_down: float | None,
    max_bars: int | None,
    # Scaling options
    scaler_type: str | None,
    # Model selection (Phase 2+)
    model_type: str | None,
    base_models: str | None,
    meta_learner: str | None,
    sequence_length: int | None,
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
            # Import canonical horizons as fallback
            from src.common.horizon_config import ACTIVE_HORIZONS
            config_kwargs['label_horizons'] = preset_config.get('horizons', list(ACTIVE_HORIZONS))
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
