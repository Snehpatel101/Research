"""
Core utilities for CLI run commands.

Provides configuration creation, lazy imports, and shared utilities
for pipeline execution commands.
"""
from pathlib import Path
from typing import Optional

from .utils import show_info


class LazyImports:
    """
    Singleton for lazy-loading CLI module dependencies.

    This pattern replaces module-level globals with a clean, testable singleton
    that ensures only one instance exists across the entire application.

    The singleton pattern prevents circular import issues while maintaining
    explicit dependency management. Each module is loaded on first access
    and cached for subsequent uses.

    Usage:
        >>> lazy = LazyImports()
        >>> config_module = lazy.pipeline_config
        >>> runner_module = lazy.pipeline_runner

    Thread Safety:
        This implementation is not thread-safe. Since CLI commands run
        sequentially in a single process, thread safety is not required.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        """
        Create or return the singleton instance.

        Returns
        -------
        LazyImports
            The singleton instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize the singleton (only runs once).

        Subsequent calls to __init__ are no-ops to preserve singleton state.
        """
        # Only initialize once per instance lifetime
        if LazyImports._initialized:
            return
        LazyImports._initialized = True

        # Private attributes to store lazy-loaded modules
        self._pipeline_config = None
        self._pipeline_runner = None
        self._presets_module = None
        self._model_config = None
        self._manifest = None

    @property
    def pipeline_config(self):
        """
        Lazy import pipeline_config module.

        Returns
        -------
        module
            The pipeline_config module from phase1
        """
        if self._pipeline_config is None:
            from ..phase1 import pipeline_config
            self._pipeline_config = pipeline_config
        return self._pipeline_config

    @property
    def pipeline_runner(self):
        """
        Lazy import pipeline module.

        Returns
        -------
        module
            The pipeline module containing PipelineRunner
        """
        if self._pipeline_runner is None:
            from .. import pipeline
            self._pipeline_runner = pipeline
        return self._pipeline_runner

    @property
    def presets(self):
        """
        Lazy import presets module.

        Returns
        -------
        module
            The presets module from phase1
        """
        if self._presets_module is None:
            from ..phase1 import presets
            self._presets_module = presets
        return self._presets_module

    @property
    def model_config(self):
        """
        Lazy import model_config module.

        Returns
        -------
        module
            The model_config module from phase1.config
        """
        if self._model_config is None:
            from ..phase1.config import model_config
            self._model_config = model_config
        return self._model_config

    @property
    def manifest(self):
        """
        Lazy import manifest module.

        Returns
        -------
        module
            The manifest module for artifact management
        """
        if self._manifest is None:
            from ..common import manifest
            self._manifest = manifest
        return self._manifest


# Backward compatibility functions (deprecated - use LazyImports directly)
def _get_pipeline_config():
    """Lazy import pipeline_config module (deprecated - use LazyImports().pipeline_config)."""
    return LazyImports().pipeline_config


def _get_pipeline_runner():
    """Lazy import pipeline module (deprecated - use LazyImports().pipeline_runner)."""
    return LazyImports().pipeline_runner


def _get_presets_module():
    """Lazy import presets module (deprecated - use LazyImports().presets)."""
    return LazyImports().presets


def _get_model_config():
    """Lazy import model_config module (deprecated - use LazyImports().model_config)."""
    return LazyImports().model_config


def _apply_preset_settings(
    preset: Optional[str],
    description: Optional[str],
    presets_mod,
) -> tuple[dict, Optional[str]]:
    """
    Apply preset configuration if specified.

    Args:
        preset: Preset name (scalping, day_trading, swing)
        description: User-provided description (may be None)
        presets_mod: Presets module

    Returns:
        Tuple of (config_kwargs dict, description string)

    Raises:
        ValueError: If preset is invalid
    """
    config_kwargs = {}

    if not preset:
        return config_kwargs, description

    # Validate and load preset
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

    return config_kwargs, description


def _parse_and_apply_symbols(
    symbols: Optional[str],
    config_kwargs: dict,
) -> None:
    """
    Parse symbols from CLI or auto-detect from data directory.

    Args:
        symbols: Comma-separated symbol list or None
        config_kwargs: Config dict to update (modified in-place)

    Raises:
        ValueError: If no symbols specified and none detected
    """
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


def _apply_basic_config_overrides(
    timeframe: Optional[str],
    horizons: Optional[str],
    feature_set: Optional[str],
    start: Optional[str],
    end: Optional[str],
    run_id: Optional[str],
    description: Optional[str],
    config_kwargs: dict,
) -> None:
    """
    Apply basic configuration overrides from CLI.

    Args:
        All CLI parameters and config_kwargs dict (modified in-place)
    """
    if timeframe is not None:
        config_kwargs['target_timeframe'] = timeframe

    if horizons is not None:
        config_kwargs['label_horizons'] = [int(h.strip()) for h in horizons.split(",")]

    if feature_set is not None:
        config_kwargs['feature_set'] = feature_set

    if start is not None:
        config_kwargs['start_date'] = start
    if end is not None:
        config_kwargs['end_date'] = end

    if run_id is not None:
        config_kwargs['run_id'] = run_id
    if description is not None:
        config_kwargs['description'] = description


def _apply_split_and_leakage_settings(
    train_ratio: Optional[float],
    val_ratio: Optional[float],
    test_ratio: Optional[float],
    purge_bars: Optional[int],
    embargo_bars: Optional[int],
    config_kwargs: dict,
) -> None:
    """
    Apply train/val/test split ratios and purge/embargo settings.

    Args:
        All ratio/leakage parameters and config_kwargs dict (modified in-place)
    """
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


def _apply_mtf_configuration(
    mtf_mode: Optional[str],
    mtf_timeframes: Optional[str],
    mtf_enable: Optional[bool],
    config_kwargs: dict,
) -> None:
    """
    Apply multi-timeframe (MTF) configuration.

    Args:
        MTF parameters and config_kwargs dict (modified in-place)
    """
    if mtf_mode is not None:
        config_kwargs['mtf_mode'] = mtf_mode
    if mtf_timeframes is not None:
        config_kwargs['mtf_timeframes'] = [tf.strip() for tf in mtf_timeframes.split(",")]
    if mtf_enable is not None:
        # If explicitly disabled, clear mtf_timeframes
        if not mtf_enable:
            config_kwargs['mtf_timeframes'] = []
            config_kwargs['mtf_mode'] = 'bars'  # Minimal mode when disabled


def _build_feature_toggles(
    enable_wavelets: Optional[bool],
    enable_microstructure: Optional[bool],
    enable_volume_features: Optional[bool],
    enable_volatility_features: Optional[bool],
) -> dict:
    """
    Build feature toggles dictionary.

    Args:
        Feature enable/disable flags

    Returns:
        Feature toggles dict (empty if no toggles specified)
    """
    feature_toggles = {}
    if enable_wavelets is not None:
        feature_toggles['wavelets'] = enable_wavelets
    if enable_microstructure is not None:
        feature_toggles['microstructure'] = enable_microstructure
    if enable_volume_features is not None:
        feature_toggles['volume'] = enable_volume_features
    if enable_volatility_features is not None:
        feature_toggles['volatility'] = enable_volatility_features
    return feature_toggles


def _build_barrier_overrides(
    k_up: Optional[float],
    k_down: Optional[float],
    max_bars: Optional[int],
) -> dict:
    """
    Build labeling barrier overrides dictionary.

    Args:
        Barrier parameters

    Returns:
        Barrier overrides dict (empty if no overrides specified)
    """
    barrier_overrides = {}
    if k_up is not None:
        barrier_overrides['k_up'] = k_up
    if k_down is not None:
        barrier_overrides['k_down'] = k_down
    if max_bars is not None:
        barrier_overrides['max_bars'] = max_bars
    return barrier_overrides


def _build_model_configuration(
    model_type: Optional[str],
    base_models: Optional[str],
    meta_learner: Optional[str],
    sequence_length: Optional[int],
) -> dict:
    """
    Build model configuration dictionary.

    Args:
        Model-related parameters

    Returns:
        Model config dict (empty if no model settings specified)
    """
    model_config_data = {}
    if model_type is not None:
        model_config_data['model_type'] = model_type
    if base_models is not None:
        model_config_data['base_models'] = [m.strip() for m in base_models.split(",")]
    if meta_learner is not None:
        model_config_data['meta_learner'] = meta_learner
    if sequence_length is not None:
        model_config_data['sequence_length'] = sequence_length
    return model_config_data


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
    config_kwargs = {'project_root': project_root_path}

    # Apply preset if specified
    try:
        preset_kwargs, description = _apply_preset_settings(preset, description, presets_mod)
        config_kwargs.update(preset_kwargs)
    except ValueError as e:
        raise ValueError(f"Invalid preset: {e}")

    # Parse and apply symbols (auto-detect if not specified)
    _parse_and_apply_symbols(symbols, config_kwargs)

    # Apply basic CLI overrides
    _apply_basic_config_overrides(
        timeframe, horizons, feature_set, start, end, run_id, description, config_kwargs
    )

    # Apply split ratios and purge/embargo settings
    _apply_split_and_leakage_settings(
        train_ratio, val_ratio, test_ratio, purge_bars, embargo_bars, config_kwargs
    )

    # Apply MTF configuration
    _apply_mtf_configuration(mtf_mode, mtf_timeframes, mtf_enable, config_kwargs)

    # Build and apply feature toggles
    feature_toggles = _build_feature_toggles(
        enable_wavelets, enable_microstructure, enable_volume_features, enable_volatility_features
    )
    if feature_toggles:
        config_kwargs['feature_toggles'] = feature_toggles

    # Build and apply barrier overrides
    barrier_overrides = _build_barrier_overrides(k_up, k_down, max_bars)
    if barrier_overrides:
        config_kwargs['barrier_overrides'] = barrier_overrides

    # Apply scaling options
    if scaler_type is not None:
        config_kwargs['scaler_type'] = scaler_type

    # Build and apply model configuration
    model_config_data = _build_model_configuration(
        model_type, base_models, meta_learner, sequence_length
    )
    if model_config_data:
        config_kwargs['model_config'] = model_config_data

    # Create and return config
    return pipeline_config.create_default_config(**config_kwargs)
