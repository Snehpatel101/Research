"""
Trading presets for the ensemble price prediction pipeline.

Provides pre-configured trading styles (scalping, day trading, swing) that can be
applied to PipelineConfig to quickly set up appropriate parameters for different
trading strategies.

Each preset defines:
- Target timeframe for resampling
- Labeling horizons for model training
- Trading sessions to include
- Labeling strategy type
- Barrier multiplier for adjusting volatility thresholds

Usage:
------
>>> from presets import TradingPreset, get_preset, apply_preset_to_config
>>>
>>> # Get preset configuration
>>> scalping_config = get_preset('scalping')
>>>
>>> # Apply preset to existing pipeline config
>>> from pipeline_config import PipelineConfig
>>> config = PipelineConfig()
>>> config_dict = config.to_dict()
>>> updated_config = apply_preset_to_config(TradingPreset.SCALPING, config_dict)
"""

from enum import Enum
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TradingPreset(Enum):
    """
    Enumeration of available trading presets.

    Each preset represents a distinct trading style with appropriate parameters:
    - SCALPING: Ultra-short-term trading with tight barriers and fast execution
    - DAY_TRADING: Intraday trading with balanced parameters
    - SWING: Multi-day position holding with wider barriers
    """

    SCALPING = 'scalping'
    DAY_TRADING = 'day_trading'
    SWING = 'swing'


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================
# Each preset provides parameters optimized for a specific trading style.
# These configurations are designed to be merged with PipelineConfig.

PRESET_CONFIGS: Dict[TradingPreset, Dict[str, Any]] = {
    # =========================================================================
    # SCALPING PRESET
    # =========================================================================
    # Ultra-short-term trading targeting quick price movements.
    # Characteristics:
    # - 1-minute bars for maximum granularity
    # - Very short horizons (1, 5 bars)
    # - Only New York session (most liquid for futures)
    # - Threshold labeling for fast execution signals
    # - Tighter barriers (0.7x) to capture small moves
    TradingPreset.SCALPING: {
        'name': 'Scalping',
        'description': 'Ultra-short-term trading with tight barriers and fast execution',

        # Timeframe and horizons
        'target_timeframe': '1min',
        'horizons': [1, 5],

        # Sessions - only most liquid
        'sessions': ['new_york'],

        # Labeling configuration
        'labeling_strategy': 'threshold',
        'barrier_multiplier': 0.7,

        # Additional parameters
        'max_bars_ahead': 10,
        'min_trade_duration_bars': 1,

        # Feature adjustments for scalping
        'feature_config': {
            'sma_periods': [5, 10, 20],
            'ema_periods': [5, 9, 12],
            'atr_periods': [5, 10],
            'rsi_period': 7,
        },

        # Risk parameters
        'risk_config': {
            'max_positions': 1,
            'stop_loss_atr_mult': 0.5,
            'take_profit_atr_mult': 0.7,
        },
    },

    # =========================================================================
    # DAY TRADING PRESET
    # =========================================================================
    # Intraday trading with positions closed before market close.
    # Characteristics:
    # - 5-minute bars for balance of signal quality and granularity
    # - Medium horizons (5, 20 bars = 25 min to 1.5 hours)
    # - New York and London sessions (primary trading hours)
    # - Triple barrier labeling for directional signals
    # - Default barriers (1.0x)
    TradingPreset.DAY_TRADING: {
        'name': 'Day Trading',
        'description': 'Intraday trading with triple barrier labeling',

        # Timeframe and horizons
        'target_timeframe': '5min',
        'horizons': [5, 20],

        # Sessions - primary trading hours
        'sessions': ['new_york', 'london'],

        # Labeling configuration
        'labeling_strategy': 'triple_barrier',
        'barrier_multiplier': 1.0,

        # Additional parameters
        'max_bars_ahead': 50,
        'min_trade_duration_bars': 3,

        # Feature adjustments for day trading
        'feature_config': {
            'sma_periods': [10, 20, 50, 100],
            'ema_periods': [9, 21, 50],
            'atr_periods': [7, 14, 21],
            'rsi_period': 14,
        },

        # Risk parameters
        'risk_config': {
            'max_positions': 2,
            'stop_loss_atr_mult': 1.0,
            'take_profit_atr_mult': 1.5,
        },
    },

    # =========================================================================
    # SWING PRESET
    # =========================================================================
    # Multi-day position trading with wider barriers.
    # Characteristics:
    # - 15-minute or 60-minute bars for reduced noise
    # - Longer horizons (20, 60, 120 bars)
    # - All major sessions (24-hour coverage)
    # - Triple barrier labeling for trend following
    # - Wider barriers (1.3x) to avoid noise
    TradingPreset.SWING: {
        'name': 'Swing Trading',
        'description': 'Multi-day position trading with wider barriers',

        # Timeframe and horizons
        'target_timeframe': '15min',
        'horizons': [20, 60, 120],

        # Sessions - all major sessions for global coverage
        'sessions': ['new_york', 'london', 'asia'],

        # Labeling configuration
        'labeling_strategy': 'triple_barrier',
        'barrier_multiplier': 1.3,

        # Additional parameters
        'max_bars_ahead': 150,
        'min_trade_duration_bars': 10,

        # Feature adjustments for swing trading
        'feature_config': {
            'sma_periods': [20, 50, 100, 200],
            'ema_periods': [21, 50, 100],
            'atr_periods': [14, 21, 50],
            'rsi_period': 14,
        },

        # Risk parameters
        'risk_config': {
            'max_positions': 3,
            'stop_loss_atr_mult': 1.5,
            'take_profit_atr_mult': 2.0,
        },
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def validate_preset(preset_name: str) -> bool:
    """
    Validate that a preset name is recognized.

    Parameters
    ----------
    preset_name : str
        Name of the preset to validate (e.g., 'scalping', 'day_trading', 'swing')

    Returns
    -------
    bool
        True if preset is valid

    Raises
    ------
    ValueError
        If preset name is not recognized

    Examples
    --------
    >>> validate_preset('scalping')
    True
    >>> validate_preset('invalid_preset')  # raises ValueError
    """
    preset_name = preset_name.lower().strip()
    valid_names = [p.value for p in TradingPreset]

    if preset_name not in valid_names:
        raise ValueError(
            f"Unknown preset: '{preset_name}'. "
            f"Valid presets are: {valid_names}"
        )

    return True


def list_available_presets() -> List[str]:
    """
    List all available trading presets.

    Returns
    -------
    List[str]
        List of preset names

    Examples
    --------
    >>> list_available_presets()
    ['scalping', 'day_trading', 'swing']
    """
    return [p.value for p in TradingPreset]


def get_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get configuration dictionary for a trading preset.

    Parameters
    ----------
    preset_name : str
        Name of the preset (e.g., 'scalping', 'day_trading', 'swing')
        Case-insensitive.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary for the preset

    Raises
    ------
    ValueError
        If preset name is not recognized

    Examples
    --------
    >>> config = get_preset('scalping')
    >>> config['target_timeframe']
    '1min'
    >>> config['horizons']
    [1, 5]
    """
    validate_preset(preset_name)

    preset_name = preset_name.lower().strip()
    preset_enum = TradingPreset(preset_name)

    # Return a copy to prevent modification of the original
    return PRESET_CONFIGS[preset_enum].copy()


def _get_preset_enum(preset: TradingPreset | str) -> TradingPreset:
    """
    Convert string or enum to TradingPreset enum.

    Parameters
    ----------
    preset : TradingPreset or str
        Preset as enum or string name

    Returns
    -------
    TradingPreset
        The corresponding enum value

    Raises
    ------
    ValueError
        If preset is not recognized
    """
    if isinstance(preset, TradingPreset):
        return preset

    if isinstance(preset, str):
        validate_preset(preset)
        return TradingPreset(preset.lower().strip())

    raise ValueError(
        f"preset must be TradingPreset or str, got {type(preset).__name__}"
    )


def apply_preset_to_config(
    preset: TradingPreset | str,
    base_config: Dict[str, Any],
    override_conflicts: bool = True
) -> Dict[str, Any]:
    """
    Apply a trading preset to an existing configuration dictionary.

    Merges preset values into the base configuration. By default, preset values
    override conflicting base config values.

    Parameters
    ----------
    preset : TradingPreset or str
        The trading preset to apply
    base_config : Dict[str, Any]
        Base configuration dictionary (typically from PipelineConfig.to_dict())
    override_conflicts : bool, default True
        If True, preset values override conflicting base config values.
        If False, base config values are preserved.

    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary

    Examples
    --------
    >>> from pipeline_config import PipelineConfig
    >>> config = PipelineConfig()
    >>> base = config.to_dict()
    >>> updated = apply_preset_to_config(TradingPreset.SCALPING, base)
    >>> updated['target_timeframe']
    '1min'
    >>> updated['label_horizons']
    [1, 5]

    Notes
    -----
    The following base_config keys are updated from the preset:
    - target_timeframe
    - label_horizons (from preset 'horizons')
    - max_bars_ahead
    - sma_periods, ema_periods, atr_periods, rsi_period (from feature_config)
    - bar_resolution (synced with target_timeframe)

    Additional preset-specific keys are added:
    - preset_name
    - preset_sessions
    - preset_labeling_strategy
    - preset_barrier_multiplier
    - preset_risk_config
    """
    preset_enum = _get_preset_enum(preset)
    preset_config = PRESET_CONFIGS[preset_enum].copy()

    # Create a copy of base config to avoid modifying the original
    result = base_config.copy()

    # Map preset keys to PipelineConfig keys
    key_mapping = {
        'target_timeframe': 'target_timeframe',
        'horizons': 'label_horizons',
        'max_bars_ahead': 'max_bars_ahead',
    }

    # Apply mapped keys
    for preset_key, config_key in key_mapping.items():
        if preset_key in preset_config:
            if override_conflicts or config_key not in result:
                result[config_key] = preset_config[preset_key]

    # Sync bar_resolution with target_timeframe (backward compatibility)
    if 'target_timeframe' in result:
        result['bar_resolution'] = result['target_timeframe']

    # Apply feature config if present
    if 'feature_config' in preset_config:
        feature_config = preset_config['feature_config']
        for key, value in feature_config.items():
            if override_conflicts or key not in result:
                result[key] = value

    # Add preset-specific metadata
    result['preset_name'] = preset_config.get('name', preset_enum.value)
    result['preset_description'] = preset_config.get('description', '')
    result['preset_sessions'] = preset_config.get('sessions', [])
    result['preset_labeling_strategy'] = preset_config.get('labeling_strategy', 'triple_barrier')
    result['preset_barrier_multiplier'] = preset_config.get('barrier_multiplier', 1.0)
    result['preset_risk_config'] = preset_config.get('risk_config', {})
    result['preset_min_trade_duration_bars'] = preset_config.get('min_trade_duration_bars', 1)

    logger.debug(
        f"Applied preset '{preset_enum.value}' to config: "
        f"timeframe={result.get('target_timeframe')}, "
        f"horizons={result.get('label_horizons')}"
    )

    return result


def get_preset_summary(preset_name: str) -> str:
    """
    Get a human-readable summary of a preset's configuration.

    Parameters
    ----------
    preset_name : str
        Name of the preset

    Returns
    -------
    str
        Formatted summary string

    Examples
    --------
    >>> print(get_preset_summary('day_trading'))
    Preset: Day Trading
    ...
    """
    config = get_preset(preset_name)

    summary = f"""
Preset: {config.get('name', preset_name)}
{'=' * 50}
Description: {config.get('description', 'N/A')}

Timeframe & Horizons:
  - Target Timeframe: {config.get('target_timeframe', 'N/A')}
  - Horizons: {config.get('horizons', [])}
  - Max Bars Ahead: {config.get('max_bars_ahead', 'N/A')}

Sessions:
  - Active Sessions: {', '.join(config.get('sessions', []))}

Labeling:
  - Strategy: {config.get('labeling_strategy', 'N/A')}
  - Barrier Multiplier: {config.get('barrier_multiplier', 1.0)}

Feature Config:
  - SMA Periods: {config.get('feature_config', {}).get('sma_periods', [])}
  - EMA Periods: {config.get('feature_config', {}).get('ema_periods', [])}
  - ATR Periods: {config.get('feature_config', {}).get('atr_periods', [])}
  - RSI Period: {config.get('feature_config', {}).get('rsi_period', 14)}

Risk Config:
  - Max Positions: {config.get('risk_config', {}).get('max_positions', 'N/A')}
  - Stop Loss ATR Mult: {config.get('risk_config', {}).get('stop_loss_atr_mult', 'N/A')}
  - Take Profit ATR Mult: {config.get('risk_config', {}).get('take_profit_atr_mult', 'N/A')}
"""
    return summary.strip()


def get_adjusted_barrier_params(
    preset: TradingPreset | str,
    symbol: str,
    horizon: int
) -> Dict[str, Any]:
    """
    Get barrier parameters adjusted by the preset's barrier multiplier.

    Fetches base barrier parameters from config.get_barrier_params() and
    applies the preset's barrier_multiplier to k_up and k_down values.

    Parameters
    ----------
    preset : TradingPreset or str
        The trading preset
    symbol : str
        Trading symbol (e.g., 'MES', 'MGC')
    horizon : int
        Labeling horizon

    Returns
    -------
    Dict[str, Any]
        Adjusted barrier parameters with keys:
        - k_up: Adjusted upper barrier multiplier
        - k_down: Adjusted lower barrier multiplier
        - max_bars: Maximum bars for label (unchanged)
        - description: Updated description

    Examples
    --------
    >>> params = get_adjusted_barrier_params('scalping', 'MES', 5)
    >>> params['k_up']  # Base k_up * 0.7
    """
    # Import here to avoid circular imports
    from config import get_barrier_params

    preset_enum = _get_preset_enum(preset)
    preset_config = PRESET_CONFIGS[preset_enum]
    multiplier = preset_config.get('barrier_multiplier', 1.0)

    # Get base barrier params
    base_params = get_barrier_params(symbol, horizon)

    # Apply multiplier
    adjusted_params = {
        'k_up': round(base_params['k_up'] * multiplier, 2),
        'k_down': round(base_params['k_down'] * multiplier, 2),
        'max_bars': base_params['max_bars'],
        'description': (
            f"{preset_enum.value.title()} adjusted: "
            f"{base_params.get('description', '')} "
            f"(barrier_mult={multiplier})"
        ),
        'barrier_multiplier': multiplier,
        'base_params': base_params,
    }

    return adjusted_params


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'TradingPreset',
    'PRESET_CONFIGS',
    'validate_preset',
    'list_available_presets',
    'get_preset',
    'apply_preset_to_config',
    'get_preset_summary',
    'get_adjusted_barrier_params',
]
