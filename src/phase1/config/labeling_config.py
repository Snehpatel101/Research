"""
Labeling strategy configuration.

This module contains configuration for various labeling strategies
including triple-barrier, directional, threshold, regression, and meta-labeling.
"""

from enum import Enum


class LabelingStrategyType(Enum):
    """Enumeration of available labeling strategies."""
    TRIPLE_BARRIER = 'triple_barrier'
    ADAPTIVE_TRIPLE_BARRIER = 'adaptive_triple_barrier'
    DIRECTIONAL = 'directional'
    THRESHOLD = 'threshold'
    REGRESSION = 'regression'
    META = 'meta'


# Default labeling strategy for the pipeline
DEFAULT_LABELING_STRATEGY = LabelingStrategyType.TRIPLE_BARRIER

# Strategy-specific default configurations
LABELING_STRATEGY_CONFIGS = {
    # Triple-barrier defaults are in BARRIER_PARAMS and BARRIER_PARAMS_DEFAULT
    LabelingStrategyType.TRIPLE_BARRIER: {
        'atr_column': 'atr_14',
        # k_up, k_down, max_bars come from get_barrier_params()
    },

    # Adaptive triple-barrier: Regime-adjusted barriers
    # Uses REGIME_BARRIER_ADJUSTMENTS and get_regime_adjusted_barriers()
    LabelingStrategyType.ADAPTIVE_TRIPLE_BARRIER: {
        'atr_column': 'atr_14',
        'volatility_regime_col': 'volatility_regime',
        'trend_regime_col': 'trend_regime',
        'structure_regime_col': 'structure_regime',
        # k_up, k_down, max_bars are dynamically adjusted per regime
    },

    # Directional labeling: Sign of forward return
    LabelingStrategyType.DIRECTIONAL: {
        'threshold': 0.0,        # Minimum return to be non-neutral (0 = any direction)
        'use_log_returns': False,
    },

    # Threshold labeling: Percentage-based barriers
    LabelingStrategyType.THRESHOLD: {
        'pct_up': 0.005,         # 0.5% upper threshold
        'pct_down': 0.005,       # 0.5% lower threshold
        'max_bars': 20,
    },

    # Regression labeling: Continuous return targets
    LabelingStrategyType.REGRESSION: {
        'use_log_returns': False,
        'winsorize_pct': 0.01,   # Clip extreme 1% tails
        'scale_factor': 100.0,   # Return as percentage
    },

    # Meta-labeling: Confidence on primary signals
    LabelingStrategyType.META: {
        'primary_signal_column': 'primary_label',
        'bet_size_method': 'probability',
    },
}

# =============================================================================
# LABEL BALANCE CONSTRAINTS (GA optimization)
# =============================================================================
# These thresholds prevent extreme class imbalance in optimized labels.
# The neutral class represents "hold" / timeout positions - important for:
# 1. Avoiding overtrading (transaction costs)
# 2. Filtering low-confidence signals
# 3. Maintaining realistic signal rates
LABEL_BALANCE_CONSTRAINTS = {
    'min_long_pct': 0.05,            # Minimum long share of total samples
    'min_short_pct': 0.05,           # Minimum short share of total samples
    'min_neutral_pct': 0.10,         # MINIMUM neutral share (HARD constraint)
    'target_neutral_low': 0.20,      # Target neutral range lower bound
    'target_neutral_high': 0.30,     # Target neutral range upper bound
    'max_neutral_pct': 0.40,         # Maximum neutral (too few signals)
    'min_short_signal_ratio': 0.10,  # Minimum short share among signals
    'max_short_signal_ratio': 0.90,  # Maximum short share among signals
    'min_any_class_pct': 0.10,       # Minimum for ANY class (fail-safe)
}


def get_labeling_strategy_config(
    strategy: LabelingStrategyType | str
) -> dict:
    """
    Get default configuration for a labeling strategy.

    Parameters
    ----------
    strategy : LabelingStrategyType or str
        The labeling strategy type

    Returns
    -------
    dict
        Default configuration for the strategy

    Raises
    ------
    ValueError
        If strategy is not recognized
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = LabelingStrategyType(strategy)
        except ValueError:
            valid_values = [t.value for t in LabelingStrategyType]
            raise ValueError(
                f"Unknown labeling strategy: '{strategy}'. "
                f"Valid values are: {valid_values}"
            )

    if strategy not in LABELING_STRATEGY_CONFIGS:
        raise ValueError(
            f"No configuration found for strategy: {strategy}. "
            f"Add it to LABELING_STRATEGY_CONFIGS."
        )

    return LABELING_STRATEGY_CONFIGS[strategy].copy()


# =============================================================================
# MULTI-LABEL CONFIGURATION
# =============================================================================
# Support for generating multiple label types per dataset.
# This is useful for:
# 1. Comparing labeling strategies on the same data
# 2. Multi-task learning with multiple prediction targets
# 3. Ensemble labeling where models vote on different labels

MULTI_LABEL_CONFIG = {
    # Enable multi-label generation
    'enabled': False,

    # List of strategies to apply
    # Each entry can be a LabelingStrategyType or a dict with type and overrides
    'strategies': [
        LabelingStrategyType.TRIPLE_BARRIER,
        LabelingStrategyType.DIRECTIONAL,
    ],

    # Column naming convention
    # {strategy}: strategy type name
    # {horizon}: horizon value
    'column_pattern': 'label_{strategy}_h{horizon}',
}


def get_multi_label_config() -> dict:
    """
    Get a copy of the multi-label configuration.

    Returns
    -------
    dict
        Copy of MULTI_LABEL_CONFIG
    """
    import copy
    return copy.deepcopy(MULTI_LABEL_CONFIG)


def validate_labeling_config() -> list[str]:
    """
    Validate labeling configuration values.

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate all strategy configs
    for strategy_type, config in LABELING_STRATEGY_CONFIGS.items():
        if not isinstance(config, dict):
            errors.append(
                f"LABELING_STRATEGY_CONFIGS[{strategy_type}] must be a dict"
            )

    # Validate threshold strategy
    threshold_config = LABELING_STRATEGY_CONFIGS.get(LabelingStrategyType.THRESHOLD, {})
    if threshold_config.get('pct_up', 0) <= 0:
        errors.append("THRESHOLD strategy 'pct_up' must be positive")
    if threshold_config.get('pct_down', 0) <= 0:
        errors.append("THRESHOLD strategy 'pct_down' must be positive")
    if threshold_config.get('max_bars', 0) <= 0:
        errors.append("THRESHOLD strategy 'max_bars' must be positive")

    # Validate regression strategy
    regression_config = LABELING_STRATEGY_CONFIGS.get(LabelingStrategyType.REGRESSION, {})
    winsorize = regression_config.get('winsorize_pct', 0)
    if not (0 <= winsorize < 0.5):
        errors.append(
            f"REGRESSION strategy 'winsorize_pct' must be in [0, 0.5), got {winsorize}"
        )

    # Validate multi-label config
    if MULTI_LABEL_CONFIG.get('enabled', False):
        strategies = MULTI_LABEL_CONFIG.get('strategies', [])
        if not strategies:
            errors.append("MULTI_LABEL_CONFIG has no strategies defined")
        for strategy in strategies:
            if isinstance(strategy, LabelingStrategyType):
                if strategy not in LABELING_STRATEGY_CONFIGS:
                    errors.append(
                        f"MULTI_LABEL_CONFIG strategy {strategy} not in LABELING_STRATEGY_CONFIGS"
                    )

    return errors
