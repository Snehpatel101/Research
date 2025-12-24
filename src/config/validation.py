"""
Configuration validation for the ensemble trading pipeline.

This module provides the main validate_config() function that validates
all configuration values for consistency and correctness.
"""

from .barriers_config import (
    BARRIER_PARAMS,
    BARRIER_PARAMS_DEFAULT,
    PERCENTAGE_BARRIER_PARAMS,
    validate_barrier_params,
    get_max_bars_across_all_params,
)
from .splits import (
    PURGE_BARS,
    validate_splits_config,
)
from .features import (
    validate_mtf_config,
    validate_feature_thresholds,
    validate_stationarity_config,
    validate_drift_config,
)
from .sessions import validate_sessions_config
from .regime_config import validate_regime_config
from .labeling_config import validate_labeling_config


def validate_config() -> None:
    """
    Validate all configuration values for consistency and correctness.

    Raises
    ------
    ValueError
        If configuration values are invalid or inconsistent

    This function ensures:
    1. PURGE_BARS >= max(max_bars) across all horizons (prevents label leakage)
    2. Split ratios sum to 1.0
    3. All barrier parameters are positive
    4. Transaction costs are non-negative
    5. Tick values are positive
    6. Feature thresholds are valid
    7. Session configuration is valid
    8. Regime configuration is valid
    9. Labeling configuration is valid
    """
    errors = []

    # === Validate purge_bars >= max_bars (CRITICAL for leakage prevention) ===
    max_max_bars, max_bars_source = get_max_bars_across_all_params()

    if PURGE_BARS < max_max_bars:
        errors.append(
            f"PURGE_BARS ({PURGE_BARS}) must be >= max_bars ({max_max_bars}) from {max_bars_source} "
            f"to prevent label leakage. Set PURGE_BARS >= {max_max_bars}."
        )

    # === Validate splits configuration ===
    errors.extend(validate_splits_config())

    # === Validate barrier parameters ===
    errors.extend(validate_barrier_params())

    # === Validate feature thresholds ===
    errors.extend(validate_feature_thresholds())

    # === Validate MTF configuration ===
    errors.extend(validate_mtf_config())

    # === Validate stationarity configuration ===
    errors.extend(validate_stationarity_config())

    # === Validate drift configuration ===
    errors.extend(validate_drift_config())

    # === Validate sessions configuration ===
    errors.extend(validate_sessions_config())

    # === Validate regime configuration ===
    errors.extend(validate_regime_config())

    # === Validate labeling configuration ===
    errors.extend(validate_labeling_config())

    # === Raise all errors at once for comprehensive feedback ===
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


def validate_config_silent() -> list[str]:
    """
    Validate all configuration values without raising an exception.

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)

    This is useful for programmatic validation where you want to
    inspect errors rather than catch an exception.
    """
    errors = []

    # Validate purge_bars >= max_bars
    max_max_bars, max_bars_source = get_max_bars_across_all_params()
    if PURGE_BARS < max_max_bars:
        errors.append(
            f"PURGE_BARS ({PURGE_BARS}) must be >= max_bars ({max_max_bars}) from {max_bars_source}"
        )

    # Collect all validation errors
    errors.extend(validate_splits_config())
    errors.extend(validate_barrier_params())
    errors.extend(validate_feature_thresholds())
    errors.extend(validate_mtf_config())
    errors.extend(validate_stationarity_config())
    errors.extend(validate_drift_config())
    errors.extend(validate_sessions_config())
    errors.extend(validate_regime_config())
    errors.extend(validate_labeling_config())

    return errors
