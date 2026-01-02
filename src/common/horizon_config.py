"""
Dynamic Horizon Configuration Module

This module provides configuration and utility functions for the dynamic horizon
labeling system. It supports:
- Configurable horizons for triple-barrier labeling
- Timeframe-aware horizon scaling
- Auto-scaling of purge and embargo bars (with timeframe awareness)
- Dynamic barrier parameter generation
- HorizonConfig dataclass for encapsulating horizon settings

Usage:
------
    from horizon_config import (
        HORIZONS,
        SUPPORTED_HORIZONS,
        HorizonConfig,
        validate_horizons,
        get_scaled_horizons,
        auto_scale_purge_embargo,
        compute_embargo_bars,  # NEW: timeframe-aware embargo calculation
        get_default_barrier_params_for_horizon,
        EMBARGO_TIME_MINUTES,  # NEW: embargo specified in calendar time
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# =============================================================================
# SUPPORTED AND ACTIVE HORIZONS
# =============================================================================
# All supported horizons for triple-barrier labeling.
# These represent the number of bars to look ahead for label calculation.
SUPPORTED_HORIZONS = [1, 5, 10, 15, 20, 30, 60, 120]

# Active horizons for model training (subset of SUPPORTED_HORIZONS).
# H1 is excluded by default because transaction costs exceed expected profit.
# Modify this list to enable/disable specific horizons.
HORIZONS = [5, 10, 15, 20]  # Default active horizons (configurable)

# Legacy aliases for backward compatibility
LOOKBACK_HORIZONS = [1, 5, 20]  # For labeling (includes H1)
ACTIVE_HORIZONS = [5, 10, 15, 20]  # For training (excludes H1)
LABEL_HORIZONS = ACTIVE_HORIZONS  # Alias for trainer validation


# =============================================================================
# TIMEFRAME CONFIGURATION
# =============================================================================
# Timeframe duration in minutes for horizon scaling.
# When resampling data to different timeframes, horizons must scale accordingly.
# Example: 5 bars at 5min (25 minutes) = ~2 bars at 15min (~30 minutes)
# The formula is: new_horizon = old_horizon * (source_minutes / target_minutes)
HORIZON_TIMEFRAME_MINUTES = {
    "1min": 1,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "20min": 20,
    "30min": 30,
    "45min": 45,
    "60min": 60,
    "1h": 60,  # Alias for 60min
}

# Legacy alias for backward compatibility
HORIZON_TIMEFRAME_SCALING = HORIZON_TIMEFRAME_MINUTES


# =============================================================================
# PURGE/EMBARGO CONFIGURATION
# =============================================================================
# Purge/embargo auto-scaling configuration.
# These multipliers determine purge and embargo bars based on max horizon.
PURGE_MULTIPLIER = 3.0  # purge_bars = max_horizon * PURGE_MULTIPLIER
EMBARGO_MULTIPLIER = 72.0  # embargo_bars = max_horizon * EMBARGO_MULTIPLIER (~5 days for H20)

# TIMEFRAME-AWARE EMBARGO CONFIGURATION
# Embargo is specified in TIME (minutes), not bars, to ensure consistent
# decorrelation periods regardless of bar timeframe.
#
# PROBLEM THIS SOLVES:
# Previously, MIN_EMBARGO_BARS=1440 assumed 5-minute bars (1440*5min = 5 days).
# But at 15-minute bars, 1440 bars = 15 days - far too long!
# At 1-minute bars, 1440 bars = 1 day - not enough!
#
# SOLUTION:
# Specify embargo in calendar time (minutes), then convert to bars based on
# the actual target_timeframe. This ensures ~5 days of buffer regardless of
# the bar resolution used.
EMBARGO_TIME_MINUTES = 7200  # 5 days * 24 hours * 60 minutes = 7200 minutes
DEFAULT_TIMEFRAME_MINUTES = 5  # Default assumption when timeframe not specified

# Legacy constant for backward compatibility (assumes 5-min bars)
MIN_EMBARGO_BARS = 1440  # Deprecated: use compute_embargo_bars() instead


# =============================================================================
# HORIZON VALIDATION
# =============================================================================
def validate_horizons(horizons: list[int], data_length: int = None) -> None:
    """
    Validate horizons against supported values and optionally against data length.

    Parameters:
    -----------
    horizons : List[int]
        List of horizon values to validate
    data_length : int, optional
        Length of the data. If provided, validates that horizons are < 10% of data

    Raises:
    -------
    ValueError : If any horizon is invalid

    Examples:
    ---------
    >>> validate_horizons([5, 20])  # OK
    >>> validate_horizons([5, 200])  # Raises ValueError (not in SUPPORTED_HORIZONS)
    >>> validate_horizons([5, 20], data_length=100)  # Raises ValueError (20 >= 10% of 100)
    """
    if not horizons:
        raise ValueError("Horizons list cannot be empty")

    for h in horizons:
        if not isinstance(h, int):
            raise ValueError(f"Horizon must be an integer, got {type(h).__name__}: {h}")
        if h <= 0:
            raise ValueError(f"Horizon must be positive, got {h}")
        if h not in SUPPORTED_HORIZONS:
            raise ValueError(
                f"Horizon {h} not in SUPPORTED_HORIZONS: {SUPPORTED_HORIZONS}. "
                f"Add it to SUPPORTED_HORIZONS in horizon_config.py if this is intentional."
            )

    if data_length is not None:
        if data_length <= 0:
            raise ValueError(f"data_length must be positive, got {data_length}")
        max_allowed = data_length // 10
        for h in horizons:
            if h >= max_allowed:
                raise ValueError(
                    f"Horizon {h} too large for data length {data_length}. "
                    f"Horizon should be < 10% of data length ({max_allowed}). "
                    f"Reduce horizon or increase data size."
                )


# =============================================================================
# TIMEFRAME-AWARE HORIZON SCALING
# =============================================================================
def get_scaled_horizons(horizons: list[int], source_tf: str, target_tf: str) -> list[int]:
    """
    Scale horizons when changing timeframe.

    When resampling data from one timeframe to another, horizons must be scaled
    to maintain the same real-time window. For example, H20 at 5min (100 minutes)
    should become H7 at 15min (~105 minutes, closest integer).

    The formula is: new_horizon = old_horizon * (source_minutes / target_minutes)
    - Going from small to large timeframe: horizons decrease
    - Going from large to small timeframe: horizons increase

    Parameters:
    -----------
    horizons : List[int]
        Original horizon values
    source_tf : str
        Source timeframe (e.g., '5min', '1min')
    target_tf : str
        Target timeframe (e.g., '15min', '60min')

    Returns:
    --------
    List[int] : Scaled horizon values (minimum 1)

    Raises:
    -------
    ValueError : If timeframe is not recognized

    Examples:
    ---------
    >>> get_scaled_horizons([5, 20], '5min', '15min')
    [2, 7]  # 5 bars @ 5min (25min) = ~2 bars @ 15min

    >>> get_scaled_horizons([5, 20], '5min', '1min')
    [25, 100]  # 5 bars @ 5min (25min) = 25 bars @ 1min
    """
    if source_tf not in HORIZON_TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unknown source timeframe: '{source_tf}'. "
            f"Supported: {list(HORIZON_TIMEFRAME_MINUTES.keys())}"
        )
    if target_tf not in HORIZON_TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unknown target timeframe: '{target_tf}'. "
            f"Supported: {list(HORIZON_TIMEFRAME_MINUTES.keys())}"
        )

    source_minutes = HORIZON_TIMEFRAME_MINUTES[source_tf]
    target_minutes = HORIZON_TIMEFRAME_MINUTES[target_tf]

    # Scale factor: how many target bars equal one source bar in real time
    # new_horizon = old_horizon * (source_minutes / target_minutes)
    # Example: 5min -> 15min: scale = 5/15 = 0.333
    #   5 bars @ 5min = 5 * 0.333 = 1.67 -> 2 bars @ 15min
    # Example: 5min -> 1min: scale = 5/1 = 5
    #   5 bars @ 5min = 5 * 5 = 25 bars @ 1min
    scale = source_minutes / target_minutes

    scaled = []
    for h in horizons:
        scaled_h = max(1, int(round(h * scale)))
        scaled.append(scaled_h)

    return scaled


# =============================================================================
# TIMEFRAME-AWARE EMBARGO CALCULATION
# =============================================================================
def compute_embargo_bars(
    timeframe: str,
    embargo_time_minutes: int = None,
) -> int:
    """
    Compute embargo bars for a given timeframe to achieve a consistent time buffer.

    This function converts embargo from TIME (minutes) to BARS based on the
    actual timeframe being used. This ensures consistent decorrelation periods
    regardless of bar resolution.

    Parameters:
    -----------
    timeframe : str
        The bar timeframe (e.g., '5min', '15min', '1h')
    embargo_time_minutes : int, optional
        Desired embargo duration in minutes (default: EMBARGO_TIME_MINUTES = 7200 = 5 days)

    Returns:
    --------
    int : Number of bars for embargo

    Examples:
    ---------
    >>> compute_embargo_bars('5min')   # 5 days at 5-min bars
    1440

    >>> compute_embargo_bars('15min')  # 5 days at 15-min bars
    480

    >>> compute_embargo_bars('1min')   # 5 days at 1-min bars
    7200

    >>> compute_embargo_bars('1h')     # 5 days at 1-hour bars
    120
    """
    if embargo_time_minutes is None:
        embargo_time_minutes = EMBARGO_TIME_MINUTES

    if embargo_time_minutes <= 0:
        raise ValueError(f"embargo_time_minutes must be positive, got {embargo_time_minutes}")

    # Get timeframe in minutes
    if timeframe in HORIZON_TIMEFRAME_MINUTES:
        tf_minutes = HORIZON_TIMEFRAME_MINUTES[timeframe]
    else:
        # Try to parse common formats
        tf_lower = timeframe.lower()
        if tf_lower.endswith("min"):
            try:
                tf_minutes = int(tf_lower[:-3])
            except ValueError:
                raise ValueError(f"Cannot parse timeframe '{timeframe}'")
        elif tf_lower.endswith("h"):
            try:
                tf_minutes = int(tf_lower[:-1]) * 60
            except ValueError:
                raise ValueError(f"Cannot parse timeframe '{timeframe}'")
        else:
            raise ValueError(
                f"Unknown timeframe: '{timeframe}'. "
                f"Supported: {list(HORIZON_TIMEFRAME_MINUTES.keys())} or formats like '15min', '1h'"
            )

    # Convert time to bars: embargo_bars = embargo_time_minutes / timeframe_minutes
    embargo_bars = int(embargo_time_minutes / tf_minutes)

    # Ensure at least 1 bar
    return max(1, embargo_bars)


# =============================================================================
# AUTO-SCALE PURGE AND EMBARGO
# =============================================================================
def auto_scale_purge_embargo(
    horizons: list[int],
    purge_multiplier: float = None,
    embargo_multiplier: float = None,
    timeframe: str | None = None,
    embargo_time_minutes: int | None = None,
) -> tuple[int, int]:
    """
    Auto-calculate purge and embargo bars based on max horizon.

    Purge bars prevent label leakage by removing samples near split boundaries.
    Embargo bars provide a buffer for feature decorrelation between splits.

    IMPORTANT: If timeframe is provided, embargo is calculated based on actual
    calendar time (EMBARGO_TIME_MINUTES = 5 days) to ensure consistent
    decorrelation regardless of bar resolution. If timeframe is not provided,
    falls back to legacy behavior assuming 5-minute bars.

    Parameters:
    -----------
    horizons : List[int]
        List of active horizons
    purge_multiplier : float, optional
        Multiplier for purge bars (default: PURGE_MULTIPLIER)
    embargo_multiplier : float, optional
        Multiplier for embargo bars (default: EMBARGO_MULTIPLIER)
        NOTE: Only used when timeframe is not provided (legacy mode)
    timeframe : str, optional
        Target timeframe (e.g., '5min', '15min'). When provided, embargo is
        calculated as EMBARGO_TIME_MINUTES / timeframe_minutes to ensure
        consistent ~5 day buffer regardless of bar resolution.
    embargo_time_minutes : int, optional
        Override for embargo duration in minutes (default: 7200 = 5 days)

    Returns:
    --------
    Tuple[int, int] : (purge_bars, embargo_bars)

    Notes:
    ------
    - purge_bars = max_horizon * purge_multiplier (ensures all labels are valid)
    - embargo_bars (with timeframe):
        embargo_time_minutes / timeframe_minutes (consistent calendar time)
    - embargo_bars (without timeframe, legacy):
        max(max_horizon * embargo_multiplier, MIN_EMBARGO_BARS)

    Examples:
    ---------
    >>> # Timeframe-aware (recommended)
    >>> auto_scale_purge_embargo([5, 20], timeframe='5min')
    (60, 1440)  # purge=60, embargo=1440 (5 days at 5min)

    >>> auto_scale_purge_embargo([5, 20], timeframe='15min')
    (60, 480)   # purge=60, embargo=480 (5 days at 15min)

    >>> # Legacy mode (no timeframe, assumes 5min bars)
    >>> auto_scale_purge_embargo([5, 20])
    (60, 1440)  # purge=60, embargo=1440 (assumes 5min bars)
    """
    if not horizons:
        raise ValueError("Horizons list cannot be empty for purge/embargo calculation")

    if purge_multiplier is None:
        purge_multiplier = PURGE_MULTIPLIER

    if purge_multiplier <= 0:
        raise ValueError(f"purge_multiplier must be positive, got {purge_multiplier}")

    max_horizon = max(horizons)

    # Purge: Must cover max_bars for barrier calculation
    # max_bars is typically 2-3x horizon, so purge = max_horizon * 3
    purge_bars = int(max_horizon * purge_multiplier)

    # Embargo: Calculate based on timeframe if provided
    if timeframe is not None:
        # TIMEFRAME-AWARE MODE (recommended)
        # Embargo is calculated from calendar time, not bar count
        embargo_bars = compute_embargo_bars(
            timeframe=timeframe,
            embargo_time_minutes=embargo_time_minutes,
        )
    else:
        # LEGACY MODE (backward compatible)
        # Uses multiplier with MIN_EMBARGO_BARS floor (assumes 5min bars)
        if embargo_multiplier is None:
            embargo_multiplier = EMBARGO_MULTIPLIER
        if embargo_multiplier <= 0:
            raise ValueError(f"embargo_multiplier must be positive, got {embargo_multiplier}")

        embargo_bars = max(int(max_horizon * embargo_multiplier), MIN_EMBARGO_BARS)

    return purge_bars, embargo_bars


# =============================================================================
# DEFAULT BARRIER PARAMETER GENERATION
# =============================================================================
def get_default_barrier_params_for_horizon(horizon: int) -> dict:
    """
    Generate default barrier parameters for a non-standard horizon.

    For horizons not explicitly defined in BARRIER_PARAMS_DEFAULT, this function
    calculates reasonable defaults based on horizon scaling patterns.

    Parameters:
    -----------
    horizon : int
        Horizon value

    Returns:
    --------
    dict : Barrier parameters with 'k_up', 'k_down', 'max_bars', 'description'

    Notes:
    ------
    - k values scale logarithmically with horizon
    - max_bars = horizon * 2.5 (allows time for barrier hits)
    - Uses symmetric barriers by default

    Examples:
    ---------
    >>> get_default_barrier_params_for_horizon(30)
    {'k_up': 1.8, 'k_down': 1.8, 'max_bars': 75, 'description': 'H30: Auto-generated defaults'}
    """
    if horizon <= 0:
        raise ValueError(f"Horizon must be positive, got {horizon}")

    # k values scale logarithmically: more bars = wider barriers
    # Base: H5 uses k~1.2, H20 uses k~2.5
    # Formula: k = 0.8 + 0.4 * log2(horizon)
    k_base = 0.8 + 0.4 * math.log2(max(1, horizon))
    k_base = max(0.5, min(k_base, 4.0))  # Clamp to reasonable range

    # max_bars: allow 2.5x horizon for barrier resolution
    max_bars = int(horizon * 2.5)
    max_bars = max(5, min(max_bars, 300))  # Clamp to reasonable range

    return {
        "k_up": round(k_base, 2),
        "k_down": round(k_base, 2),
        "max_bars": max_bars,
        "description": f"H{horizon}: Auto-generated defaults (symmetric)",
    }


# =============================================================================
# HORIZON CONFIG DATACLASS
# =============================================================================
@dataclass
class HorizonConfig:
    """
    Configuration for dynamic horizon labeling.

    This class encapsulates all horizon-related settings, including:
    - Active horizons for labeling
    - Timeframe-aware horizon scaling
    - Auto-scaling of purge and embargo bars

    Examples:
    ---------
    >>> # Default configuration
    >>> config = HorizonConfig()
    >>> config.horizons
    [5, 20]

    >>> # Custom horizons with auto-scaling
    >>> config = HorizonConfig(horizons=[5, 20, 60])
    >>> purge, embargo = config.get_purge_embargo()
    >>> purge  # 180 (60 * 3)

    >>> # Horizons for different timeframe
    >>> config = HorizonConfig(horizons=[5, 20], source_timeframe='5min')
    >>> config.get_scaled_horizons('15min')
    [2, 7]
    """

    # Active horizons for model training
    # Must be subset of SUPPORTED_HORIZONS from this module
    horizons: list[int] = field(default_factory=lambda: list(ACTIVE_HORIZONS))

    # Source timeframe (for horizon scaling when resampling)
    source_timeframe: str = "5min"

    # Auto-scale purge/embargo based on max horizon
    auto_scale_purge_embargo: bool = True

    # Manual overrides (used when auto_scale_purge_embargo=False)
    manual_purge_bars: int | None = None
    manual_embargo_bars: int | None = None

    # Purge/embargo multipliers (used when auto_scale=True)
    purge_multiplier: float = 3.0
    embargo_multiplier: float = 15.0

    def get_purge_embargo(self, target_timeframe: str | None = None) -> tuple[int, int]:
        """
        Get purge and embargo bars based on configuration.

        When auto_scale_purge_embargo=True, calculates values from max horizon.
        Otherwise, uses manual_purge_bars and manual_embargo_bars.

        IMPORTANT: If target_timeframe is provided (or defaults to source_timeframe),
        embargo is calculated based on calendar time (5 days = 7200 minutes) to ensure
        consistent decorrelation regardless of bar resolution.

        Parameters:
        -----------
        target_timeframe : str, optional
            Timeframe for embargo calculation. Defaults to source_timeframe.
            When provided, embargo_bars = 7200 / timeframe_minutes.

        Returns:
        --------
        Tuple[int, int] : (purge_bars, embargo_bars)

        Examples:
        ---------
        >>> config = HorizonConfig(horizons=[5, 20], source_timeframe='5min')
        >>> config.get_purge_embargo()  # uses source_timeframe='5min'
        (60, 1440)  # 5 days at 5-min bars

        >>> config.get_purge_embargo(target_timeframe='15min')
        (60, 480)   # 5 days at 15-min bars
        """
        if target_timeframe is None:
            target_timeframe = self.source_timeframe

        if self.auto_scale_purge_embargo:
            return auto_scale_purge_embargo(
                self.horizons,
                self.purge_multiplier,
                self.embargo_multiplier,
                timeframe=target_timeframe,  # Use timeframe-aware calculation
            )
        else:
            # Use manual values, with defaults if not specified
            purge = self.manual_purge_bars if self.manual_purge_bars is not None else 60
            embargo = self.manual_embargo_bars if self.manual_embargo_bars is not None else 288
            return purge, embargo

    def get_scaled_horizons(self, target_timeframe: str) -> list[int]:
        """
        Get horizons scaled to a target timeframe.

        Useful when resampling data to a different resolution.
        Horizons are scaled proportionally to maintain the same real-time window.

        Parameters:
        -----------
        target_timeframe : str
            Target timeframe (e.g., '15min', '60min')

        Returns:
        --------
        List[int] : Scaled horizons

        Examples:
        ---------
        >>> config = HorizonConfig(horizons=[5, 20], source_timeframe='5min')
        >>> config.get_scaled_horizons('15min')
        [2, 7]  # 5 bars @ 5min = ~2 bars @ 15min
        """
        return get_scaled_horizons(self.horizons, self.source_timeframe, target_timeframe)

    def validate(self) -> list[str]:
        """
        Validate horizon configuration.

        Returns:
        --------
        List[str] : List of validation error messages (empty if valid)
        """
        issues = []

        if not self.horizons:
            issues.append("At least one horizon must be specified")

        for h in self.horizons:
            if not isinstance(h, int):
                issues.append(f"Horizon must be an integer, got {type(h).__name__}")
            elif h <= 0:
                issues.append(f"Horizon must be positive, got {h}")

        if self.purge_multiplier <= 0:
            issues.append(f"purge_multiplier must be positive, got {self.purge_multiplier}")

        if self.embargo_multiplier <= 0:
            issues.append(f"embargo_multiplier must be positive, got {self.embargo_multiplier}")

        if not self.auto_scale_purge_embargo:
            if self.manual_purge_bars is not None and self.manual_purge_bars < 0:
                issues.append(
                    f"manual_purge_bars must be non-negative, got {self.manual_purge_bars}"
                )
            if self.manual_embargo_bars is not None and self.manual_embargo_bars < 0:
                issues.append(
                    f"manual_embargo_bars must be non-negative, got {self.manual_embargo_bars}"
                )

        return issues
