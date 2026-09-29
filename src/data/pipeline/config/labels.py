"""
Label column definitions and templates.

Centralizes all label-related column naming conventions and templates
used across the pipeline stages for consistency.
"""

# =============================================================================
# LABEL COLUMN TEMPLATES
# =============================================================================
# These templates use Python .format() syntax with {h} for horizon values.

# Required label columns that must exist for each horizon
REQUIRED_LABEL_TEMPLATES: list[str] = [
    "label_h{h}",  # Primary label: -1 (short), 0 (neutral), 1 (long)
    "sample_weight_h{h}",  # Quality-based sample weights
]

# Optional label columns that may exist for each horizon
OPTIONAL_LABEL_TEMPLATES: list[str] = [
    "quality_h{h}",  # Label quality score (0-1)
    "bars_to_hit_h{h}",  # Bars until barrier hit
    "label_end_time_h{h}",  # Datetime when label outcome is known (for purging)
    "mae_h{h}",  # Maximum adverse excursion
    "mfe_h{h}",  # Maximum favorable excursion
    "touch_type_h{h}",  # Which barrier was hit first
    "pain_to_gain_h{h}",  # MAE/MFE ratio
    "time_weighted_dd_h{h}",  # Time-weighted drawdown
    "fwd_return_h{h}",  # Forward return (simple)
    "fwd_return_log_h{h}",  # Forward return (log)
    "time_to_hit_h{h}",  # Time to first barrier hit
]
