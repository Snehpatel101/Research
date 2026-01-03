"""
Feature set constants.

These constants are used by feature_selection.py and feature_sets.py.
Extracted to avoid circular imports.
"""

METADATA_COLUMNS = {
    "datetime",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "timestamp",
    "date",
    "time",
    "timeframe",
    "session_id",
    "missing_bar",
    "roll_event",
    "roll_window",
    "filled",
}

LABEL_PREFIXES = (
    "label_",
    "bars_to_hit_",
    "mae_",
    "mfe_",
    "quality_",
    "sample_weight_",
    "touch_type_",
    "pain_to_gain_",
    "time_weighted_dd_",
    "fwd_return_",
    "fwd_return_log_",
    "time_to_hit_",
)
