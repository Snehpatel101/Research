"""
Pipeline constants - canonical column exclusion definitions.
"""

# OHLCV and metadata columns (not features)
EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {
        "datetime",
        "timestamp",
        "date",
        "time",
        "symbol",
        "timeframe",
        "session_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "missing_bar",
        "roll_event",
        "roll_window",
        "filled",
    }
)

# Prefixes for label/target columns (not features)
EXCLUDED_PREFIXES: tuple[str, ...] = (
    "label_",
    "target_",
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


def is_feature_column(col: str) -> bool:
    """Check if column name represents a feature (not excluded)."""
    if col in EXCLUDED_COLUMNS:
        return False
    return all(not col.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


__all__ = ["EXCLUDED_COLUMNS", "EXCLUDED_PREFIXES", "is_feature_column"]
