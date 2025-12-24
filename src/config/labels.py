"""
Label column definitions and templates.

Centralizes all label-related column naming conventions and templates
used across the pipeline stages for consistency.
"""
from typing import Dict, List

# =============================================================================
# LABEL COLUMN TEMPLATES
# =============================================================================
# These templates use Python .format() syntax with {h} for horizon values.

# Required label columns that must exist for each horizon
REQUIRED_LABEL_TEMPLATES: List[str] = [
    "label_h{h}",           # Primary label: -1 (short), 0 (neutral), 1 (long)
    "sample_weight_h{h}",   # Quality-based sample weights
]

# Optional label columns that may exist for each horizon
OPTIONAL_LABEL_TEMPLATES: List[str] = [
    "quality_h{h}",          # Label quality score (0-1)
    "bars_to_hit_h{h}",      # Bars until barrier hit
    "mae_h{h}",              # Maximum adverse excursion
    "mfe_h{h}",              # Maximum favorable excursion
    "touch_type_h{h}",       # Which barrier was hit first
    "pain_to_gain_h{h}",     # MAE/MFE ratio
    "time_weighted_dd_h{h}", # Time-weighted drawdown
    "fwd_return_h{h}",       # Forward return (simple)
    "fwd_return_log_h{h}",   # Forward return (log)
    "time_to_hit_h{h}",      # Time to first barrier hit
]

# All label templates combined
ALL_LABEL_TEMPLATES: List[str] = REQUIRED_LABEL_TEMPLATES + OPTIONAL_LABEL_TEMPLATES


# =============================================================================
# LABEL COLUMN RESOLUTION
# =============================================================================

def get_required_label_columns(horizon: int) -> List[str]:
    """
    Get required label column names for a horizon.

    Args:
        horizon: Label horizon value

    Returns:
        List of required column names
    """
    return [template.format(h=horizon) for template in REQUIRED_LABEL_TEMPLATES]


def get_optional_label_columns(horizon: int) -> List[str]:
    """
    Get optional label column names for a horizon.

    Args:
        horizon: Label horizon value

    Returns:
        List of optional column names
    """
    return [template.format(h=horizon) for template in OPTIONAL_LABEL_TEMPLATES]


def get_all_label_columns(horizon: int) -> List[str]:
    """
    Get all possible label column names for a horizon.

    Args:
        horizon: Label horizon value

    Returns:
        List of all column names
    """
    return [template.format(h=horizon) for template in ALL_LABEL_TEMPLATES]


def is_label_column(column_name: str) -> bool:
    """
    Check if a column name matches any label pattern.

    Args:
        column_name: Column name to check

    Returns:
        True if column is a label column
    """
    prefixes = (
        "label_", "sample_weight_", "quality_", "bars_to_hit_",
        "mae_", "mfe_", "touch_type_", "pain_to_gain_",
        "time_weighted_dd_", "fwd_return_", "time_to_hit_",
    )
    return any(column_name.startswith(prefix) for prefix in prefixes)


# =============================================================================
# LABEL METADATA
# =============================================================================

LABEL_COLUMN_METADATA: Dict[str, Dict] = {
    "label_h{h}": {
        "description": "Primary triple-barrier label",
        "dtype": "int8",
        "values": [-1, 0, 1],
        "meanings": {-1: "short", 0: "neutral", 1: "long"},
    },
    "sample_weight_h{h}": {
        "description": "Quality-based sample weight",
        "dtype": "float32",
        "range": [0.5, 1.5],
    },
    "quality_h{h}": {
        "description": "Label quality score",
        "dtype": "float32",
        "range": [0.0, 1.0],
    },
    "bars_to_hit_h{h}": {
        "description": "Bars until first barrier hit",
        "dtype": "int32",
        "range": [1, None],
    },
    "mae_h{h}": {
        "description": "Maximum adverse excursion (worst drawdown)",
        "dtype": "float32",
    },
    "mfe_h{h}": {
        "description": "Maximum favorable excursion (best profit)",
        "dtype": "float32",
    },
    "touch_type_h{h}": {
        "description": "First barrier touched",
        "dtype": "int8",
        "values": [-1, 0, 1],
        "meanings": {-1: "lower", 0: "time", 1: "upper"},
    },
    "pain_to_gain_h{h}": {
        "description": "MAE/MFE ratio (lower is better)",
        "dtype": "float32",
    },
    "time_weighted_dd_h{h}": {
        "description": "Time-weighted drawdown during label period",
        "dtype": "float32",
    },
    "fwd_return_h{h}": {
        "description": "Forward simple return",
        "dtype": "float32",
    },
    "fwd_return_log_h{h}": {
        "description": "Forward log return",
        "dtype": "float32",
    },
    "time_to_hit_h{h}": {
        "description": "Time to first barrier (minutes)",
        "dtype": "int32",
    },
}


def get_label_metadata(column_template: str, horizon: int) -> Dict:
    """
    Get metadata for a specific label column.

    Args:
        column_template: Template with {h} placeholder
        horizon: Horizon value

    Returns:
        Metadata dict for the column
    """
    metadata = LABEL_COLUMN_METADATA.get(column_template, {}).copy()
    metadata["column_name"] = column_template.format(h=horizon)
    metadata["horizon"] = horizon
    return metadata
