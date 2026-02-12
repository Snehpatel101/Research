"""
Feature set constants.

These constants are used by feature_selection.py and feature_sets.py.
Extracted to avoid circular imports.

DATA-003: Metadata columns are versioned and documented for schema evolution tracking.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# METADATA COLUMNS SCHEMA (DATA-003)
# =============================================================================

# Version for tracking schema changes
METADATA_COLUMNS_VERSION = "1.0"

# Metadata columns with documentation
# These columns are excluded from feature extraction as they contain:
# - Identifiers (datetime, symbol, timestamp)
# - Raw OHLCV data (open, high, low, close, volume)
# - Pipeline-generated metadata (session_id, missing_bar, etc.)
METADATA_COLUMNS_DEFINITION = {
    # Temporal identifiers
    "datetime": "Primary datetime index or column",
    "timestamp": "Unix timestamp or alternative datetime representation",
    "date": "Date component (YYYY-MM-DD)",
    "time": "Time component (HH:MM:SS)",
    # Symbol identifiers
    "symbol": "Trading instrument symbol (e.g., MES, MGC)",
    # Raw OHLCV data (not features, source data)
    "open": "Bar open price",
    "high": "Bar high price",
    "low": "Bar low price",
    "close": "Bar close price",
    "volume": "Bar volume",
    # Pipeline metadata
    "timeframe": "Bar timeframe (e.g., 5min, 15min)",
    "session_id": "Trading session identifier",
    "missing_bar": "Flag indicating if bar was interpolated",
    "roll_event": "Contract roll event marker",
    "roll_window": "Contract roll window period",
    "filled": "Flag indicating if bar was forward-filled",
}

# The set used for column filtering (backward compatible)
METADATA_COLUMNS = set(METADATA_COLUMNS_DEFINITION.keys())

# Label column prefixes (these are target variables, not features)
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


# =============================================================================
# METADATA VALIDATION FUNCTIONS (DATA-003)
# =============================================================================


def validate_metadata_columns(df_columns: list[str] | set[str]) -> dict[str, Any]:
    """
    Validate DataFrame columns against expected metadata schema.

    Checks for:
    - Unexpected columns in metadata position (new columns not in schema)
    - Missing expected metadata columns
    - Schema version compatibility

    Args:
        df_columns: List or set of column names from a DataFrame

    Returns:
        Dictionary with validation results:
        - is_valid: bool - True if no unexpected metadata columns
        - schema_version: str - Current schema version
        - unexpected_columns: list - Columns that look like metadata but aren't in schema
        - missing_metadata: list - Expected metadata columns not present

    Note:
        This function warns but does not block on validation issues.
        Unexpected columns may indicate schema evolution needs.
    """
    df_columns_set = set(df_columns)

    # Find columns that are in the DataFrame but not in our metadata schema
    # Only flag columns that look like they could be metadata (not feature columns)
    # Feature columns typically have prefixes like rsi_, atr_, etc.
    potential_metadata_prefixes = ("session", "roll", "missing", "filled", "bar_")

    unexpected = []
    for col in df_columns_set:
        col_lower = col.lower()
        # Check if column looks like metadata but isn't in our schema
        if col not in METADATA_COLUMNS and any(
            col_lower.startswith(prefix) for prefix in potential_metadata_prefixes
        ):
            unexpected.append(col)

    # Find expected metadata that's missing (for informational purposes)
    missing = [col for col in METADATA_COLUMNS if col not in df_columns_set]

    # Log warnings for unexpected metadata-like columns
    if unexpected:
        logger.warning(
            f"Metadata validation: Found {len(unexpected)} unexpected metadata-like columns: "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}. "
            f"Consider updating METADATA_COLUMNS_DEFINITION (version {METADATA_COLUMNS_VERSION})."
        )

    return {
        "is_valid": len(unexpected) == 0,
        "schema_version": METADATA_COLUMNS_VERSION,
        "unexpected_columns": unexpected,
        "missing_metadata": missing,
        "total_columns_checked": len(df_columns_set),
    }


def get_metadata_columns_info() -> dict[str, Any]:
    """
    Get metadata columns schema information.

    Returns:
        Dictionary with schema version, column count, and definitions.
    """
    return {
        "version": METADATA_COLUMNS_VERSION,
        "column_count": len(METADATA_COLUMNS),
        "columns": METADATA_COLUMNS_DEFINITION,
        "label_prefixes": LABEL_PREFIXES,
    }
