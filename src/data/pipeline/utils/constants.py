"""
Feature set constants.

These constants are used by feature_selection.py and feature_sets.py.
Extracted to avoid circular imports.

DATA-003: Metadata columns are versioned and documented for schema evolution tracking.
"""

import logging

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
