"""Pipeline Utilities.

Provides stage results and constants. Feature set utilities available via direct import.
"""

from src.data.pipeline.utils.constants import LABEL_PREFIXES, METADATA_COLUMNS
from src.data.pipeline.utils_core import (
    StageResult,
    StageStatus,
    create_failed_result,
    create_stage_result,
)

__all__ = [
    "StageResult",
    "StageStatus",
    "create_stage_result",
    "create_failed_result",
    "METADATA_COLUMNS",
    "LABEL_PREFIXES",
]
