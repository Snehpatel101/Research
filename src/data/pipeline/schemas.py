"""Pipeline stage schemas for inter-stage validation.

Defines input/output schemas for pipeline stages to enable
validation between stages. This ensures data flows correctly
through the pipeline and catches issues early.

Part of Phase 7B: Inter-Stage Schema Validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.core.exceptions import StageValidationError

logger = logging.getLogger(__name__)


@dataclass
class StageSchema:
    """Schema definition for a pipeline stage.

    Attributes:
        required_columns: Columns that must be present in the DataFrame
        min_rows: Minimum number of rows required (default: 0)
        max_nan_ratio: Maximum allowed NaN ratio (default: 1.0, no limit)
        description: Optional description of the stage requirements
    """

    required_columns: list[str] = field(default_factory=list)
    min_rows: int = 0
    max_nan_ratio: float = 1.0
    description: str = ""


# Stage schemas for common pipeline stages
STAGE_SCHEMAS: dict[str, StageSchema] = {
    "data_generation": StageSchema(
        required_columns=["datetime", "open", "high", "low", "close", "volume"],
        min_rows=100,
        description="Raw OHLCV data from data generation",
    ),
    "data_cleaning": StageSchema(
        required_columns=["datetime", "open", "high", "low", "close", "volume"],
        min_rows=1000,
        description="Cleaned OHLCV data with outliers removed",
    ),
    "feature_engineering": StageSchema(
        required_columns=["datetime"],  # Dynamic based on features
        min_rows=500,
        max_nan_ratio=0.01,
        description="Feature-engineered data with NaN columns cleaned",
    ),
    "initial_labeling": StageSchema(
        required_columns=["datetime"],
        min_rows=500,
        description="Data with initial labels applied",
    ),
    "final_labels": StageSchema(
        required_columns=["datetime"],
        min_rows=100,
        description="Data with final optimized labels",
    ),
    "create_splits": StageSchema(
        required_columns=["datetime"],
        min_rows=50,
        description="Data split into train/val/test",
    ),
    "feature_scaling": StageSchema(
        required_columns=["datetime"],
        min_rows=50,
        max_nan_ratio=0.0,  # No NaNs allowed after scaling
        description="Scaled features ready for training",
    ),
    "build_datasets": StageSchema(
        required_columns=["datetime"],
        min_rows=50,
        max_nan_ratio=0.0,
        description="Final datasets built for training",
    ),
}


def validate_stage_output(
    df: pd.DataFrame,
    stage_name: str,
    schema: StageSchema | None = None,
    raise_on_failure: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate dataframe against stage schema.

    Args:
        df: DataFrame to validate
        stage_name: Name of the pipeline stage
        schema: Optional explicit schema (defaults to STAGE_SCHEMAS lookup)
        raise_on_failure: If True, raise StageValidationError on failure

    Returns:
        Tuple of (is_valid, list of issues)

    Raises:
        StageValidationError: If raise_on_failure=True and validation fails
    """
    if schema is None:
        schema = STAGE_SCHEMAS.get(stage_name)

    if schema is None:
        # No schema defined for this stage, skip validation
        logger.debug(f"No schema defined for stage '{stage_name}', skipping validation")
        return True, []

    issues: list[str] = []

    # Check required columns
    if schema.required_columns:
        missing = set(schema.required_columns) - set(df.columns)
        if missing:
            issues.append(f"Missing required columns: {sorted(missing)}")

    # Check minimum rows
    if len(df) < schema.min_rows:
        issues.append(f"Insufficient rows: {len(df)} < {schema.min_rows} required")

    # Check NaN ratio (only for non-empty DataFrames)
    if schema.max_nan_ratio < 1.0 and len(df) > 0 and len(df.columns) > 0:
        total_cells = len(df) * len(df.columns)
        nan_count = int(df.isna().sum().sum())
        nan_ratio = nan_count / total_cells if total_cells > 0 else 0.0

        if nan_ratio > schema.max_nan_ratio:
            issues.append(
                f"NaN ratio too high: {nan_ratio:.2%} > {schema.max_nan_ratio:.2%} allowed "
                f"({nan_count} NaN cells out of {total_cells})"
            )

    # Log result
    is_valid = len(issues) == 0
    if is_valid:
        logger.debug(
            f"Stage '{stage_name}' validation passed: " f"{len(df)} rows, {len(df.columns)} columns"
        )
    else:
        logger.warning(f"Stage '{stage_name}' validation failed: {issues}")

    # Raise if requested
    if raise_on_failure and not is_valid:
        raise StageValidationError(stage_name, issues)

    return is_valid, issues


def get_stage_schema(stage_name: str) -> StageSchema | None:
    """Get the schema for a stage by name."""
    return STAGE_SCHEMAS.get(stage_name)


def register_stage_schema(stage_name: str, schema: StageSchema) -> None:
    """Register a custom schema for a stage."""
    STAGE_SCHEMAS[stage_name] = schema
    logger.debug(f"Registered schema for stage '{stage_name}'")


__all__ = [
    "StageSchema",
    "StageValidationError",
    "STAGE_SCHEMAS",
    "validate_stage_output",
    "get_stage_schema",
    "register_stage_schema",
]
