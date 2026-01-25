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
from src.data.pipeline.stage_registry import StageName

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
# Uses StageName enum values for type safety (Phase 12.5G)
STAGE_SCHEMAS: dict[str, StageSchema] = {
    StageName.DATA_GENERATION.value: StageSchema(
        required_columns=["datetime", "open", "high", "low", "close", "volume"],
        min_rows=100,
        description="Raw OHLCV data from data generation",
    ),
    StageName.DATA_CLEANING.value: StageSchema(
        required_columns=["datetime", "open", "high", "low", "close", "volume"],
        min_rows=1000,
        description="Cleaned OHLCV data with outliers removed",
    ),
    StageName.FEATURE_ENGINEERING.value: StageSchema(
        required_columns=["datetime"],  # Dynamic based on features
        min_rows=500,
        max_nan_ratio=0.01,
        description="Feature-engineered data with NaN columns cleaned",
    ),
    StageName.INITIAL_LABELING.value: StageSchema(
        required_columns=["datetime"],
        min_rows=500,
        description="Data with initial labels applied",
    ),
    StageName.GA_OPTIMIZE.value: StageSchema(
        required_columns=["datetime"],
        min_rows=100,
        description="Data with GA-optimized labeling parameters (profit/loss thresholds)",
    ),
    StageName.FINAL_LABELS.value: StageSchema(
        required_columns=["datetime"],
        min_rows=100,
        description="Data with final optimized labels",
    ),
    StageName.CREATE_SPLITS.value: StageSchema(
        required_columns=["datetime"],
        min_rows=50,
        description="Data split into train/val/test",
    ),
    StageName.FEATURE_SCALING.value: StageSchema(
        required_columns=["datetime"],
        min_rows=50,
        max_nan_ratio=0.0,  # No NaNs allowed after scaling
        description="Scaled features ready for training",
    ),
    StageName.BUILD_DATASETS.value: StageSchema(
        required_columns=["datetime"],
        min_rows=50,
        max_nan_ratio=0.0,
        description="Final datasets built for training",
    ),
    StageName.VALIDATE_SCALED.value: StageSchema(
        required_columns=["datetime"],
        min_rows=50,
        max_nan_ratio=0.0,  # No NaNs allowed in scaled data
        description="Validation of scaled data before training",
    ),
    StageName.VALIDATE.value: StageSchema(
        required_columns=["datetime"],
        min_rows=50,
        description="General data validation stage",
    ),
    StageName.GENERATE_REPORT.value: StageSchema(
        required_columns=[],  # Report generation may not produce DataFrames
        min_rows=0,
        description="Pipeline report generation (metadata/statistics)",
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


# Define expected column patterns for stage transitions
STAGE_TRANSITION_REQUIREMENTS: dict[tuple[str, str], dict[str, list[str]]] = {
    # From data_generation to data_cleaning
    (StageName.DATA_GENERATION.value, StageName.DATA_CLEANING.value): {
        "required": ["datetime", "open", "high", "low", "close", "volume"],
        "no_nan": ["open", "high", "low", "close"],
    },
    # From data_cleaning to feature_engineering
    (StageName.DATA_CLEANING.value, StageName.FEATURE_ENGINEERING.value): {
        "required": ["datetime", "open", "high", "low", "close", "volume"],
        "no_nan": ["open", "high", "low", "close"],
    },
    # From feature_engineering to initial_labeling
    (StageName.FEATURE_ENGINEERING.value, StageName.INITIAL_LABELING.value): {
        "required": ["datetime"],
        "no_nan": [],  # Features may have some NaN at edges
    },
    # From initial_labeling to ga_optimize
    (StageName.INITIAL_LABELING.value, StageName.GA_OPTIMIZE.value): {
        "required": ["datetime"],
        "no_nan": [],
    },
    # From ga_optimize to final_labels
    (StageName.GA_OPTIMIZE.value, StageName.FINAL_LABELS.value): {
        "required": ["datetime"],
        "no_nan": [],
    },
    # From final_labels to create_splits
    (StageName.FINAL_LABELS.value, StageName.CREATE_SPLITS.value): {
        "required": ["datetime"],
        "no_nan": [],
    },
    # From create_splits to feature_scaling
    (StageName.CREATE_SPLITS.value, StageName.FEATURE_SCALING.value): {
        "required": ["datetime"],
        "no_nan": [],
    },
    # From feature_scaling to build_datasets
    (StageName.FEATURE_SCALING.value, StageName.BUILD_DATASETS.value): {
        "required": ["datetime"],
        "no_nan": [],  # All columns should be clean after scaling
    },
    # From build_datasets to validate_scaled
    (StageName.BUILD_DATASETS.value, StageName.VALIDATE_SCALED.value): {
        "required": ["datetime"],
        "no_nan": [],
    },
}


def validate_stage_transition(
    output_df: pd.DataFrame,
    from_stage: str,
    to_stage: str,
    strict: bool = True,
) -> list[str]:
    """Validate data when transitioning between pipeline stages.

    This function checks that data meets requirements when passing from
    one pipeline stage to the next. It catches data corruption, missing
    columns, and unexpected NaN values early.

    Args:
        output_df: DataFrame output from the source stage
        from_stage: Name of the source stage (use StageName.value)
        to_stage: Name of the target stage (use StageName.value)
        strict: If True, raise StageValidationError on critical errors

    Returns:
        List of warning messages (non-blocking issues)

    Raises:
        StageValidationError: If strict=True and critical validation fails

    Example:
        >>> from src.data.pipeline.stage_registry import StageName
        >>> warnings = validate_stage_transition(
        ...     df,
        ...     StageName.DATA_CLEANING.value,
        ...     StageName.FEATURE_ENGINEERING.value,
        ... )
    """
    warnings: list[str] = []
    critical_issues: list[str] = []

    # Get transition-specific requirements
    transition_key = (from_stage, to_stage)
    requirements = STAGE_TRANSITION_REQUIREMENTS.get(transition_key)

    if requirements is None:
        # No specific requirements for this transition, use target schema
        logger.debug(
            f"No specific transition requirements for {from_stage} -> {to_stage}, "
            "using target stage schema"
        )
        # Validate against target schema
        is_valid, issues = validate_stage_output(output_df, to_stage, raise_on_failure=False)
        if not is_valid:
            if strict:
                raise StageValidationError(
                    f"{from_stage} -> {to_stage}",
                    issues,
                )
            warnings.extend(issues)
        return warnings

    # Check required columns exist
    required_cols = requirements.get("required", [])
    missing_cols = set(required_cols) - set(output_df.columns)
    if missing_cols:
        critical_issues.append(f"Missing required columns for {to_stage}: {sorted(missing_cols)}")

    # Check no NaN in key columns
    no_nan_cols = requirements.get("no_nan", [])
    for col in no_nan_cols:
        if col in output_df.columns:
            nan_count = int(output_df[col].isna().sum())
            if nan_count > 0:
                critical_issues.append(
                    f"Column '{col}' has {nan_count} NaN values "
                    f"(not allowed for {from_stage} -> {to_stage} transition)"
                )

    # Check data types for key columns (OHLCV should be numeric)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        col_exists_and_required = col in output_df.columns and col in required_cols
        if col_exists_and_required and not pd.api.types.is_numeric_dtype(output_df[col]):
            critical_issues.append(f"Column '{col}' should be numeric, got {output_df[col].dtype}")

    # Check datetime column is proper datetime type
    if "datetime" in output_df.columns:
        dt_col_not_datetime = not pd.api.types.is_datetime64_any_dtype(output_df["datetime"])
        idx_not_datetime = not pd.api.types.is_datetime64_any_dtype(output_df.index)
        if dt_col_not_datetime and idx_not_datetime:
            warnings.append(
                f"Column 'datetime' is not datetime type: {output_df['datetime'].dtype}"
            )

    # Check minimum row count
    target_schema = STAGE_SCHEMAS.get(to_stage)
    if target_schema and len(output_df) < target_schema.min_rows:
        critical_issues.append(
            f"Insufficient rows for {to_stage}: {len(output_df)} < {target_schema.min_rows}"
        )

    # Log results
    if critical_issues:
        logger.warning(
            f"Stage transition {from_stage} -> {to_stage} validation failed: " f"{critical_issues}"
        )
        if strict:
            raise StageValidationError(f"{from_stage} -> {to_stage}", critical_issues)
        warnings.extend(critical_issues)
    else:
        logger.debug(
            f"Stage transition {from_stage} -> {to_stage} validated: "
            f"{len(output_df)} rows, {len(output_df.columns)} columns"
        )

    return warnings


def register_stage_schema(stage_name: str, schema: StageSchema) -> None:
    """Register a custom schema for a stage."""
    STAGE_SCHEMAS[stage_name] = schema
    logger.debug(f"Registered schema for stage '{stage_name}'")


__all__ = [
    "StageName",
    "StageSchema",
    "StageValidationError",
    "STAGE_SCHEMAS",
    "STAGE_TRANSITION_REQUIREMENTS",
    "validate_stage_output",
    "validate_stage_transition",
    "get_stage_schema",
    "register_stage_schema",
]
