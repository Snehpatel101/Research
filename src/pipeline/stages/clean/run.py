"""
Stage 2: Data Cleaning.

Orchestration logic for Stage 2 of the pipeline.
Cleans and resamples validated 1-minute OHLCV data to target timeframe bars.
Uses the DataCleaner module for comprehensive gap detection, outlier removal,
and quality reporting.

Supports multi-timeframe output via config.output_timeframes or --output-timeframes CLI flag.
"""

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# RAW DATA SCHEMA VALIDATION (DATA-002)
# =============================================================================

# Required OHLCV columns for raw data
REQUIRED_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}

# Columns that can serve as datetime identifier
DATETIME_COLUMNS = {"datetime", "timestamp", "date", "time"}


@dataclass
class RawDataValidationResult:
    """Result of raw data schema validation."""

    is_valid: bool
    missing_columns: list[str]
    has_datetime: bool
    dtype_issues: dict[str, str]
    row_count: int
    warnings: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "is_valid": self.is_valid,
            "missing_columns": self.missing_columns,
            "has_datetime": self.has_datetime,
            "dtype_issues": self.dtype_issues,
            "row_count": self.row_count,
            "warnings": self.warnings,
        }


def validate_raw_data_schema(df: pd.DataFrame, file_path: Path) -> RawDataValidationResult:
    """
    Validate raw data schema for OHLCV requirements.

    Checks:
    - Required OHLCV columns exist: open, high, low, close, volume
    - Datetime index or timestamp column exists
    - Data types are correct (numeric for OHLCV)

    Args:
        df: DataFrame to validate
        file_path: Path to the file (for logging)

    Returns:
        RawDataValidationResult with validation details

    Note:
        This function does NOT block the pipeline on validation failure.
        It logs warnings and returns results for visibility.
    """
    warnings: list[str] = []
    dtype_issues: dict[str, str] = {}

    # Normalize column names for case-insensitive check
    df_columns_lower = {col.lower(): col for col in df.columns}

    # Check for required OHLCV columns
    missing_columns = []
    for col in REQUIRED_OHLCV_COLUMNS:
        if col not in df_columns_lower:
            missing_columns.append(col)

    # Check for datetime
    has_datetime = False

    # Check if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        has_datetime = True
    else:
        # Check for datetime columns
        for dt_col in DATETIME_COLUMNS:
            if dt_col in df_columns_lower:
                actual_col = df_columns_lower[dt_col]
                try:
                    # Try to convert to datetime to validate
                    pd.to_datetime(df[actual_col].head(10))
                    has_datetime = True
                    break
                except (ValueError, TypeError):
                    warnings.append(f"Column '{actual_col}' exists but cannot be parsed as datetime")

    if not has_datetime:
        warnings.append("No datetime index or valid datetime column found")

    # Validate data types for OHLCV columns
    numeric_columns = {"open", "high", "low", "close", "volume"}
    for col in numeric_columns:
        if col in df_columns_lower:
            actual_col = df_columns_lower[col]
            if not pd.api.types.is_numeric_dtype(df[actual_col]):
                dtype_issues[col] = f"Expected numeric, got {df[actual_col].dtype}"

    # Determine overall validity
    is_valid = len(missing_columns) == 0 and has_datetime and len(dtype_issues) == 0

    result = RawDataValidationResult(
        is_valid=is_valid,
        missing_columns=missing_columns,
        has_datetime=has_datetime,
        dtype_issues=dtype_issues,
        row_count=len(df),
        warnings=warnings,
    )

    # Log validation results
    if is_valid:
        logger.info(f"Raw data validation PASSED for {file_path.name}: {len(df)} rows")
    else:
        logger.warning(
            f"Raw data validation issues for {file_path.name}: "
            f"missing_cols={missing_columns}, dtype_issues={dtype_issues}, "
            f"has_datetime={has_datetime}"
        )
        for warn in warnings:
            logger.warning(f"  - {warn}")

    return result

# Import from local modules
from . import clean_symbol_data, clean_symbol_data_multi_timeframe

# StageResult imports - adjust path based on pipeline structure
try:
    from src.pipeline.utils import StageResult, create_failed_result, create_stage_result
except ImportError:
    # Fallback for different import paths
    from pipeline.utils import StageResult, create_failed_result, create_stage_result


def run_data_cleaning(config: "PipelineConfig", manifest: "ArtifactManifest") -> "StageResult":
    """
    Stage 2: Data Cleaning - Multi-Timeframe aware.

    Uses validated data from Stage 1 and resamples from 1-minute to target timeframe(s).
    Handles missing data, outliers, and ensures data quality through the DataCleaner
    module.

    Configuration options (from config):
        - target_timeframe: Primary training timeframe (default: '5min')
        - output_timeframes: List of timeframes to produce (default: [target_timeframe])
        - max_gap_minutes: Maximum gap to fill in minutes (default: 30)

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 2: Data Cleaning")
    logger.info("=" * 70)

    try:
        # Use validated data from Stage 1
        validated_data_dir = config.raw_data_dir / "validated"

        # Get cleaning configuration with defaults
        max_gap_minutes = getattr(config, "max_gap_minutes", 30)

        # Get effective output timeframes (single or multiple)
        output_timeframes = config.effective_output_timeframes
        is_multi_tf = len(output_timeframes) > 1

        logger.info(f"Output timeframes: {output_timeframes}")
        logger.info(f"Multi-TF mode: {is_multi_tf}")
        logger.info(f"Max gap fill: {max_gap_minutes} minutes")

        artifacts = []
        cleaning_metadata = {}

        for symbol in config.symbols:
            # Look for validated data first, fall back to raw if not found
            input_path = validated_data_dir / f"{symbol}_1m_validated.parquet"
            using_raw_fallback = False

            if not input_path.exists():
                # Fall back to raw data (for backward compatibility)
                logger.warning(f"Validated data not found for {symbol}, using raw data")
                using_raw_fallback = True
                input_path = config.raw_data_dir / f"{symbol}_1m.parquet"
                if not input_path.exists():
                    input_path = config.raw_data_dir / f"{symbol}_1m.csv"

            if not input_path.exists():
                raise FileNotFoundError(f"No input data found for {symbol}")

            cleaning_metadata[symbol] = {}

            # DATA-002: Validate raw data schema when using fallback
            if using_raw_fallback:
                try:
                    if input_path.suffix == ".parquet":
                        raw_df = pd.read_parquet(input_path)
                    else:
                        raw_df = pd.read_csv(input_path)

                    validation_result = validate_raw_data_schema(raw_df, input_path)
                    cleaning_metadata[symbol]["raw_data_validation"] = validation_result.to_dict()
                    cleaning_metadata[symbol]["used_raw_fallback"] = True

                    if not validation_result.is_valid:
                        logger.warning(
                            f"Raw data for {symbol} has schema issues but continuing: "
                            f"{validation_result.to_dict()}"
                        )
                except Exception as e:
                    logger.warning(f"Could not validate raw data schema for {symbol}: {e}")
                    cleaning_metadata[symbol]["raw_validation_error"] = str(e)

            if is_multi_tf:
                # Multi-TF path: use clean_symbol_data_multi_timeframe
                logger.info(f"Cleaning {symbol} to {len(output_timeframes)} timeframes...")

                results = clean_symbol_data_multi_timeframe(
                    input_path=input_path,
                    output_dir=config.clean_data_dir,
                    symbol=symbol,
                    timeframes=output_timeframes,
                    include_timeframe_metadata=True,
                    max_gap_minutes=max_gap_minutes,
                )

                # Register each timeframe output
                for tf in output_timeframes:
                    output_path = config.clean_data_dir / f"{symbol}_{tf}_clean.parquet"
                    if output_path.exists():
                        artifacts.append(output_path)
                        file_size = output_path.stat().st_size

                        cleaning_metadata[symbol][tf] = {
                            "output_file": str(output_path),
                            "file_size_bytes": file_size,
                            "rows": len(results.get(tf, [])) if tf in results else 0,
                        }

                        manifest.add_artifact(
                            name=f"clean_data_{symbol}_{tf}",
                            file_path=output_path,
                            stage="data_cleaning",
                            metadata={
                                "symbol": symbol,
                                "source": str(input_path),
                                "timeframe": tf,
                            },
                        )
            else:
                # Single-TF path: use clean_symbol_data (backward compat)
                target_timeframe = output_timeframes[0]
                output_path = config.clean_data_dir / f"{symbol}_{target_timeframe}_clean.parquet"

                logger.info(f"Cleaning {symbol}: {input_path.name} -> {output_path.name}")

                clean_symbol_data(
                    input_path=input_path,
                    output_path=output_path,
                    symbol=symbol,
                    target_timeframe=target_timeframe,
                    include_timeframe_metadata=True,
                    max_gap_minutes=max_gap_minutes,
                )

                if output_path.exists():
                    artifacts.append(output_path)
                    file_size = output_path.stat().st_size

                    cleaning_metadata[symbol][target_timeframe] = {
                        "input_file": str(input_path),
                        "output_file": str(output_path),
                        "file_size_bytes": file_size,
                    }

                    manifest.add_artifact(
                        name=f"clean_data_{symbol}_{target_timeframe}",
                        file_path=output_path,
                        stage="data_cleaning",
                        metadata={
                            "symbol": symbol,
                            "source": str(input_path),
                            "timeframe": target_timeframe,
                        },
                    )

        return create_stage_result(
            stage_name="data_cleaning",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                "cleaning_results": cleaning_metadata,
                "output_timeframes": output_timeframes,
                "multi_tf_mode": is_multi_tf,
            },
        )

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(stage_name="data_cleaning", start_time=start_time, error=str(e))
