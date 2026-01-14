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
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

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
            if not input_path.exists():
                # Fall back to raw data (for backward compatibility)
                logger.warning(f"Validated data not found for {symbol}, using raw data")
                input_path = config.raw_data_dir / f"{symbol}_1m.parquet"
                if not input_path.exists():
                    input_path = config.raw_data_dir / f"{symbol}_1m.csv"

            if not input_path.exists():
                raise FileNotFoundError(f"No input data found for {symbol}")

            cleaning_metadata[symbol] = {}

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
