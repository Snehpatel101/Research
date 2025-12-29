"""
Stage 2: Data Cleaning.

Orchestration logic for Stage 2 of the pipeline.
Cleans and resamples validated 1-minute OHLCV data to target timeframe bars.
Uses the DataCleaner module for comprehensive gap detection, outlier removal,
and quality reporting.
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
from . import clean_symbol_data

# StageResult imports - adjust path based on pipeline structure
try:
    from src.pipeline.utils import StageResult, create_failed_result, create_stage_result
except ImportError:
    # Fallback for different import paths
    from pipeline.utils import StageResult, create_failed_result, create_stage_result


def run_data_cleaning(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> 'StageResult':
    """
    Stage 2: Data Cleaning.

    Uses validated data from Stage 1 and resamples from 1-minute to target timeframe.
    Handles missing data, outliers, and ensures data quality through the DataCleaner
    module.

    Configuration options (from config):
        - target_timeframe: Target timeframe for resampling (default: '5min')
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
        target_timeframe = getattr(config, 'target_timeframe', '5min')
        max_gap_minutes = getattr(config, 'max_gap_minutes', 30)

        logger.info(f"Target timeframe: {target_timeframe}")
        logger.info(f"Max gap fill: {max_gap_minutes} minutes")

        artifacts = []
        cleaning_metadata = {}

        for symbol in config.symbols:
            # Look for validated data first, fall back to raw if not found
            input_path = validated_data_dir / f"{symbol}_1m_validated.parquet"
            if not input_path.exists():
                # Fall back to raw data (for backward compatibility)
                logger.warning(
                    f"Validated data not found for {symbol}, using raw data"
                )
                input_path = config.raw_data_dir / f"{symbol}_1m.parquet"
                if not input_path.exists():
                    input_path = config.raw_data_dir / f"{symbol}_1m.csv"

            if not input_path.exists():
                raise FileNotFoundError(f"No input data found for {symbol}")

            # Build output filename based on target timeframe
            output_path = config.clean_data_dir / f"{symbol}_{target_timeframe}_clean.parquet"

            logger.info(f"Cleaning {symbol}: {input_path.name} -> {output_path.name}")

            # Use clean_symbol_data with full configuration
            clean_symbol_data(
                input_path=input_path,
                output_path=output_path,
                symbol=symbol,
                target_timeframe=target_timeframe,
                include_timeframe_metadata=True,
                max_gap_minutes=max_gap_minutes
            )

            if output_path.exists():
                artifacts.append(output_path)

                # Get file size for metadata
                file_size = output_path.stat().st_size

                cleaning_metadata[symbol] = {
                    'input_file': str(input_path),
                    'output_file': str(output_path),
                    'file_size_bytes': file_size,
                    'target_timeframe': target_timeframe,
                    'max_gap_minutes': max_gap_minutes
                }

                manifest.add_artifact(
                    name=f"clean_data_{symbol}",
                    file_path=output_path,
                    stage="data_cleaning",
                    metadata={
                        'symbol': symbol,
                        'source': str(input_path),
                        'timeframe': target_timeframe
                    }
                )

        return create_stage_result(
            stage_name="data_cleaning",
            start_time=start_time,
            artifacts=artifacts,
            metadata={'cleaning_results': cleaning_metadata}
        )

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="data_cleaning",
            start_time=start_time,
            error=str(e)
        )
