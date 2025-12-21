"""
Stage 2: Data Cleaning.

Cleans and resamples validated 1-minute OHLCV data to 5-minute bars.
"""
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import StageResult, StageStatus, create_stage_result, create_failed_result

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)


def run_data_cleaning(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 2: Data Cleaning.

    Uses validated data from Stage 1 and resamples from 1-minute to 5-minute bars.
    Handles missing data, outliers, and ensures data quality.

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
        from data_cleaning import clean_symbol_data

        # Use validated data from Stage 1
        validated_data_dir = config.raw_data_dir / "validated"

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

            output_path = config.clean_data_dir / f"{symbol}_5m_clean.parquet"

            logger.info(f"Cleaning {symbol}: {input_path.name} -> {output_path.name}")
            clean_symbol_data(input_path, output_path, symbol)

            if output_path.exists():
                artifacts.append(output_path)

                # Get file size for metadata
                file_size = output_path.stat().st_size

                cleaning_metadata[symbol] = {
                    'input_file': str(input_path),
                    'output_file': str(output_path),
                    'file_size_bytes': file_size
                }

                manifest.add_artifact(
                    name=f"clean_data_{symbol}",
                    file_path=output_path,
                    stage="data_cleaning",
                    metadata={'symbol': symbol, 'source': str(input_path)}
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
