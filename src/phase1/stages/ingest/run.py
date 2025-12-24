"""
Stage 1: Data Generation and Ingestion with Validation.

Orchestration logic for Stage 1 of the pipeline.
Generates synthetic data if needed and validates/standardizes raw OHLCV data.
"""
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)

# Import from local modules
from . import DataIngestor

# StageResult imports - adjust path based on pipeline structure
try:
    from src.pipeline.utils import StageResult, StageStatus, create_stage_result, create_failed_result
except ImportError:
    # Fallback for different import paths
    from pipeline.utils import StageResult, StageStatus, create_stage_result, create_failed_result


def run_data_generation(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> 'StageResult':
    """
    Stage 1: Data Generation / Acquisition & Validation.

    This stage:
    1. Generates synthetic data if needed or raw data doesn't exist
    2. Validates and standardizes OHLCV data using DataIngestor
    3. Fixes any OHLCV violations (high < low, etc.)
    4. Saves validated data to parquet format

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 1: Data Generation / Acquisition & Validation")
    logger.info("=" * 70)

    try:
        from src.phase1.generate_synthetic_data import main as generate_data

        # Check if raw data exists
        raw_files_exist = all(
            (config.raw_data_dir / f"{s}_1m.parquet").exists() or
            (config.raw_data_dir / f"{s}_1m.csv").exists()
            for s in config.symbols
        )

        artifacts = []
        ingestion_metadata = {}

        # Step 1: Generate synthetic data if needed
        if not raw_files_exist or config.use_synthetic_data:
            logger.info("Generating synthetic data...")
            generate_data()
            logger.info("Synthetic data generation complete.")
        else:
            logger.info("Raw data files already exist. Skipping generation.")

        # Step 2: ALWAYS run DataIngestor for validation and standardization
        logger.info("\nRunning DataIngestor for validation and standardization...")

        # Create validated data output directory
        validated_data_dir = config.raw_data_dir / "validated"
        validated_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DataIngestor
        ingestor = DataIngestor(
            raw_data_dir=config.raw_data_dir,
            output_dir=validated_data_dir,
            source_timezone='UTC',
            symbol_col='symbol'
        )

        # Process each symbol
        total_violations = 0
        for symbol in config.symbols:
            # Find the raw data file
            raw_file = None
            for ext in ['.parquet', '.csv']:
                candidate = config.raw_data_dir / f"{symbol}_1m{ext}"
                if candidate.exists():
                    raw_file = candidate
                    break

            if raw_file is None:
                raise FileNotFoundError(
                    f"Raw data file not found for symbol {symbol}. "
                    f"Expected: {config.raw_data_dir}/{symbol}_1m.parquet or .csv"
                )

            logger.info(f"\nProcessing {symbol} from {raw_file.name}...")

            # Ingest and validate the file
            df, metadata = ingestor.ingest_file(
                file_path=raw_file,
                symbol=symbol,
                validate=True
            )

            # Check for validation issues
            validation_info = metadata.get('validation', {})
            violations = validation_info.get('violations', {})
            if violations:
                violation_count = sum(violations.values())
                total_violations += violation_count
                logger.warning(
                    f"Symbol {symbol}: Fixed {violation_count} OHLCV violations: {violations}"
                )

            # Save validated data
            output_path = ingestor.save_parquet(df, f"{symbol}_1m_validated", metadata)
            artifacts.append(output_path)

            # Store metadata for this symbol
            ingestion_metadata[symbol] = {
                'source_file': str(raw_file),
                'validated_file': str(output_path),
                'raw_rows': metadata.get('raw_rows', 0),
                'final_rows': metadata.get('final_rows', 0),
                'date_range': metadata.get('date_range', {}),
                'violations_fixed': sum(violations.values()) if violations else 0,
                'validation_details': violations
            }

            # Add to manifest
            manifest.add_artifact(
                name=f"validated_data_{symbol}",
                file_path=output_path,
                stage="data_generation",
                metadata=ingestion_metadata[symbol]
            )

            logger.info(
                f"Validated {symbol}: {metadata.get('raw_rows', 0):,} -> "
                f"{metadata.get('final_rows', 0):,} rows"
            )

        # Log summary
        logger.info("\n" + "-" * 50)
        logger.info("INGESTION SUMMARY")
        logger.info("-" * 50)
        for symbol, meta in ingestion_metadata.items():
            logger.info(
                f"  {symbol}: {meta['raw_rows']:,} raw -> {meta['final_rows']:,} validated "
                f"({meta['violations_fixed']} fixes)"
            )
        if total_violations > 0:
            logger.warning(f"Total OHLCV violations fixed: {total_violations}")
        else:
            logger.info("No OHLCV violations found - data is clean!")

        return create_stage_result(
            stage_name="data_generation",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                'symbols': config.symbols,
                'ingestion_results': ingestion_metadata,
                'total_violations_fixed': total_violations
            }
        )

    except Exception as e:
        logger.error(f"Data generation/ingestion failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="data_generation",
            start_time=start_time,
            error=str(e)
        )
