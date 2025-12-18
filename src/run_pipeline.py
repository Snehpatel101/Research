#!/usr/bin/env python3
"""
Complete data pipeline runner for Phase 1.

This script runs all three stages:
1. Data Ingestion
2. Data Cleaning
3. Feature Engineering

Usage:
    python run_pipeline.py --raw-dir data/raw --output-dir data
"""

import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

from stages import DataIngestor, DataCleaner, FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    raw_data_dir: str = 'data/raw',
    output_base_dir: str = 'data',
    timeframe: str = '1min',
    source_timezone: str = 'UTC',
    gap_fill_method: str = 'forward',
    max_gap_fill_minutes: int = 5,
    outlier_method: str = 'atr',
    atr_threshold: float = 5.0,
    pattern: str = '*.parquet'
):
    """
    Run complete data pipeline.

    Parameters:
    -----------
    raw_data_dir : Directory containing raw data files
    output_base_dir : Base directory for output
    timeframe : Data timeframe (e.g., '1min', '5min')
    source_timezone : Source timezone of raw data
    gap_fill_method : Gap filling method ('forward', 'interpolate', 'none')
    max_gap_fill_minutes : Maximum gap to fill in minutes
    outlier_method : Outlier detection method ('atr', 'zscore', 'iqr', 'all')
    atr_threshold : ATR multiplier for spike detection
    pattern : File pattern to match

    Returns:
    --------
    dict : Pipeline execution report
    """
    output_base_dir = Path(output_base_dir)
    raw_dir = Path(raw_data_dir)
    clean_dir = output_base_dir / 'clean'
    features_dir = output_base_dir / 'features'

    pipeline_report = {
        'pipeline_start': datetime.now().isoformat(),
        'configuration': {
            'raw_data_dir': str(raw_data_dir),
            'output_base_dir': str(output_base_dir),
            'timeframe': timeframe,
            'source_timezone': source_timezone,
            'gap_fill_method': gap_fill_method,
            'max_gap_fill_minutes': max_gap_fill_minutes,
            'outlier_method': outlier_method,
            'atr_threshold': atr_threshold,
            'pattern': pattern
        },
        'stages': {}
    }

    # ========================================================================
    # STAGE 1: DATA INGESTION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("="*80 + "\n")

    try:
        ingestor = DataIngestor(
            raw_data_dir=raw_dir,
            output_dir=raw_dir,  # Ingest updates files in-place
            source_timezone=source_timezone
        )

        ingestion_results = ingestor.ingest_directory(pattern=pattern, validate=True)

        pipeline_report['stages']['stage1_ingestion'] = {
            'status': 'success',
            'symbols_processed': len(ingestion_results),
            'results': ingestion_results
        }

        logger.info(f"\n✓ Stage 1 complete. Processed {len(ingestion_results)} symbols.")

    except Exception as e:
        logger.error(f"Stage 1 failed: {e}", exc_info=True)
        pipeline_report['stages']['stage1_ingestion'] = {
            'status': 'failed',
            'error': str(e)
        }
        return pipeline_report

    # ========================================================================
    # STAGE 2: DATA CLEANING
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: DATA CLEANING")
    logger.info("="*80 + "\n")

    try:
        cleaner = DataCleaner(
            input_dir=raw_dir,
            output_dir=clean_dir,
            timeframe=timeframe,
            gap_fill_method=gap_fill_method,
            max_gap_fill_minutes=max_gap_fill_minutes,
            outlier_method=outlier_method,
            atr_threshold=atr_threshold
        )

        cleaning_results = cleaner.clean_directory(pattern=pattern)

        pipeline_report['stages']['stage2_cleaning'] = {
            'status': 'success',
            'symbols_processed': len(cleaning_results),
            'results': cleaning_results
        }

        logger.info(f"\n✓ Stage 2 complete. Cleaned {len(cleaning_results)} symbols.")

    except Exception as e:
        logger.error(f"Stage 2 failed: {e}", exc_info=True)
        pipeline_report['stages']['stage2_cleaning'] = {
            'status': 'failed',
            'error': str(e)
        }
        return pipeline_report

    # ========================================================================
    # STAGE 3: FEATURE ENGINEERING
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("="*80 + "\n")

    try:
        engineer = FeatureEngineer(
            input_dir=clean_dir,
            output_dir=features_dir,
            timeframe=timeframe
        )

        feature_results = engineer.process_directory(pattern=pattern)

        pipeline_report['stages']['stage3_features'] = {
            'status': 'success',
            'symbols_processed': len(feature_results),
            'results': feature_results
        }

        logger.info(f"\n✓ Stage 3 complete. Generated features for {len(feature_results)} symbols.")

    except Exception as e:
        logger.error(f"Stage 3 failed: {e}", exc_info=True)
        pipeline_report['stages']['stage3_features'] = {
            'status': 'failed',
            'error': str(e)
        }
        return pipeline_report

    # ========================================================================
    # PIPELINE COMPLETE
    # ========================================================================
    pipeline_report['pipeline_end'] = datetime.now().isoformat()
    pipeline_report['status'] = 'success'

    # Save pipeline report
    report_path = output_base_dir / 'pipeline_report.json'
    with open(report_path, 'w') as f:
        json.dump(pipeline_report, f, indent=2, default=str)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nPipeline report saved to: {report_path}")

    return pipeline_report


def print_summary(report: dict):
    """Print pipeline execution summary."""
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)

    for stage_name, stage_data in report['stages'].items():
        stage_title = stage_name.replace('_', ' ').title()
        print(f"\n{stage_title}:")
        print(f"  Status: {stage_data['status']}")

        if stage_data['status'] == 'success':
            print(f"  Symbols processed: {stage_data['symbols_processed']}")

            if stage_name == 'stage1_ingestion':
                for symbol, data in stage_data['results'].items():
                    print(f"\n  {symbol}:")
                    print(f"    Rows: {data['raw_rows']:,} -> {data['final_rows']:,}")
                    violations = data.get('validation', {}).get('violations', {})
                    if violations:
                        print(f"    Violations fixed: {sum(violations.values())}")

            elif stage_name == 'stage2_cleaning':
                for symbol, data in stage_data['results'].items():
                    print(f"\n  {symbol}:")
                    print(f"    Rows: {data['initial_rows']:,} -> {data['final_rows']:,}")
                    print(f"    Retention: {data['retention_pct']:.2f}%")
                    print(f"    Duplicates removed: {data['duplicates']['n_duplicates']}")
                    print(f"    Outliers removed: {data['outliers']['total_outliers']}")

            elif stage_name == 'stage3_features':
                for symbol, data in stage_data['results'].items():
                    print(f"\n  {symbol}:")
                    print(f"    Features added: {data['features_added']}")
                    print(f"    Total columns: {data['final_columns']}")
                    print(f"    Final rows: {data['final_rows']:,}")

        else:
            print(f"  Error: {stage_data.get('error', 'Unknown error')}")

    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Complete data pipeline for Phase 1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--raw-dir', type=str, default='data/raw',
                        help='Raw data directory')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output base directory')
    parser.add_argument('--timeframe', type=str, default='1min',
                        help='Data timeframe (e.g., 1min, 5min)')
    parser.add_argument('--timezone', type=str, default='UTC',
                        help='Source timezone of raw data')
    parser.add_argument('--gap-fill', type=str, default='forward',
                        choices=['forward', 'interpolate', 'none'],
                        help='Gap filling method')
    parser.add_argument('--max-gap-fill', type=int, default=5,
                        help='Maximum gap to fill (minutes)')
    parser.add_argument('--outlier-method', type=str, default='atr',
                        choices=['atr', 'zscore', 'iqr', 'all'],
                        help='Outlier detection method')
    parser.add_argument('--atr-threshold', type=float, default=5.0,
                        help='ATR threshold for spike detection')
    parser.add_argument('--pattern', type=str, default='*.parquet',
                        help='File pattern to match')

    args = parser.parse_args()

    # Run pipeline
    report = run_complete_pipeline(
        raw_data_dir=args.raw_dir,
        output_base_dir=args.output_dir,
        timeframe=args.timeframe,
        source_timezone=args.timezone,
        gap_fill_method=args.gap_fill,
        max_gap_fill_minutes=args.max_gap_fill,
        outlier_method=args.outlier_method,
        atr_threshold=args.atr_threshold,
        pattern=args.pattern
    )

    # Print summary
    print_summary(report)


if __name__ == '__main__':
    main()
