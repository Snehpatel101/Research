#!/usr/bin/env python3
"""
Complete Phase 1 Pipeline Runner
Runs all stages including new splitting, validation, backtesting, and reporting modules
"""
import sys
sys.path.insert(0, '/home/user/Research/src')

import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user/Research/logs/phase1_complete.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("phase1_complete")


def main():
    """Run the complete Phase 1 pipeline with all new stages."""
    start_time = datetime.now()

    logger.info("="*80)
    logger.info("PHASE 1: COMPLETE DATA PREPARATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Started at: {start_time}")

    from config import (
        FINAL_DATA_DIR, SPLITS_DIR, RESULTS_DIR, REPORTS_DIR,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO, PURGE_BARS, EMBARGO_BARS
    )

    # Verify that labeled data exists
    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    if not data_path.exists():
        logger.error(f"Combined labeled data not found at {data_path}")
        logger.error("Please run the basic Phase 1 pipeline first (data cleaning, features, labeling)")
        sys.exit(1)

    logger.info(f"Using data file: {data_path}")

    # =========================================================================
    # STAGE 7: CREATE SPLITS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 7: TIME-BASED SPLITTING WITH PURGING AND EMBARGO")
    logger.info("="*80)

    try:
        from stages.stage7_splits import create_splits

        # Generate run ID
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        split_metadata = create_splits(
            data_path=data_path,
            output_dir=SPLITS_DIR,
            run_id=run_id,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            purge_bars=PURGE_BARS,
            embargo_bars=EMBARGO_BARS
        )

        logger.info(f"✓ Stage 7 completed successfully")
        logger.info(f"  Split directory: {split_metadata['split_dir']}")

        # Store paths for later stages
        split_dir = Path(split_metadata['split_dir'])
        split_config_path = split_dir / "split_config.json"
        train_indices_path = split_dir / "train.npy"
        val_indices_path = split_dir / "val.npy"
        test_indices_path = split_dir / "test.npy"

    except Exception as e:
        logger.error(f"✗ Stage 7 failed: {e}", exc_info=True)
        sys.exit(1)

    # =========================================================================
    # STAGE 8: COMPREHENSIVE VALIDATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 8: COMPREHENSIVE DATA VALIDATION")
    logger.info("="*80)

    try:
        from stages.stage8_validate import validate_data

        validation_output_path = RESULTS_DIR / "validation_report.json"

        validation_summary = validate_data(
            data_path=data_path,
            output_path=validation_output_path,
            horizons=[1, 5, 20]
        )

        logger.info(f"✓ Stage 8 completed successfully")
        logger.info(f"  Validation status: {validation_summary['status']}")
        logger.info(f"  Issues: {validation_summary['issues_count']}")
        logger.info(f"  Warnings: {validation_summary['warnings_count']}")

        if validation_summary['status'] == 'FAILED':
            logger.warning("⚠ Validation found issues. Review the validation report before proceeding.")

    except Exception as e:
        logger.error(f"✗ Stage 8 failed: {e}", exc_info=True)
        logger.warning("Continuing to next stage despite validation failure...")
        validation_output_path = None

    # =========================================================================
    # BASELINE BACKTEST
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("BASELINE BACKTEST: SIMPLE LABEL-FOLLOWING STRATEGY")
    logger.info("="*80)

    try:
        from stages.baseline_backtest import run_baseline_backtest

        backtest_output_dir = RESULTS_DIR / "baseline_backtest"

        # Run backtest for all horizons
        backtest_results = {}
        for horizon in [1, 5, 20]:
            logger.info(f"\nRunning backtest for horizon {horizon}...")

            result = run_baseline_backtest(
                data_path=data_path,
                split_indices_path=test_indices_path,
                output_dir=backtest_output_dir,
                horizon=horizon,
                quality_threshold=0.5,
                commission=0.0001
            )

            backtest_results[horizon] = result

            if result['metrics'].get('total_trades', 0) > 0:
                logger.info(f"  Horizon {horizon}: {result['metrics']['total_trades']} trades, "
                          f"{result['metrics']['win_rate']:.1f}% win rate, "
                          f"{result['metrics']['total_return_pct']:.2f}% return")

        logger.info(f"\n✓ Baseline backtest completed successfully")
        logger.info(f"  Output directory: {backtest_output_dir}")

    except Exception as e:
        logger.error(f"✗ Baseline backtest failed: {e}", exc_info=True)
        logger.warning("Continuing to report generation despite backtest failure...")
        backtest_output_dir = None

    # =========================================================================
    # GENERATE COMPREHENSIVE REPORT
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPREHENSIVE PHASE 1 SUMMARY REPORT")
    logger.info("="*80)

    try:
        from stages.generate_report import generate_phase1_report

        output_files = generate_phase1_report(
            data_path=data_path,
            output_dir=REPORTS_DIR,
            validation_report_path=validation_output_path if validation_output_path else None,
            split_config_path=split_config_path,
            backtest_results_dir=backtest_output_dir
        )

        logger.info(f"\n✓ Report generation completed successfully")
        logger.info(f"  Markdown: {output_files['markdown']}")
        logger.info(f"  HTML:     {output_files['html']}")
        logger.info(f"  JSON:     {output_files['json']}")
        logger.info(f"  Charts:   {output_files['charts_dir']}")

    except Exception as e:
        logger.error(f"✗ Report generation failed: {e}", exc_info=True)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    end_time = datetime.now()
    elapsed = end_time - start_time

    logger.info("\n" + "="*80)
    logger.info("PHASE 1 PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"Started:  {start_time}")
    logger.info(f"Finished: {end_time}")
    logger.info(f"Duration: {elapsed}")

    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("1. Review the comprehensive report:")
    logger.info(f"   - Markdown: {REPORTS_DIR}/phase1_summary.md")
    logger.info(f"   - HTML:     {REPORTS_DIR}/phase1_summary.html")
    logger.info("")
    logger.info("2. Check validation results:")
    logger.info(f"   - {RESULTS_DIR}/validation_report.json")
    logger.info("")
    logger.info("3. Review baseline backtest:")
    logger.info(f"   - {RESULTS_DIR}/baseline_backtest/")
    logger.info("")
    logger.info("4. If all quality gates pass, proceed to Phase 2:")
    logger.info("   - Train base models (N-HiTS, TFT, PatchTST)")
    logger.info("   - Use splits from: {split_dir}")
    logger.info("")
    logger.info("="*80)


if __name__ == "__main__":
    main()
