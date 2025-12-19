#!/usr/bin/env python3
"""
Master runner for Labeling Pipeline (Stages 4-6)
Orchestrates triple-barrier labeling, GA optimization, and final label application
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("labeling_pipeline")


def main():
    """Run the complete labeling pipeline (Stages 4-6)."""
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("LABELING PIPELINE: STAGES 4-6")
    logger.info("=" * 80)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Stage 4: Initial Triple-Barrier Labeling
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 4: TRIPLE-BARRIER LABELING")
    logger.info("=" * 80)
    stage4_start = time.time()

    from stages.stage4_labeling import main as stage4_main
    try:
        stage4_main()
        stage4_time = time.time() - stage4_start
        logger.info(f"Stage 4 completed in {stage4_time:.1f} seconds")
    except Exception as e:
        logger.error(f"Stage 4 failed: {e}", exc_info=True)
        return

    # Stage 5: GA Optimization
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 5: GENETIC ALGORITHM OPTIMIZATION")
    logger.info("=" * 80)
    stage5_start = time.time()

    from stages.stage5_ga_optimize import main as stage5_main
    try:
        stage5_main()
        stage5_time = time.time() - stage5_start
        logger.info(f"Stage 5 completed in {stage5_time:.1f} seconds")
    except Exception as e:
        logger.error(f"Stage 5 failed: {e}", exc_info=True)
        return

    # Stage 6: Final Labels with Quality Scores
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 6: FINAL LABELING WITH QUALITY SCORES")
    logger.info("=" * 80)
    stage6_start = time.time()

    from stages.stage6_final_labels import main as stage6_main
    try:
        stage6_main()
        stage6_time = time.time() - stage6_start
        logger.info(f"Stage 6 completed in {stage6_time:.1f} seconds")
    except Exception as e:
        logger.error(f"Stage 6 failed: {e}", exc_info=True)
        return

    # Summary
    end_time = datetime.now()
    total_time = end_time - start_time

    logger.info("\n" + "=" * 80)
    logger.info("LABELING PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_time}")
    logger.info(f"  Stage 4 (Labeling):     {stage4_time:6.1f}s")
    logger.info(f"  Stage 5 (GA Optimize):  {stage5_time:6.1f}s")
    logger.info(f"  Stage 6 (Final Labels): {stage6_time:6.1f}s")
    logger.info("")
    logger.info("Output locations:")
    logger.info("  - Initial labels: data/labels/")
    logger.info("  - GA results:     config/ga_results/")
    logger.info("  - GA plots:       results/ga_plots/")
    logger.info("  - Final labels:   data/final/")
    logger.info("  - Report:         results/labeling_report.md")
    logger.info("")


if __name__ == "__main__":
    main()
