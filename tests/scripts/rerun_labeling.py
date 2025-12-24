#!/usr/bin/env python3
"""
Re-run Phase 1 labeling stages with optimized barrier parameters.

This script re-runs stages 4-8 of the pipeline with fixed/optimized parameters:
- Stage 4: Triple-Barrier Labeling (with new parameters)
- Stage 5: GA Optimization (optional, can be skipped with preset params)
- Stage 6: Final Labels with Quality Scores
- Stage 7: Time-Based Splits
- Stage 8: Validation

Usage:
    python scripts/rerun_labeling.py --symbols MES,MGC
    python scripts/rerun_labeling.py --skip-ga  # Use preset params, skip GA
    python scripts/rerun_labeling.py --symbols MGC --skip-ga --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "stages"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / f"rerun_labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# OPTIMIZED BARRIER PARAMETERS
# These parameters were determined through extensive testing and optimization
# ==============================================================================
OPTIMIZED_BARRIER_PARAMS = {
    1: {"k_up": 0.5, "k_down": 0.5, "max_bars": 3},
    5: {"k_up": 0.75, "k_down": 0.75, "max_bars": 12},
    20: {"k_up": 1.0, "k_down": 1.0, "max_bars": 40},
}

# Default configuration
DEFAULT_SYMBOLS = ["MES", "MGC"]
HORIZONS = [1, 5, 20]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
PURGE_BARS = 60  # = max_bars for H20 (prevents label leakage)
EMBARGO_BARS = 288

# Validation thresholds
MAX_SINGLE_CLASS_PCT = 0.50  # No class should exceed 50%
MIN_SIGNAL_RATE = 0.50  # Long + Short should be at least 50%
MIN_LONG_SHORT_RATIO = 0.40
MAX_LONG_SHORT_RATIO = 0.60


class LabelingResult:
    """Container for labeling results and metrics."""

    def __init__(self):
        self.distributions: Dict[int, Dict[str, float]] = {}
        self.metrics: Dict[str, float] = {}
        self.validation_passed: bool = False
        self.issues: List[str] = []
        self.warnings: List[str] = []


def validate_label_distribution(
    df: pd.DataFrame, horizons: List[int]
) -> Tuple[bool, List[str], Dict[int, Dict]]:
    """
    Validate label distributions meet quality criteria.

    Returns:
        (passed, issues, distributions_by_horizon)
    """
    issues = []
    distributions = {}

    for horizon in horizons:
        label_col = f"label_h{horizon}"
        if label_col not in df.columns:
            issues.append(f"Missing label column: {label_col}")
            continue

        labels = df[label_col]
        total = len(labels)

        n_long = (labels == 1).sum()
        n_short = (labels == -1).sum()
        n_neutral = (labels == 0).sum()

        long_pct = n_long / total
        short_pct = n_short / total
        neutral_pct = n_neutral / total
        signal_rate = long_pct + short_pct

        # Long/short ratio (among signals only)
        if n_long + n_short > 0:
            long_ratio = n_long / (n_long + n_short)
        else:
            long_ratio = 0.5

        distributions[horizon] = {
            "long_count": int(n_long),
            "short_count": int(n_short),
            "neutral_count": int(n_neutral),
            "long_pct": float(long_pct),
            "short_pct": float(short_pct),
            "neutral_pct": float(neutral_pct),
            "signal_rate": float(signal_rate),
            "long_ratio": float(long_ratio),
        }

        # Check validation criteria
        # 1. No single class exceeds 50%
        for class_name, pct in [("long", long_pct), ("short", short_pct), ("neutral", neutral_pct)]:
            if pct > MAX_SINGLE_CLASS_PCT:
                issues.append(
                    f"h{horizon}: {class_name} class exceeds {MAX_SINGLE_CLASS_PCT*100:.0f}% "
                    f"({pct*100:.1f}%)"
                )

        # 2. Signal rate at least 50%
        if signal_rate < MIN_SIGNAL_RATE:
            issues.append(
                f"h{horizon}: Signal rate too low "
                f"({signal_rate*100:.1f}% < {MIN_SIGNAL_RATE*100:.0f}%)"
            )

        # 3. Long/short ratio between 0.4 and 0.6
        if not (MIN_LONG_SHORT_RATIO <= long_ratio <= MAX_LONG_SHORT_RATIO):
            issues.append(
                f"h{horizon}: Long/short ratio out of range "
                f"({long_ratio:.2f} not in [{MIN_LONG_SHORT_RATIO}, {MAX_LONG_SHORT_RATIO}])"
            )

    passed = len(issues) == 0
    return passed, issues, distributions


def run_stage4_labeling(
    symbols: List[str],
    horizons: List[int],
    barrier_params: Dict[int, Dict],
    dry_run: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run Stage 4: Triple-Barrier Labeling with specified parameters.

    Returns:
        Dictionary mapping symbol -> labeled DataFrame
    """
    from stage4_labeling import apply_triple_barrier

    logger.info("=" * 70)
    logger.info("STAGE 4: TRIPLE-BARRIER LABELING (Re-run)")
    logger.info("=" * 70)

    features_dir = PROJECT_ROOT / "data" / "features"
    labels_dir = PROJECT_ROOT / "data" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    labeled_data = {}

    for symbol in symbols:
        logger.info(f"\nProcessing {symbol}...")

        input_path = features_dir / f"{symbol}_5m_features.parquet"
        if not input_path.exists():
            logger.warning(f"Features file not found: {input_path}")
            continue

        df = pd.read_parquet(input_path)
        logger.info(f"  Loaded {len(df):,} rows from {input_path}")

        # Apply labeling for each horizon with optimized parameters
        for horizon in horizons:
            params = barrier_params[horizon]
            logger.info(
                f"  Horizon {horizon}: k_up={params['k_up']:.2f}, "
                f"k_down={params['k_down']:.2f}, max_bars={params['max_bars']}"
            )

            df = apply_triple_barrier(
                df=df,
                horizon=horizon,
                k_up=params["k_up"],
                k_down=params["k_down"],
                max_bars=params["max_bars"],
                atr_column="atr_14",
            )

        if not dry_run:
            output_path = labels_dir / f"{symbol}_labels_init.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"  Saved to {output_path}")

        labeled_data[symbol] = df

    logger.info("\nStage 4 complete.")
    return labeled_data


def run_stage5_ga_optimize(
    symbols: List[str],
    horizons: List[int],
    population_size: int = 50,
    generations: int = 30,
    dry_run: bool = False,
) -> Dict[str, Dict[int, Dict]]:
    """
    Run Stage 5: GA Optimization.

    Returns:
        Dictionary mapping symbol -> horizon -> best_params
    """
    from stage5_ga_optimize import process_symbol_ga

    logger.info("=" * 70)
    logger.info("STAGE 5: GA OPTIMIZATION")
    logger.info("=" * 70)

    all_results = {}

    for symbol in symbols:
        try:
            results = process_symbol_ga(
                symbol=symbol,
                horizons=horizons,
                population_size=population_size,
                generations=generations,
            )
            all_results[symbol] = results
        except Exception as e:
            logger.error(f"Error optimizing {symbol}: {e}", exc_info=True)

    logger.info("\nStage 5 complete.")
    return all_results


def run_stage6_final_labels(
    symbols: List[str],
    horizons: List[int],
    barrier_params: Dict[int, Dict],
    dry_run: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run Stage 6: Apply Final Labels with Quality Scores.

    Returns:
        Dictionary mapping symbol -> final labeled DataFrame
    """
    from stage6_final_labels import apply_optimized_labels

    logger.info("=" * 70)
    logger.info("STAGE 6: FINAL LABELING WITH QUALITY SCORES")
    logger.info("=" * 70)

    features_dir = PROJECT_ROOT / "data" / "features"
    final_dir = PROJECT_ROOT / "data" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_data = {}

    for symbol in symbols:
        logger.info(f"\nProcessing {symbol}...")

        input_path = features_dir / f"{symbol}_5m_features.parquet"
        if not input_path.exists():
            logger.warning(f"Features file not found: {input_path}")
            continue

        df = pd.read_parquet(input_path)
        logger.info(f"  Loaded {len(df):,} rows")

        # Apply optimized labeling for each horizon
        for horizon in horizons:
            params = barrier_params[horizon]
            df = apply_optimized_labels(
                df=df,
                horizon=horizon,
                best_params=params,
                atr_column="atr_14",
            )

        if not dry_run:
            output_path = final_dir / f"{symbol}_final_labeled.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"  Saved to {output_path}")

        final_data[symbol] = df

    # Combine all symbols
    if final_data and not dry_run:
        combined_df = pd.concat(final_data.values(), ignore_index=True)
        combined_df = combined_df.sort_values("datetime").reset_index(drop=True)
        combined_path = final_dir / "combined_final_labeled.parquet"
        combined_df.to_parquet(combined_path, index=False)
        logger.info(f"\nSaved combined data ({len(combined_df):,} rows) to {combined_path}")

    logger.info("\nStage 6 complete.")
    return final_data


def run_stage7_splits(
    dry_run: bool = False,
) -> Dict:
    """
    Run Stage 7: Time-Based Splits.

    Returns:
        Split metadata dictionary
    """
    from stage7_splits import create_splits

    logger.info("=" * 70)
    logger.info("STAGE 7: TIME-BASED SPLITTING")
    logger.info("=" * 70)

    data_path = PROJECT_ROOT / "data" / "final" / "combined_final_labeled.parquet"
    splits_dir = PROJECT_ROOT / "data" / "splits"

    if not data_path.exists():
        logger.error(f"Combined data file not found: {data_path}")
        return {}

    if dry_run:
        logger.info("Dry run - skipping split creation")
        return {"dry_run": True}

    metadata = create_splits(
        data_path=data_path,
        output_dir=splits_dir,
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        purge_bars=PURGE_BARS,
        embargo_bars=EMBARGO_BARS,
    )

    logger.info("\nStage 7 complete.")
    return metadata


def run_stage8_validation(
    horizons: List[int],
    dry_run: bool = False,
) -> Dict:
    """
    Run Stage 8: Validation.

    Returns:
        Validation summary dictionary
    """
    from stage8_validate import validate_data

    logger.info("=" * 70)
    logger.info("STAGE 8: VALIDATION")
    logger.info("=" * 70)

    data_path = PROJECT_ROOT / "data" / "final" / "combined_final_labeled.parquet"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.error(f"Combined data file not found: {data_path}")
        return {"status": "FAILED", "error": "Data file not found"}

    output_path = results_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    if dry_run:
        logger.info("Dry run - skipping validation")
        return {"dry_run": True}

    summary = validate_data(
        data_path=data_path,
        output_path=output_path,
        horizons=horizons,
    )

    logger.info("\nStage 8 complete.")
    return summary


def compare_distributions(
    before_path: Optional[Path],
    after_df: pd.DataFrame,
    horizons: List[int],
) -> Dict:
    """
    Compare label distributions before and after re-labeling.

    Returns:
        Comparison dictionary
    """
    comparison = {}

    if before_path and before_path.exists():
        before_df = pd.read_parquet(before_path)
        logger.info("\nLabel Distribution Comparison:")
        logger.info("-" * 60)

        for horizon in horizons:
            label_col = f"label_h{horizon}"
            if label_col not in before_df.columns or label_col not in after_df.columns:
                continue

            before_counts = before_df[label_col].value_counts().sort_index()
            after_counts = after_df[label_col].value_counts().sort_index()

            total_before = len(before_df)
            total_after = len(after_df)

            logger.info(f"\nHorizon {horizon}:")
            logger.info(f"  {'Class':<12} {'Before':>12} {'After':>12} {'Change':>12}")
            logger.info(f"  {'-'*48}")

            horizon_comparison = {}
            for label_val in [-1, 0, 1]:
                label_name = {-1: "Short", 0: "Neutral", 1: "Long"}[label_val]
                before_count = before_counts.get(label_val, 0)
                after_count = after_counts.get(label_val, 0)
                before_pct = before_count / total_before * 100
                after_pct = after_count / total_after * 100
                change = after_pct - before_pct

                logger.info(
                    f"  {label_name:<12} {before_pct:>11.1f}% {after_pct:>11.1f}% {change:>+11.1f}%"
                )

                horizon_comparison[label_name.lower()] = {
                    "before_pct": before_pct,
                    "after_pct": after_pct,
                    "change_pct": change,
                }

            comparison[horizon] = horizon_comparison
    else:
        logger.info("\nNo previous data found for comparison")

    return comparison


def generate_summary_report(
    result: LabelingResult,
    comparison: Dict,
    run_timestamp: str,
) -> str:
    """Generate a markdown summary report."""
    report = f"""# Re-Labeling Summary Report

**Run Timestamp:** {run_timestamp}
**Status:** {"PASSED" if result.validation_passed else "FAILED"}

## Optimized Parameters Used

| Horizon | k_up | k_down | max_bars |
|---------|------|--------|----------|
"""

    for horizon, params in OPTIMIZED_BARRIER_PARAMS.items():
        report += f"| {horizon} | {params['k_up']} | {params['k_down']} | {params['max_bars']} |\n"

    report += "\n## Label Distributions\n\n"

    for horizon, dist in result.distributions.items():
        report += f"""### Horizon {horizon}

| Metric | Value |
|--------|-------|
| Long | {dist['long_pct']*100:.1f}% ({dist['long_count']:,}) |
| Short | {dist['short_pct']*100:.1f}% ({dist['short_count']:,}) |
| Neutral | {dist['neutral_pct']*100:.1f}% ({dist['neutral_count']:,}) |
| Signal Rate | {dist['signal_rate']*100:.1f}% |
| Long/Short Ratio | {dist['long_ratio']:.2f} |

"""

    if comparison:
        report += "## Before/After Comparison\n\n"
        for horizon, comp in comparison.items():
            report += f"### Horizon {horizon}\n\n"
            report += "| Class | Before | After | Change |\n"
            report += "|-------|--------|-------|--------|\n"
            for class_name, values in comp.items():
                report += (
                    f"| {class_name.title()} | {values['before_pct']:.1f}% | "
                    f"{values['after_pct']:.1f}% | {values['change_pct']:+.1f}% |\n"
                )
            report += "\n"

    if result.issues:
        report += "## Validation Issues\n\n"
        for issue in result.issues:
            report += f"- {issue}\n"
        report += "\n"

    if result.warnings:
        report += "## Warnings\n\n"
        for warning in result.warnings:
            report += f"- {warning}\n"
        report += "\n"

    report += "---\n*Generated by rerun_labeling.py*\n"

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Re-run Phase 1 labeling stages with optimized parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/rerun_labeling.py --symbols MES,MGC
    python scripts/rerun_labeling.py --skip-ga
    python scripts/rerun_labeling.py --symbols MGC --skip-ga --dry-run
        """,
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated list of symbols (default: {','.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--skip-ga",
        action="store_true",
        help="Skip GA optimization, use preset optimal parameters",
    )
    parser.add_argument(
        "--ga-population",
        type=int,
        default=50,
        help="GA population size (default: 50)",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=30,
        help="GA generations (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving any files",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the validation stage",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=[4, 5, 6, 7, 8],
        default=4,
        help="Start from a specific stage (default: 4)",
    )
    parser.add_argument(
        "--custom-params",
        type=str,
        help="Path to JSON file with custom barrier parameters",
    )

    args = parser.parse_args()

    # Parse arguments
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load barrier parameters
    if args.custom_params:
        with open(args.custom_params, "r") as f:
            barrier_params = json.load(f)
        # Convert string keys to int
        barrier_params = {int(k): v for k, v in barrier_params.items()}
        logger.info(f"Loaded custom parameters from {args.custom_params}")
    else:
        barrier_params = OPTIMIZED_BARRIER_PARAMS.copy()

    logger.info("=" * 70)
    logger.info("PHASE 1 RE-LABELING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {run_timestamp}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Skip GA: {args.skip_ga}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"Starting from Stage: {args.from_stage}")
    logger.info("")
    logger.info("Barrier Parameters:")
    for h, params in barrier_params.items():
        logger.info(f"  h{h}: k_up={params['k_up']}, k_down={params['k_down']}, max_bars={params['max_bars']}")
    logger.info("")

    # Ensure logs directory exists
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # Store path to existing data for comparison
    existing_data_path = PROJECT_ROOT / "data" / "final" / "combined_final_labeled.parquet"

    result = LabelingResult()

    try:
        # Stage 4: Labeling
        if args.from_stage <= 4:
            run_stage4_labeling(
                symbols=symbols,
                horizons=HORIZONS,
                barrier_params=barrier_params,
                dry_run=args.dry_run,
            )

        # Stage 5: GA Optimization (optional)
        if args.from_stage <= 5 and not args.skip_ga:
            ga_results = run_stage5_ga_optimize(
                symbols=symbols,
                horizons=HORIZONS,
                population_size=args.ga_population,
                generations=args.ga_generations,
                dry_run=args.dry_run,
            )
            # Update barrier params with GA results
            for symbol, symbol_results in ga_results.items():
                for horizon, res in symbol_results.items():
                    barrier_params[horizon] = {
                        "k_up": res["best_k_up"],
                        "k_down": res["best_k_down"],
                        "max_bars": res["best_max_bars"],
                    }
            logger.info("\nUpdated parameters from GA optimization:")
            for h, params in barrier_params.items():
                logger.info(f"  h{h}: k_up={params['k_up']:.3f}, k_down={params['k_down']:.3f}, max_bars={params['max_bars']}")

        # Stage 6: Final Labels
        if args.from_stage <= 6:
            final_data = run_stage6_final_labels(
                symbols=symbols,
                horizons=HORIZONS,
                barrier_params=barrier_params,
                dry_run=args.dry_run,
            )

            # Combine for validation
            if final_data:
                combined_df = pd.concat(final_data.values(), ignore_index=True)

        # Stage 7: Splits
        if args.from_stage <= 7:
            split_metadata = run_stage7_splits(dry_run=args.dry_run)

        # Stage 8: Validation
        if args.from_stage <= 8 and not args.skip_validation:
            validation_summary = run_stage8_validation(
                horizons=HORIZONS,
                dry_run=args.dry_run,
            )

        # Final validation of label distributions
        if not args.dry_run and "combined_df" in locals():
            logger.info("\n" + "=" * 70)
            logger.info("FINAL LABEL DISTRIBUTION VALIDATION")
            logger.info("=" * 70)

            passed, issues, distributions = validate_label_distribution(
                combined_df, HORIZONS
            )
            result.validation_passed = passed
            result.issues = issues
            result.distributions = distributions

            for horizon, dist in distributions.items():
                logger.info(f"\nHorizon {horizon}:")
                logger.info(f"  Long:     {dist['long_pct']*100:5.1f}% ({dist['long_count']:,})")
                logger.info(f"  Short:    {dist['short_pct']*100:5.1f}% ({dist['short_count']:,})")
                logger.info(f"  Neutral:  {dist['neutral_pct']*100:5.1f}% ({dist['neutral_count']:,})")
                logger.info(f"  Signal Rate: {dist['signal_rate']*100:.1f}%")
                logger.info(f"  Long/Short Ratio: {dist['long_ratio']:.2f}")

            if passed:
                logger.info("\n[PASSED] All validation criteria met!")
            else:
                logger.warning("\n[FAILED] Validation issues found:")
                for issue in issues:
                    logger.warning(f"  - {issue}")

            # Compare with previous distributions
            comparison = compare_distributions(existing_data_path, combined_df, HORIZONS)

            # Generate summary report
            report = generate_summary_report(result, comparison, run_timestamp)
            report_path = PROJECT_ROOT / "results" / f"relabeling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"\nSummary report saved to: {report_path}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    if not args.dry_run:
        logger.info("\nOutput files:")
        logger.info(f"  - Final labeled data: {PROJECT_ROOT / 'data' / 'final'}")
        logger.info(f"  - Splits: {PROJECT_ROOT / 'data' / 'splits'}")
        logger.info(f"  - Validation reports: {PROJECT_ROOT / 'results'}")
        logger.info(f"  - Logs: {PROJECT_ROOT / 'logs'}")

    sys.exit(0 if result.validation_passed else 1)


if __name__ == "__main__":
    main()
