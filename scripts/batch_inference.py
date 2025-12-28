#!/usr/bin/env python3
"""
Batch Inference CLI.

Run batch inference on a dataset using a bundled model.

Usage:
    # Basic batch inference
    python scripts/batch_inference.py \
        --bundle ./bundles/xgb_h20 \
        --input data/test.parquet \
        --output predictions.parquet

    # With custom batch size
    python scripts/batch_inference.py \
        --bundle ./bundles/xgb_h20 \
        --input data/test.parquet \
        --output predictions.parquet \
        --batch-size 5000

    # Ensemble inference
    python scripts/batch_inference.py \
        --bundles ./bundles/xgb_h20,./bundles/lgbm_h20 \
        --input data/test.parquet \
        --output ensemble_predictions.parquet \
        --ensemble
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    BatchPredictor,
    BatchProgress,
    InferencePipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference using a bundled model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model inference
  python scripts/batch_inference.py --bundle ./bundles/xgb_h20 --input data.parquet --output predictions.parquet

  # Ensemble inference
  python scripts/batch_inference.py --bundles ./bundles/xgb_h20,./bundles/lgbm_h20 --input data.parquet --output predictions.parquet --ensemble
        """,
    )

    # Model specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--bundle",
        type=Path,
        help="Path to single model bundle",
    )
    model_group.add_argument(
        "--bundles",
        type=str,
        help="Comma-separated paths to multiple bundles",
    )

    # Data paths
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input parquet file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path for output predictions parquet",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Samples per batch (default: 10000)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable probability calibration",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble voting (requires multiple bundles)",
    )
    parser.add_argument(
        "--voting",
        type=str,
        default="soft_vote",
        choices=["soft_vote", "hard_vote"],
        help="Ensemble voting method (default: soft_vote)",
    )

    # Output options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def progress_bar(progress: BatchProgress) -> None:
    """Display progress bar."""
    bar_width = 40
    filled = int(bar_width * progress.progress_pct / 100)
    bar = "=" * filled + "-" * (bar_width - filled)

    eta_str = f"{progress.eta_seconds:.1f}s" if progress.eta_seconds < 3600 else "N/A"

    print(
        f"\r[{bar}] {progress.progress_pct:.1f}% "
        f"({progress.processed_samples}/{progress.total_samples}) "
        f"ETA: {eta_str}",
        end="",
        flush=True,
    )


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load bundles
    if args.bundle:
        bundle_paths = [args.bundle]
    else:
        bundle_paths = [Path(p.strip()) for p in args.bundles.split(",")]

    for path in bundle_paths:
        if not path.exists():
            logger.error(f"Bundle not found: {path}")
            return 1

    logger.info(f"Loading {len(bundle_paths)} bundle(s)...")

    # Create predictor
    if len(bundle_paths) == 1:
        predictor = BatchPredictor.from_bundle(
            bundle_paths[0], batch_size=args.batch_size
        )
    else:
        predictor = BatchPredictor.from_bundles(
            bundle_paths, batch_size=args.batch_size
        )

    # Log model info
    for info in predictor.pipeline.get_model_info():
        logger.info(f"  Model: {info['name']} (H{info['horizon']}, {info['features']} features)")

    # Run inference
    logger.info(f"Processing {args.input}...")

    if args.ensemble and len(bundle_paths) > 1:
        # Ensemble mode
        import pandas as pd

        data = pd.read_parquet(args.input)
        result = predictor.pipeline.predict_ensemble(
            data,
            method=args.voting,
            calibrate=not args.no_calibrate,
        )

        # Save ensemble predictions
        output_df = result.to_dataframe()
        output_df.to_parquet(args.output, index=False)

        logger.info(
            f"Saved ensemble predictions to {args.output} "
            f"({len(output_df)} samples, {result.inference_time_ms:.1f}ms)"
        )
    else:
        # Standard batch inference
        result = predictor.predict_batch(
            args.input,
            output_path=args.output,
            progress_callback=None if args.quiet else progress_bar,
            calibrate=not args.no_calibrate,
        )

        if not args.quiet:
            print()  # Newline after progress bar

        logger.info(
            f"Completed: {result.n_samples} predictions, "
            f"{result.samples_per_second:.0f} samples/sec, "
            f"{len(result.errors)} errors"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
