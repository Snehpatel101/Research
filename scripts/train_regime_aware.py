#!/usr/bin/env python3
"""
Train regime-specific models.

Detects market regimes and trains separate models for each regime.
This improves performance by allowing different strategies for different
market conditions (trending vs mean-reverting, high-vol vs low-vol).

Usage:
    python scripts/train_regime_aware.py \
        --model xgboost \
        --horizon 20 \
        --regimes trending_low_vol,mean_reverting_high_vol

    python scripts/train_regime_aware.py \
        --model lstm \
        --horizon 20 \
        --regimes all \
        --seq-len 60
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TrainerConfig, Trainer
from src.phase1.stages.datasets import TimeSeriesDataContainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Detect market regimes from features.

    Combines trend and volatility to classify 4 regimes:
    - trending_low_vol: Strong trend, normal volatility
    - trending_high_vol: Strong trend, elevated volatility
    - mean_reverting_low_vol: Weak trend, normal volatility
    - mean_reverting_high_vol: Weak trend, elevated volatility

    Args:
        df: DataFrame with features (must include adx_14 and atr_14)

    Returns:
        Series with regime labels
    """
    adx = df.get("adx_14", df.get("adx", None))
    atr = df["atr_14"]

    atr_mean = atr.rolling(100, min_periods=20).mean()

    if adx is not None:
        trending = adx > 25
    else:
        close_returns = df["close"].pct_change()
        abs_autocorr = close_returns.rolling(20).apply(lambda x: abs(x.autocorr(lag=1)), raw=False)
        trending = abs_autocorr > 0.3

    high_vol = atr > 1.5 * atr_mean

    regime = pd.Series("unknown", index=df.index)
    regime[trending & ~high_vol] = "trending_low_vol"
    regime[trending & high_vol] = "trending_high_vol"
    regime[~trending & ~high_vol] = "mean_reverting_low_vol"
    regime[~trending & high_vol] = "mean_reverting_high_vol"

    return regime


def train_regime_specific_model(
    container: TimeSeriesDataContainer,
    regime_label: str,
    config: TrainerConfig,
    output_dir: Path,
) -> dict | None:
    """
    Train model on subset of data matching specific regime.

    Args:
        container: TimeSeriesDataContainer with full dataset
        regime_label: Regime to train on
        config: TrainerConfig for model
        output_dir: Directory to save regime-specific model

    Returns:
        Training results dict or None if insufficient data
    """
    regime = detect_regimes(container.X_train)

    regime_mask = regime == regime_label

    min_samples = 100
    if regime_mask.sum() < min_samples:
        logger.warning(
            f"Skipping {regime_label}: insufficient data "
            f"({regime_mask.sum()} < {min_samples} samples)"
        )
        return None

    logger.info(f"\n{'='*60}")
    logger.info(f"Training on regime: {regime_label}")
    logger.info(f"  Samples: {regime_mask.sum()} ({regime_mask.mean()*100:.1f}%)")
    logger.info("=" * 60)

    regime_container = TimeSeriesDataContainer(
        X_train=container.X_train[regime_mask],
        y_train=container.y_train[regime_mask],
        X_val=container.X_val,
        y_val=container.y_val,
        X_test=container.X_test,
        y_test=container.y_test,
        sample_weights=(
            container.sample_weights[regime_mask] if container.sample_weights is not None else None
        ),
    )

    trainer = Trainer(config)
    results = trainer.run(regime_container)

    model_path = output_dir / f"model_{regime_label}.pkl"
    trainer.save(model_path)

    logger.info(f"\nRegime {regime_label} Results:")
    logger.info(f"  Val F1: {results['evaluation_metrics']['val_f1']:.4f}")
    logger.info(f"  Val Accuracy: {results['evaluation_metrics']['val_accuracy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train regime-specific models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost on trending regimes only
  python scripts/train_regime_aware.py --model xgboost --horizon 20 \\
      --regimes trending_low_vol,trending_high_vol

  # Train LSTM on all 4 regimes
  python scripts/train_regime_aware.py --model lstm --horizon 20 \\
      --regimes all --seq-len 60

  # Train CatBoost on mean-reverting regimes
  python scripts/train_regime_aware.py --model catboost --horizon 20 \\
      --regimes mean_reverting_low_vol,mean_reverting_high_vol
""",
    )

    parser.add_argument("--model", required=True, help="Model name (xgboost, lstm, catboost, etc.)")
    parser.add_argument("--horizon", type=int, required=True, help="Label horizon (5, 10, 15, 20)")
    parser.add_argument(
        "--regimes", default="all", help='Comma-separated regime list or "all" for all 4 regimes'
    )
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length for neural models")
    parser.add_argument(
        "--output-dir",
        default="experiments/regime_aware",
        help="Output directory for regime-specific models",
    )
    parser.add_argument(
        "--data-dir", default="data/splits/scaled", help="Directory containing scaled data"
    )

    args = parser.parse_args()

    all_regimes = [
        "trending_low_vol",
        "trending_high_vol",
        "mean_reverting_low_vol",
        "mean_reverting_high_vol",
    ]

    if args.regimes == "all":
        regimes = all_regimes
    else:
        regimes = args.regimes.split(",")

        invalid = [r for r in regimes if r not in all_regimes]
        if invalid:
            logger.error(f"Invalid regimes: {invalid}")
            logger.error(f"Valid regimes: {all_regimes}")
            sys.exit(1)

    output_dir = Path(args.output_dir) / f"{args.model}_h{args.horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {args.data_dir}...")
    container = TimeSeriesDataContainer.from_parquet_dir(args.data_dir, horizon=args.horizon)

    logger.info(f"Data loaded:")
    logger.info(f"  Train: {container.X_train.shape}")
    logger.info(f"  Val: {container.X_val.shape}")
    logger.info(f"  Test: {container.X_test.shape}")

    config = TrainerConfig(
        model_name=args.model,
        horizon=args.horizon,
        sequence_length=args.seq_len,
    )

    results_by_regime = {}

    for regime in regimes:
        results = train_regime_specific_model(container, regime, config, output_dir)
        if results is not None:
            results_by_regime[regime] = results

    logger.info(f"\n{'='*60}")
    logger.info("REGIME-SPECIFIC TRAINING COMPLETE")
    logger.info("=" * 60)

    if results_by_regime:
        logger.info("\nResults Summary:")
        for regime, results in results_by_regime.items():
            val_f1 = results["evaluation_metrics"]["val_f1"]
            val_acc = results["evaluation_metrics"]["val_accuracy"]
            logger.info(f"  {regime:30s}: F1={val_f1:.4f}, Acc={val_acc:.4f}")

        summary = {
            "model": args.model,
            "horizon": args.horizon,
            "regimes_trained": list(results_by_regime.keys()),
            "results": {
                regime: {
                    "val_f1": res["evaluation_metrics"]["val_f1"],
                    "val_accuracy": res["evaluation_metrics"]["val_accuracy"],
                }
                for regime, res in results_by_regime.items()
            },
        }

        summary_path = output_dir / "regime_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nModels saved to: {output_dir}")
        logger.info(f"Summary saved to: {summary_path}")
    else:
        logger.error("No models trained (insufficient data for all regimes)")
        sys.exit(1)


if __name__ == "__main__":
    main()
