#!/usr/bin/env python3
"""
Meta-labeling pipeline (Lopez de Prado 2018).

Step 1: Train primary model (predicts side: long/short/neutral)
Step 2: Train meta-model (predicts bet size: 0.0-1.0 based on confidence)

This implements Lopez de Prado's meta-labeling methodology where:
- Primary model: What position to take (direction)
- Meta-model: How much to bet (sizing based on confidence)

Usage:
    python scripts/train_meta_labeling.py \
        --primary-model xgboost \
        --meta-model logistic \
        --horizon 20

    python scripts/train_meta_labeling.py \
        --primary-model catboost \
        --meta-model ridge \
        --horizon 20 \
        --meta-threshold 0.3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Trainer, TrainerConfig
from src.phase1.stages.datasets import TimeSeriesDataContainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_bet_size_labels(
    y_actual: np.ndarray,
    y_pred_primary: np.ndarray,
    quality_metrics: np.ndarray,
) -> np.ndarray:
    """
    Compute meta-labels (bet sizes) from primary predictions.

    The meta-label represents how much to bet on the primary model's
    prediction. High values mean high confidence, low values mean
    low confidence or should skip trade.

    Args:
        y_actual: True labels {-1, 0, 1}
        y_pred_primary: Primary model predictions {-1, 0, 1}
        quality_metrics: Quality scores for each sample [0.0, 1.0]

    Returns:
        Bet sizes [0.0, 1.0] where:
        - 0.0 = skip trade (low confidence)
        - 1.0 = full bet (high confidence)
    """
    correct = (y_actual == y_pred_primary).astype(float)
    bet_size = correct * quality_metrics
    return np.clip(bet_size, 0.0, 1.0)


def train_primary_model(
    container: TimeSeriesDataContainer,
    config: TrainerConfig,
) -> tuple[Trainer, dict]:
    """
    Train primary model (predicts side).

    Args:
        container: TimeSeriesDataContainer with data
        config: TrainerConfig for primary model

    Returns:
        Tuple of (trained Trainer, results dict)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: TRAIN PRIMARY MODEL (Side Prediction)")
    logger.info("=" * 60)

    trainer = Trainer(config)
    results = trainer.run(container)

    logger.info("\nPrimary Model Results:")
    logger.info(f"  Val F1: {results['evaluation_metrics']['val_f1']:.4f}")
    logger.info(f"  Val Accuracy: {results['evaluation_metrics']['val_accuracy']:.4f}")

    return trainer, results


def train_meta_model(
    container: TimeSeriesDataContainer,
    primary_trainer: Trainer,
    config_meta: TrainerConfig,
    quality_metrics: dict,
) -> tuple[Trainer, dict]:
    """
    Train meta-model (predicts bet size).

    Args:
        container: TimeSeriesDataContainer with data
        primary_trainer: Trained primary model
        config_meta: TrainerConfig for meta-model
        quality_metrics: Quality scores by split

    Returns:
        Tuple of (trained meta-Trainer, results dict)
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: TRAIN META-MODEL (Bet Size)")
    logger.info("=" * 60)

    primary_preds_train = primary_trainer.model.predict(container.X_train)
    primary_preds_val = primary_trainer.model.predict(container.X_val)

    y_meta_train = compute_bet_size_labels(
        container.y_train,
        primary_preds_train,
        quality_metrics["train"],
    )
    y_meta_val = compute_bet_size_labels(
        container.y_val,
        primary_preds_val,
        quality_metrics["val"],
    )

    logger.info("Meta-label statistics (train):")
    logger.info(f"  Mean: {y_meta_train.mean():.3f}")
    logger.info(f"  Std:  {y_meta_train.std():.3f}")
    logger.info(f"  Min:  {y_meta_train.min():.3f}")
    logger.info(f"  Max:  {y_meta_train.max():.3f}")

    meta_container = TimeSeriesDataContainer(
        X_train=container.X_train,
        y_train=y_meta_train,
        X_val=container.X_val,
        y_val=y_meta_val,
        X_test=container.X_test,
        y_test=container.y_test,
        sample_weights=container.sample_weights,
    )

    trainer_meta = Trainer(config_meta)
    results_meta = trainer_meta.run(meta_container)

    logger.info("\nMeta-Model Results:")
    logger.info(f"  Val MAE: {results_meta['evaluation_metrics'].get('val_mae', 'N/A')}")

    return trainer_meta, results_meta


def evaluate_meta_labeling(
    container: TimeSeriesDataContainer,
    primary_trainer: Trainer,
    meta_trainer: Trainer,
    quality_metrics: dict,
    meta_threshold: float = 0.3,
) -> dict:
    """
    Evaluate combined primary + meta system on test set.

    Args:
        container: TimeSeriesDataContainer with test data
        primary_trainer: Trained primary model
        meta_trainer: Trained meta-model
        quality_metrics: Quality scores by split
        meta_threshold: Minimum bet size to take trade

    Returns:
        Evaluation results dict
    """
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION: Primary + Meta System")
    logger.info("=" * 60)

    primary_preds = primary_trainer.model.predict(container.X_test)
    meta_preds = meta_trainer.model.predict(container.X_test)

    if hasattr(meta_preds, "flatten"):
        meta_preds = meta_preds.flatten()

    y_actual = container.y_test

    correct_primary = (primary_preds == y_actual).mean()

    trades_taken = meta_preds >= meta_threshold

    if trades_taken.sum() > 0:
        correct_meta = (primary_preds[trades_taken] == y_actual[trades_taken]).mean()
        trade_fraction = trades_taken.mean()
    else:
        correct_meta = 0.0
        trade_fraction = 0.0

    logger.info("\nTest Set Results:")
    logger.info("  Primary Only:")
    logger.info(f"    Accuracy: {correct_primary:.4f}")
    logger.info("    Trades: 100%")
    logger.info(f"\n  With Meta-Labeling (threshold={meta_threshold}):")
    logger.info(f"    Accuracy: {correct_meta:.4f}")
    logger.info(f"    Trades Taken: {trade_fraction*100:.1f}%")
    logger.info(f"    Trades Skipped: {(1-trade_fraction)*100:.1f}%")

    if trade_fraction > 0:
        improvement = correct_meta - correct_primary
        logger.info(f"    Accuracy Improvement: {improvement:+.4f}")

    return {
        "primary_accuracy": float(correct_primary),
        "meta_accuracy": float(correct_meta),
        "trade_fraction": float(trade_fraction),
        "meta_threshold": meta_threshold,
        "improvement": float(correct_meta - correct_primary) if trade_fraction > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train meta-labeling system (Lopez de Prado 2018)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # XGBoost primary + Logistic meta
  python scripts/train_meta_labeling.py \\
      --primary-model xgboost \\
      --meta-model logistic \\
      --horizon 20

  # CatBoost primary + Ridge meta with custom threshold
  python scripts/train_meta_labeling.py \\
      --primary-model catboost \\
      --meta-model ridge \\
      --horizon 20 \\
      --meta-threshold 0.4

  # LSTM primary + MLP meta (for sequence models)
  python scripts/train_meta_labeling.py \\
      --primary-model lstm \\
      --meta-model logistic \\
      --horizon 20 \\
      --seq-len 60
""",
    )

    parser.add_argument(
        "--primary-model",
        required=True,
        help="Primary model (predicts side): xgboost, catboost, lstm, etc.",
    )
    parser.add_argument(
        "--meta-model", required=True, help="Meta-model (predicts bet size): logistic, ridge, etc."
    )
    parser.add_argument("--horizon", type=int, required=True, help="Label horizon (5, 10, 15, 20)")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length for neural models")
    parser.add_argument(
        "--meta-threshold", type=float, default=0.3, help="Minimum bet size to take trade (0.0-1.0)"
    )
    parser.add_argument(
        "--output-dir", default="experiments/meta_labeling", help="Output directory"
    )
    parser.add_argument(
        "--data-dir", default="data/splits/scaled", help="Directory containing scaled data"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"{args.primary_model}_{args.meta_model}_h{args.horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {args.data_dir}...")
    container = TimeSeriesDataContainer.from_parquet_dir(args.data_dir, horizon=args.horizon)

    logger.info("Data loaded:")
    logger.info(f"  Train: {container.X_train.shape}")
    logger.info(f"  Val: {container.X_val.shape}")
    logger.info(f"  Test: {container.X_test.shape}")

    quality_metrics = {
        "train": (
            container.sample_weights
            if container.sample_weights is not None
            else np.ones(len(container.X_train))
        ),
        "val": np.ones(len(container.X_val)),
        "test": np.ones(len(container.X_test)),
    }

    config_primary = TrainerConfig(
        model_name=args.primary_model,
        horizon=args.horizon,
        sequence_length=args.seq_len,
    )

    primary_trainer, primary_results = train_primary_model(container, config_primary)

    primary_model_path = output_dir / "primary_model.pkl"
    primary_trainer.save(primary_model_path)
    logger.info(f"Primary model saved to: {primary_model_path}")

    config_meta = TrainerConfig(
        model_name=args.meta_model,
        horizon=args.horizon,
        sequence_length=args.seq_len,
    )

    meta_trainer, meta_results = train_meta_model(
        container, primary_trainer, config_meta, quality_metrics
    )

    meta_model_path = output_dir / "meta_model.pkl"
    meta_trainer.save(meta_model_path)
    logger.info(f"Meta-model saved to: {meta_model_path}")

    eval_results = evaluate_meta_labeling(
        container, primary_trainer, meta_trainer, quality_metrics, args.meta_threshold
    )

    summary = {
        "primary": {
            "model": args.primary_model,
            "val_f1": primary_results["evaluation_metrics"]["val_f1"],
            "val_accuracy": primary_results["evaluation_metrics"]["val_accuracy"],
        },
        "meta": {
            "model": args.meta_model,
        },
        "combined": eval_results,
        "config": {
            "horizon": args.horizon,
            "meta_threshold": args.meta_threshold,
        },
    }

    summary_path = output_dir / "meta_labeling_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("META-LABELING TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nModels saved to: {output_dir}")
    logger.info(f"Results saved to: {summary_path}")

    if eval_results["improvement"] > 0:
        logger.info(f"\n✅ Meta-labeling improved accuracy by {eval_results['improvement']:+.2%}")
    else:
        logger.info("\n⚠️ Meta-labeling did not improve accuracy (try adjusting threshold)")


if __name__ == "__main__":
    main()
