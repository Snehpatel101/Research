"""
Meta-labeling training mode.

Implements Lopez de Prado's meta-labeling methodology (2018) where:
- Primary model: Predicts trade direction (side)
- Meta-model: Predicts bet size based on confidence (sizing)

This two-stage approach improves trading by filtering low-confidence trades
and sizing positions according to model confidence.

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer

from src.models import Trainer, TrainerConfig

from ..config import ExperimentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class MetaLabelingConfig:
    """
    Configuration for meta-labeling training.

    Attributes:
        primary_model: Model for side prediction (direction)
        meta_model: Model for bet size prediction
        meta_threshold: Minimum bet size to take trade (0.0-1.0)
        use_quality_weights: Use sample quality as part of meta-labels
        calibrate_probabilities: Apply probability calibration to primary
        sequence_length: Sequence length for neural models
    """

    primary_model: str = "xgboost"
    meta_model: str = "logistic"
    meta_threshold: float = 0.3
    use_quality_weights: bool = True
    calibrate_probabilities: bool = False
    sequence_length: int = 60


@dataclass
class MetaLabelingResult:
    """
    Results from meta-labeling training.

    Attributes:
        primary_model: Name of primary model
        meta_model: Name of meta model
        horizon: Label horizon
        primary_metrics: Primary model validation metrics
        meta_metrics: Meta model validation metrics
        combined_metrics: Combined system evaluation metrics
        primary_model_path: Path to saved primary model
        meta_model_path: Path to saved meta model
        total_time: Total training time
    """

    primary_model: str
    meta_model: str
    horizon: int
    primary_metrics: dict[str, float] = field(default_factory=dict)
    meta_metrics: dict[str, float] = field(default_factory=dict)
    combined_metrics: dict[str, float] = field(default_factory=dict)
    primary_model_path: Path | None = None
    meta_model_path: Path | None = None
    total_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "primary_model": self.primary_model,
            "meta_model": self.meta_model,
            "horizon": self.horizon,
            "primary_metrics": self.primary_metrics,
            "meta_metrics": self.meta_metrics,
            "combined_metrics": self.combined_metrics,
            "primary_model_path": str(self.primary_model_path) if self.primary_model_path else None,
            "meta_model_path": str(self.meta_model_path) if self.meta_model_path else None,
            "total_time": self.total_time,
        }


# =============================================================================
# META-LABEL GENERATION
# =============================================================================


def compute_bet_size_labels(
    y_actual: np.ndarray,
    y_pred_primary: np.ndarray,
    quality_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute meta-labels (bet sizes) from primary predictions.

    The meta-label represents how much to bet on the primary model's
    prediction. High values mean high confidence, low values mean
    low confidence or should skip trade.

    Meta-label formula:
        bet_size = correct_prediction * quality_weight

    Where:
        - correct_prediction: 1 if primary matches actual, 0 otherwise
        - quality_weight: Sample quality (default 1.0)

    Args:
        y_actual: True labels (e.g., {-1, 0, 1} or {0, 1, 2})
        y_pred_primary: Primary model predictions
        quality_weights: Optional quality scores for each sample [0.0, 1.0]

    Returns:
        Bet sizes [0.0, 1.0] where:
        - 0.0 = skip trade (low confidence / wrong prediction)
        - 1.0 = full bet (high confidence / correct prediction)
    """
    # Binary correct/incorrect
    correct = (y_actual == y_pred_primary).astype(float)

    # Apply quality weights if provided
    if quality_weights is not None:
        bet_size = correct * quality_weights
    else:
        bet_size = correct

    return np.clip(bet_size, 0.0, 1.0)


def compute_directional_meta_labels(
    y_actual: np.ndarray,
    y_pred_primary: np.ndarray,
    primary_proba: np.ndarray | None = None,
    quality_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute directional meta-labels using prediction confidence.

    Alternative approach that incorporates prediction probability
    to create more granular bet sizes.

    Args:
        y_actual: True labels
        y_pred_primary: Primary predictions
        primary_proba: Class probabilities from primary model (optional)
        quality_weights: Sample quality weights (optional)

    Returns:
        Bet sizes [0.0, 1.0]
    """
    # Start with correct/incorrect
    correct = (y_actual == y_pred_primary).astype(float)

    # If we have probabilities, use max probability as confidence
    if primary_proba is not None:
        confidence = np.max(primary_proba, axis=1)
        bet_size = correct * confidence
    else:
        bet_size = correct

    # Apply quality weights
    if quality_weights is not None:
        bet_size = bet_size * quality_weights

    return np.clip(bet_size, 0.0, 1.0)


# =============================================================================
# META-LABELING TRAINER
# =============================================================================


class MetaLabelingTrainer:
    """
    Meta-labeling trainer implementing Lopez de Prado's two-stage approach.

    Stage 1: Train primary model to predict direction (side)
    Stage 2: Train meta-model to predict bet size (confidence)

    At inference time:
    1. Primary model predicts direction
    2. Meta-model predicts confidence/bet size
    3. Only take trades where bet_size >= threshold

    Example:
        >>> config = ExperimentConfig(symbol="MES", horizons=[20], models=["xgboost"])
        >>> trainer = MetaLabelingTrainer(config)
        >>> results = trainer.run(container)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        meta_config: MetaLabelingConfig | None = None,
    ) -> None:
        """
        Initialize meta-labeling trainer.

        Args:
            config: Experiment configuration
            meta_config: Meta-labeling specific configuration (optional)
        """
        self.config = config
        self.meta_config = meta_config or MetaLabelingConfig()
        self.output_dir = config.output_dir / "meta_labeling"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Trained models
        self.primary_trainer: Trainer | None = None
        self.meta_trainer: Trainer | None = None

        logger.info("Initialized MetaLabelingTrainer")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Primary model: {self.meta_config.primary_model}")
        logger.info(f"  Meta model: {self.meta_config.meta_model}")
        logger.info(f"  Meta threshold: {self.meta_config.meta_threshold}")

    def run(
        self,
        container: TimeSeriesDataContainer | None = None,
        save_models: bool = True,
    ) -> dict[str, Any]:
        """
        Run meta-labeling training.

        Args:
            container: TimeSeriesDataContainer with data. If None, loads from config.
            save_models: Whether to save trained models

        Returns:
            Dict with keys:
                - result: MetaLabelingResult
                - summary: Summary metrics
                - output_path: Path to output directory
        """
        start_time = time.time()

        # Load container if not provided
        if container is None:
            from src.core.container import TimeSeriesDataContainer

            container = TimeSeriesDataContainer.from_parquet_dir(
                path=self.config.data_dir,
                horizon=self.config.horizons[0],
            )

        logger.info("=" * 60)
        logger.info("META-LABELING TRAINING")
        logger.info("=" * 60)
        logger.info(f"Container: {container}")
        logger.info(f"Primary model: {self.meta_config.primary_model}")
        logger.info(f"Meta model: {self.meta_config.meta_model}")

        # Create output directory for this model combo
        model_dir = (
            self.output_dir
            / f"{self.meta_config.primary_model}_{self.meta_config.meta_model}_h{container.horizon}"
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        # Get data
        X_train, y_train, w_train = container.get_sklearn_arrays("train", return_df=True)
        X_val, y_val, _ = container.get_sklearn_arrays("val", return_df=True)
        X_test, y_test, _ = container.get_sklearn_arrays("test", return_df=True)

        # Get quality weights
        if self.meta_config.use_quality_weights and w_train is not None:
            quality_train = w_train.values
            quality_val = np.ones(len(X_val))
            quality_test = np.ones(len(X_test))
        else:
            quality_train = np.ones(len(X_train))
            quality_val = np.ones(len(X_val))
            quality_test = np.ones(len(X_test))

        # Stage 1: Train primary model
        logger.info("-" * 60)
        logger.info("STAGE 1: TRAIN PRIMARY MODEL (Side Prediction)")
        logger.info("-" * 60)

        primary_trainer, primary_results = self._train_primary_model(
            container=container,
            model_dir=model_dir,
            save_models=save_models,
        )
        self.primary_trainer = primary_trainer

        primary_f1 = primary_results["evaluation_metrics"].get("macro_f1", 0)
        primary_acc = primary_results["evaluation_metrics"].get("accuracy", 0)

        logger.info(f"  Primary Val F1: {primary_f1:.4f}")
        logger.info(f"  Primary Val Accuracy: {primary_acc:.4f}")

        # Generate primary predictions for meta-label creation
        primary_preds_train = primary_trainer.model.predict(np.asarray(X_train))
        primary_preds_val = primary_trainer.model.predict(np.asarray(X_val))

        # Stage 2: Compute meta-labels and train meta-model
        logger.info("-" * 60)
        logger.info("STAGE 2: TRAIN META-MODEL (Bet Size)")
        logger.info("-" * 60)

        # Compute meta-labels
        meta_labels_train = compute_directional_meta_labels(
            y_actual=np.asarray(y_train),
            y_pred_primary=primary_preds_train.class_predictions,
            primary_proba=primary_preds_train.class_probabilities,
            quality_weights=quality_train,
        )
        meta_labels_val = compute_directional_meta_labels(
            y_actual=np.asarray(y_val),
            y_pred_primary=primary_preds_val.class_predictions,
            primary_proba=primary_preds_val.class_probabilities,
            quality_weights=quality_val,
        )

        logger.info("  Meta-label statistics (train):")
        logger.info(f"    Mean: {meta_labels_train.mean():.3f}")
        logger.info(f"    Std:  {meta_labels_train.std():.3f}")
        logger.info(f"    Min:  {meta_labels_train.min():.3f}")
        logger.info(f"    Max:  {meta_labels_train.max():.3f}")

        meta_trainer, meta_results = self._train_meta_model(
            X_train=X_train,
            meta_labels_train=meta_labels_train,
            X_val=X_val,
            meta_labels_val=meta_labels_val,
            sample_weights=w_train,
            model_dir=model_dir,
            horizon=container.horizon,
            save_models=save_models,
        )
        self.meta_trainer = meta_trainer

        # Stage 3: Evaluate combined system on test set
        logger.info("-" * 60)
        logger.info("STAGE 3: EVALUATE COMBINED SYSTEM")
        logger.info("-" * 60)

        combined_metrics = self._evaluate_combined_system(
            X_test=X_test,
            y_test=y_test,
            quality_test=quality_test,
        )

        total_time = time.time() - start_time

        # Build result
        result = MetaLabelingResult(
            primary_model=self.meta_config.primary_model,
            meta_model=self.meta_config.meta_model,
            horizon=container.horizon,
            primary_metrics={
                "val_f1": primary_f1,
                "val_accuracy": primary_acc,
            },
            meta_metrics=meta_results.get("evaluation_metrics", {}),
            combined_metrics=combined_metrics,
            primary_model_path=model_dir / "primary_model.pkl" if save_models else None,
            meta_model_path=model_dir / "meta_model.pkl" if save_models else None,
            total_time=total_time,
        )

        # Save summary
        if save_models:
            summary_path = model_dir / "meta_labeling_results.json"
            with open(summary_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Summary saved to: {summary_path}")

        logger.info("=" * 60)
        logger.info("META-LABELING TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")

        self._log_summary(result)

        return {
            "result": result,
            "summary": result.to_dict(),
            "output_path": str(model_dir),
            "total_time": total_time,
        }

    def _train_primary_model(
        self,
        container: TimeSeriesDataContainer,
        model_dir: Path,
        save_models: bool = True,
    ) -> tuple[Trainer, dict[str, Any]]:
        """
        Train primary model for side prediction.

        Args:
            container: Data container
            model_dir: Output directory
            save_models: Whether to save model

        Returns:
            Tuple of (Trainer, results dict)
        """
        trainer_config = TrainerConfig(
            model_name=self.meta_config.primary_model,
            horizon=container.horizon,
            sequence_length=self.meta_config.sequence_length,
            output_dir=model_dir,
        )

        trainer = Trainer(trainer_config)
        results = trainer.run(container, skip_save=not save_models)

        if save_models:
            model_path = model_dir / "primary_model.pkl"
            trainer.model.save(model_path)
            logger.info(f"  Primary model saved to: {model_path}")

        return trainer, results

    def _train_meta_model(
        self,
        X_train: pd.DataFrame,
        meta_labels_train: np.ndarray,
        X_val: pd.DataFrame,
        meta_labels_val: np.ndarray,
        sample_weights: pd.Series | None,
        model_dir: Path,
        horizon: int,
        save_models: bool = True,
    ) -> tuple[Trainer, dict[str, Any]]:
        """
        Train meta-model for bet size prediction.

        Args:
            X_train: Training features
            meta_labels_train: Meta-labels (bet sizes) for training
            X_val: Validation features
            meta_labels_val: Meta-labels for validation
            sample_weights: Sample weights
            model_dir: Output directory
            horizon: Label horizon
            save_models: Whether to save model

        Returns:
            Tuple of (Trainer, results dict)
        """
        from src.core.container import TimeSeriesDataContainer

        # Create container for meta-model training
        # Note: meta-labels are continuous [0, 1], but we'll discretize for classification
        # Or use regression-capable model

        # For simplicity, we discretize into 3 classes: skip (0-0.3), partial (0.3-0.7), full (0.7-1.0)
        def discretize_meta_labels(labels: np.ndarray) -> np.ndarray:
            discrete = np.zeros(len(labels), dtype=int)
            discrete[labels >= 0.7] = 2  # Full bet
            discrete[(labels >= 0.3) & (labels < 0.7)] = 1  # Partial bet
            # 0 = skip (default)
            return discrete

        y_meta_train_discrete = discretize_meta_labels(meta_labels_train)
        y_meta_val_discrete = discretize_meta_labels(meta_labels_val)

        logger.info("  Discretized meta-labels (train):")
        for label, name in [(0, "skip"), (1, "partial"), (2, "full")]:
            count = (y_meta_train_discrete == label).sum()
            pct = 100 * count / len(y_meta_train_discrete)
            logger.info(f"    {name} ({label}): {count} ({pct:.1f}%)")

        # Build DataFrame for container
        train_df = X_train.copy().reset_index(drop=True)
        train_df[f"label_h{horizon}"] = y_meta_train_discrete
        if sample_weights is not None:
            train_df[f"sample_weight_h{horizon}"] = sample_weights.values

        val_df = X_val.copy().reset_index(drop=True)
        val_df[f"label_h{horizon}"] = y_meta_val_discrete

        meta_container = TimeSeriesDataContainer.from_dataframes(
            train_df=train_df,
            val_df=val_df,
            horizon=horizon,
            feature_columns=list(X_train.columns),
        )

        # Train meta-model
        trainer_config = TrainerConfig(
            model_name=self.meta_config.meta_model,
            horizon=horizon,
            sequence_length=self.meta_config.sequence_length,
            output_dir=model_dir,
        )

        trainer = Trainer(trainer_config)
        results = trainer.run(meta_container, skip_save=not save_models)

        if save_models:
            model_path = model_dir / "meta_model.pkl"
            trainer.model.save(model_path)
            logger.info(f"  Meta model saved to: {model_path}")

        return trainer, results

    def _evaluate_combined_system(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        quality_test: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate combined primary + meta system on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            quality_test: Quality weights for test

        Returns:
            Dict with evaluation metrics
        """
        if self.primary_trainer is None or self.meta_trainer is None:
            raise RuntimeError("Models not trained. Call run() first.")

        # Get primary predictions
        primary_preds = self.primary_trainer.model.predict(np.asarray(X_test))

        # Get meta predictions (bet sizes)
        meta_preds = self.meta_trainer.model.predict(np.asarray(X_test))

        # Convert meta predictions to bet sizes
        # If meta model outputs probabilities, use class 2 (full) probability as bet size
        if meta_preds.class_probabilities is not None:
            # P(full_bet) + 0.5 * P(partial_bet)
            bet_sizes = (
                meta_preds.class_probabilities[:, 2] + 0.5 * meta_preds.class_probabilities[:, 1]
            )
        else:
            # Map discrete predictions to bet sizes
            bet_sizes = np.where(
                meta_preds.class_predictions == 2,
                1.0,
                np.where(meta_preds.class_predictions == 1, 0.5, 0.0),
            )

        y_actual = np.asarray(y_test)

        # Primary-only accuracy
        primary_correct = (primary_preds.class_predictions == y_actual).mean()

        # Combined accuracy (only on trades taken)
        trades_taken = bet_sizes >= self.meta_config.meta_threshold
        n_trades = trades_taken.sum()

        if n_trades > 0:
            combined_correct = (
                primary_preds.class_predictions[trades_taken] == y_actual[trades_taken]
            ).mean()
            trade_fraction = n_trades / len(X_test)
        else:
            combined_correct = 0.0
            trade_fraction = 0.0

        # Log results
        logger.info("  Test Set Results:")
        logger.info("    Primary Only:")
        logger.info(f"      Accuracy: {primary_correct:.4f}")
        logger.info("      Trades: 100%")
        logger.info(f"\n    With Meta-Labeling (threshold={self.meta_config.meta_threshold}):")
        logger.info(f"      Accuracy: {combined_correct:.4f}")
        logger.info(f"      Trades Taken: {trade_fraction*100:.1f}%")
        logger.info(f"      Trades Skipped: {(1-trade_fraction)*100:.1f}%")

        improvement = combined_correct - primary_correct if trade_fraction > 0 else 0.0

        if improvement > 0:
            logger.info(f"      Accuracy Improvement: +{improvement:.4f}")
        elif trade_fraction > 0:
            logger.info(f"      Accuracy Change: {improvement:.4f}")

        return {
            "primary_accuracy": float(primary_correct),
            "combined_accuracy": float(combined_correct),
            "trade_fraction": float(trade_fraction),
            "trades_taken": int(n_trades),
            "total_samples": len(X_test),
            "meta_threshold": self.meta_config.meta_threshold,
            "improvement": float(improvement),
        }

    def _log_summary(self, result: MetaLabelingResult) -> None:
        """Log summary to console."""
        logger.info("Meta-Labeling Summary:")
        logger.info(f"  Primary model: {result.primary_model}")
        logger.info(f"  Meta model: {result.meta_model}")
        logger.info(f"  Primary Val F1: {result.primary_metrics.get('val_f1', 0):.4f}")
        logger.info(f"  Primary Val Accuracy: {result.primary_metrics.get('val_accuracy', 0):.4f}")
        logger.info(
            f"  Combined Test Accuracy: {result.combined_metrics.get('combined_accuracy', 0):.4f}"
        )
        logger.info(
            f"  Trade Fraction: {result.combined_metrics.get('trade_fraction', 0)*100:.1f}%"
        )
        logger.info(f"  Improvement: {result.combined_metrics.get('improvement', 0):+.4f}")
        logger.info(f"  Total time: {result.total_time:.1f}s")

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        return_details: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions using meta-labeling system.

        Args:
            X: Features (DataFrame or array)
            return_details: If True, return (predictions, bet_sizes, trade_mask)

        Returns:
            Predictions array, or (predictions, bet_sizes, trade_mask) if return_details
        """
        if self.primary_trainer is None or self.meta_trainer is None:
            raise RuntimeError("Models not trained. Call run() first.")

        X_arr = np.asarray(X)

        # Get primary predictions
        primary_preds = self.primary_trainer.model.predict(X_arr)

        # Get meta predictions
        meta_preds = self.meta_trainer.model.predict(X_arr)

        # Convert to bet sizes
        if meta_preds.class_probabilities is not None:
            bet_sizes = (
                meta_preds.class_probabilities[:, 2] + 0.5 * meta_preds.class_probabilities[:, 1]
            )
        else:
            bet_sizes = np.where(
                meta_preds.class_predictions == 2,
                1.0,
                np.where(meta_preds.class_predictions == 1, 0.5, 0.0),
            )

        # Apply threshold
        trade_mask = bet_sizes >= self.meta_config.meta_threshold

        # Predictions (set to neutral/0 for skipped trades)
        predictions = primary_preds.class_predictions.copy()
        # Could optionally set skipped trades to neutral: predictions[~trade_mask] = 0

        if return_details:
            return predictions, bet_sizes, trade_mask
        return predictions

    def get_position_sizes(
        self,
        X: pd.DataFrame | np.ndarray,
        base_position: float = 1.0,
    ) -> np.ndarray:
        """
        Get position sizes based on bet sizes and predictions.

        Args:
            X: Features
            base_position: Base position size (default 1.0)

        Returns:
            Array of position sizes (positive for long, negative for short, 0 for skip)
        """
        predictions, bet_sizes, trade_mask = self.predict(X, return_details=True)

        # Convert predictions to direction (-1, 0, 1)
        # Assuming predictions are {0, 1, 2} -> {short, neutral, long}
        direction = predictions - 1  # Map to {-1, 0, 1}

        # Position size = direction * bet_size * base_position
        position_sizes = direction * bet_sizes * base_position

        # Zero out skipped trades
        position_sizes[~trade_mask] = 0.0

        return position_sizes
