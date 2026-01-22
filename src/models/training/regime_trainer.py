"""
RegimeAwareTrainer - Regime-specific model training for PHASE_3.

PHASE_3: Training Orchestration

This module provides a complete RegimeAwareTrainer that:
1. Detects market regimes using RegimeDetector
2. Splits data by regime
3. Trains separate models per regime
4. Provides regime-specific predictions

The trainer integrates with UnifiedTrainingOrchestrator and uses
PipelineConfig as the ONLY configuration source.

Example:
    from src.core import PipelineConfig
    from src.models.training.regime_trainer import RegimeAwareTrainer

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./experiments",
        training_mode="regime_aware",
        regime_detection_method="combined",
        n_regimes=3,
        models=["xgboost", "lightgbm"],
    )

    trainer = RegimeAwareTrainer(config)
    result = trainer.train(prepared_data, horizon=20)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .regime_detector import (
    RegimeDetector,
    RegimeResult,
)

if TYPE_CHECKING:
    from src.core import PipelineConfig
    from src.data.adapters import PreparedData

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class RegimeModelResult:
    """
    Result from training a model on a single regime.

    Attributes:
        regime: Regime label
        model_name: Name of the model
        n_samples: Number of training samples
        sample_fraction: Fraction of total samples
        metrics: Validation metrics dict
        trainer: Trained model instance
        training_time_seconds: Training time
    """

    regime: str
    model_name: str
    n_samples: int
    sample_fraction: float
    metrics: dict[str, float] = field(default_factory=dict)
    trainer: Any | None = None
    training_time_seconds: float = 0.0

    @property
    def val_f1(self) -> float:
        """Get validation F1 score."""
        return self.metrics.get("val_f1", self.metrics.get("macro_f1", 0.0))

    @property
    def val_accuracy(self) -> float:
        """Get validation accuracy."""
        return self.metrics.get("val_accuracy", self.metrics.get("accuracy", 0.0))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime,
            "model_name": self.model_name,
            "n_samples": self.n_samples,
            "sample_fraction": self.sample_fraction,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
        }


@dataclass
class RegimeTrainingResult:
    """
    Complete result from regime-aware training.

    Attributes:
        horizon: Prediction horizon
        regime_method: Regime detection method used
        n_regimes: Number of regimes trained
        regime_results: Dict mapping (model_name, regime) -> RegimeModelResult
        regime_distribution: Distribution of samples across regimes
        aggregated_metrics: Overall aggregated metrics
        total_time_seconds: Total training time
        detector: RegimeDetector instance for inference
    """

    horizon: int
    regime_method: str
    n_regimes: int
    regime_results: dict[tuple[str, str], RegimeModelResult] = field(default_factory=dict)
    regime_distribution: dict[str, int] = field(default_factory=dict)
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    total_time_seconds: float = 0.0
    detector: RegimeDetector | None = None

    def get_model(self, model_name: str, regime: str) -> Any | None:
        """Get trained model for a specific model/regime combination."""
        key = (model_name, regime)
        if key in self.regime_results:
            return self.regime_results[key].trainer
        return None

    def get_regime_models(self, model_name: str) -> dict[str, Any]:
        """Get all regime models for a given model name."""
        result = {}
        for (name, regime), res in self.regime_results.items():
            if name == model_name and res.trainer is not None:
                result[regime] = res.trainer
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "horizon": self.horizon,
            "regime_method": self.regime_method,
            "n_regimes": self.n_regimes,
            "regime_results": {
                f"{k[0]}_{k[1]}": v.to_dict() for k, v in self.regime_results.items()
            },
            "regime_distribution": self.regime_distribution,
            "aggregated_metrics": self.aggregated_metrics,
            "total_time_seconds": self.total_time_seconds,
        }


# =============================================================================
# REGIME-AWARE TRAINER
# =============================================================================


class RegimeAwareTrainer:
    """
    Regime-aware model trainer for PHASE_3.

    Trains separate models for different market regimes, allowing
    different strategies for different market conditions.

    Features:
    - Multiple regime detection methods (volatility, trend, combined)
    - Configurable number of regimes (2 or 3)
    - Trains all specified models on each regime
    - Provides regime-specific predictions at inference time

    Usage:
        from src.core import PipelineConfig
        from src.models.training.regime_trainer import RegimeAwareTrainer

        config = PipelineConfig(
            symbol="MES",
            training_mode="regime_aware",
            regime_detection_method="combined",
            n_regimes=3,
        )

        trainer = RegimeAwareTrainer(config)
        result = trainer.train(prepared_data, horizon=20)

        # Predictions use regime detection automatically
        predictions = trainer.predict(X_test)
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize RegimeAwareTrainer.

        Args:
            config: PipelineConfig with regime-aware settings
        """
        self.config = config
        self.output_dir = config.output_dir / "regime_aware"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize regime detector
        self.detector = RegimeDetector.from_config(config)

        # Trained models storage: (model_name, regime) -> Trainer
        self._trainers: dict[tuple[str, str], Any] = {}

        # Store training result for inference
        self._training_result: RegimeTrainingResult | None = None

        logger.info("Initialized RegimeAwareTrainer")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Detection method: {config.regime_detection_method}")
        logger.info(f"  Number of regimes: {config.n_regimes}")
        logger.info(f"  Separate models: {config.train_separate_regime_models}")

    def train(
        self,
        prepared: PreparedData,
        horizon: int,
        model_name: str | None = None,
        save_models: bool = True,
    ) -> RegimeTrainingResult:
        """
        Train regime-aware models.

        Args:
            prepared: PreparedData from PHASE_2 adapters
            horizon: Prediction horizon
            model_name: Optional specific model to train (None = all from config)
            save_models: Whether to save trained models

        Returns:
            RegimeTrainingResult with all trained models and metrics
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("REGIME-AWARE TRAINING")
        logger.info("=" * 60)
        logger.info(f"Horizon: {horizon}")
        logger.info(f"Detection method: {self.config.regime_detection_method}")

        # Determine models to train
        models_to_train = [model_name] if model_name else self.config.models

        # Detect regimes on training data
        # Need to reconstruct DataFrame from prepared data for regime detection
        train_df = self._create_df_for_detection(prepared)
        regime_result = self.detector.detect(train_df)

        logger.info("\nRegime distribution (training data):")
        for regime, count in regime_result.regime_counts.items():
            pct = 100 * regime_result.regime_fractions[regime]
            logger.info(f"  {regime}: {count} samples ({pct:.1f}%)")

        # Train models per regime
        all_results: dict[tuple[str, str], RegimeModelResult] = {}

        if self.config.train_separate_regime_models:
            all_results = self._train_separate_models(
                prepared=prepared,
                horizon=horizon,
                models=models_to_train,
                regime_result=regime_result,
                save_models=save_models,
            )
        else:
            # Train single model with regime as feature
            all_results = self._train_with_regime_feature(
                prepared=prepared,
                horizon=horizon,
                models=models_to_train,
                regime_result=regime_result,
                save_models=save_models,
            )

        # Compute aggregated metrics
        aggregated_metrics = self._compute_aggregated_metrics(all_results)

        total_time = time.time() - start_time

        # Build result
        result = RegimeTrainingResult(
            horizon=horizon,
            regime_method=self.config.regime_detection_method,
            n_regimes=regime_result.n_regimes,
            regime_results=all_results,
            regime_distribution=regime_result.regime_counts,
            aggregated_metrics=aggregated_metrics,
            total_time_seconds=total_time,
            detector=self.detector,
        )

        # Store for inference
        self._training_result = result

        # Save summary
        if save_models:
            self._save_summary(result)

        logger.info("\n" + "=" * 60)
        logger.info("REGIME-AWARE TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Regimes trained: {result.n_regimes}")

        for metric, value in aggregated_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")

        return result

    def _train_separate_models(
        self,
        prepared: PreparedData,
        horizon: int,
        models: list[str],
        regime_result: RegimeResult,
        save_models: bool,
    ) -> dict[tuple[str, str], RegimeModelResult]:
        """Train separate models for each regime."""
        from src.core.container import TimeSeriesDataContainer
        from src.models import Trainer, TrainerConfig

        all_results: dict[tuple[str, str], RegimeModelResult] = {}

        for regime_label in regime_result.get_regime_labels():
            logger.info(f"\n--- Regime: {regime_label} ---")

            # Get mask for this regime
            mask = regime_result.get_mask(regime_label)
            train_mask = mask[: len(prepared.y_train)].values

            # Check minimum samples
            n_train = train_mask.sum()
            if n_train < self.config.regime_min_samples:
                logger.warning(
                    f"Skipping {regime_label}: insufficient samples "
                    f"({n_train} < {self.config.regime_min_samples})"
                )
                continue

            logger.info(f"Training samples: {n_train}")

            # Extract regime-specific data
            X_train_regime = prepared.X_train[train_mask]
            y_train_regime = prepared.y_train[train_mask]

            # Handle validation set
            # Note: We train on regime-specific data but may validate on all data
            # or regime-specific validation data depending on use case
            X_val = prepared.X_val
            y_val = prepared.y_val

            for model_name in models:
                logger.info(f"\n  Training {model_name}...")
                model_start = time.time()

                try:
                    # Prepare data for model
                    if prepared.data_rank == 2:
                        X_train_df = pd.DataFrame(
                            X_train_regime,
                            columns=prepared.feature_names,
                        )
                        X_val_df = pd.DataFrame(
                            X_val,
                            columns=prepared.feature_names,
                        )
                    else:
                        # Flatten 3D/4D for tabular models
                        n_train = X_train_regime.shape[0]
                        n_val = X_val.shape[0]
                        n_features = np.prod(X_train_regime.shape[1:])

                        X_train_df = pd.DataFrame(
                            X_train_regime.reshape(n_train, -1),
                            columns=[f"f{i}" for i in range(n_features)],
                        )
                        X_val_df = pd.DataFrame(
                            X_val.reshape(n_val, -1),
                            columns=[f"f{i}" for i in range(n_features)],
                        )

                    # Create container
                    container = TimeSeriesDataContainer(
                        X_train=X_train_df,
                        y_train=pd.Series(y_train_regime),
                        X_val=X_val_df,
                        y_val=pd.Series(y_val),
                        X_test=pd.DataFrame(),
                        y_test=pd.Series(dtype=float),
                        sample_weights=pd.Series(np.ones(len(y_train_regime))),
                    )

                    # Create trainer config
                    model_dir = self.output_dir / f"h{horizon}" / regime_label
                    model_dir.mkdir(parents=True, exist_ok=True)

                    trainer_config = TrainerConfig(
                        model_name=model_name,
                        horizon=horizon,
                        output_dir=model_dir,
                    )

                    # Train
                    trainer = Trainer(trainer_config)
                    results = trainer.run(container)

                    training_time = time.time() - model_start

                    # Store trainer
                    key = (model_name, regime_label)
                    self._trainers[key] = trainer

                    # Save model
                    if save_models:
                        model_path = model_dir / f"{model_name}.pkl"
                        try:
                            trainer.save(model_path)
                        except Exception as e:
                            logger.warning(f"Failed to save model: {e}")

                    # Build result
                    metrics = results.get("evaluation_metrics", {})
                    result = RegimeModelResult(
                        regime=regime_label,
                        model_name=model_name,
                        n_samples=n_train,
                        sample_fraction=n_train / len(prepared.y_train),
                        metrics=metrics,
                        trainer=trainer,
                        training_time_seconds=training_time,
                    )

                    all_results[key] = result

                    logger.info(
                        f"    val_f1={result.val_f1:.4f}, "
                        f"val_accuracy={result.val_accuracy:.4f}, "
                        f"time={training_time:.1f}s"
                    )

                except Exception as e:
                    logger.error(f"Failed to train {model_name} on {regime_label}: {e}")
                    raise

        return all_results

    def _train_with_regime_feature(
        self,
        prepared: PreparedData,
        horizon: int,
        models: list[str],
        regime_result: RegimeResult,
        save_models: bool,
    ) -> dict[tuple[str, str], RegimeModelResult]:
        """Train single model with regime as additional feature."""
        from src.core.container import TimeSeriesDataContainer
        from src.models import Trainer, TrainerConfig

        all_results: dict[tuple[str, str], RegimeModelResult] = {}

        # Add regime as one-hot encoded feature
        regime_dummies = pd.get_dummies(
            regime_result.regimes[: len(prepared.y_train)],
            prefix="regime",
        )

        # Prepare augmented features
        if prepared.data_rank == 2:
            X_train_df = pd.DataFrame(
                prepared.X_train,
                columns=prepared.feature_names,
            )
            X_val_df = pd.DataFrame(
                prepared.X_val,
                columns=prepared.feature_names,
            )
        else:
            n_train = prepared.X_train.shape[0]
            n_val = prepared.X_val.shape[0]
            n_features = np.prod(prepared.X_train.shape[1:])

            X_train_df = pd.DataFrame(
                prepared.X_train.reshape(n_train, -1),
                columns=[f"f{i}" for i in range(n_features)],
            )
            X_val_df = pd.DataFrame(
                prepared.X_val.reshape(n_val, -1),
                columns=[f"f{i}" for i in range(n_features)],
            )

        # Concatenate regime features
        X_train_augmented = pd.concat(
            [X_train_df.reset_index(drop=True), regime_dummies.reset_index(drop=True)],
            axis=1,
        )

        # For validation, detect regimes and add
        val_df = self._create_df_for_detection_from_array(prepared.X_val, prepared)
        val_regimes = self.detector.detect(val_df)
        val_regime_dummies = pd.get_dummies(val_regimes.regimes, prefix="regime")

        # Ensure same columns
        for col in regime_dummies.columns:
            if col not in val_regime_dummies.columns:
                val_regime_dummies[col] = 0
        val_regime_dummies = val_regime_dummies[regime_dummies.columns]

        X_val_augmented = pd.concat(
            [X_val_df.reset_index(drop=True), val_regime_dummies.reset_index(drop=True)],
            axis=1,
        )

        logger.info(
            f"Added {len(regime_dummies.columns)} regime features "
            f"(total: {X_train_augmented.shape[1]})"
        )

        for model_name in models:
            logger.info(f"\nTraining {model_name} with regime features...")
            model_start = time.time()

            try:
                # Create container
                container = TimeSeriesDataContainer(
                    X_train=X_train_augmented,
                    y_train=pd.Series(prepared.y_train),
                    X_val=X_val_augmented,
                    y_val=pd.Series(prepared.y_val),
                    X_test=pd.DataFrame(),
                    y_test=pd.Series(dtype=float),
                    sample_weights=pd.Series(np.ones(len(prepared.y_train))),
                )

                # Create trainer config
                model_dir = self.output_dir / f"h{horizon}" / "combined"
                model_dir.mkdir(parents=True, exist_ok=True)

                trainer_config = TrainerConfig(
                    model_name=model_name,
                    horizon=horizon,
                    output_dir=model_dir,
                )

                # Train
                trainer = Trainer(trainer_config)
                results = trainer.run(container)

                training_time = time.time() - model_start

                # Store trainer with special "combined" regime key
                key = (model_name, "combined")
                self._trainers[key] = trainer

                # Save model
                if save_models:
                    model_path = model_dir / f"{model_name}.pkl"
                    try:
                        trainer.save(model_path)
                    except Exception as e:
                        logger.warning(f"Failed to save model: {e}")

                # Build result
                metrics = results.get("evaluation_metrics", {})
                result = RegimeModelResult(
                    regime="combined",
                    model_name=model_name,
                    n_samples=len(prepared.y_train),
                    sample_fraction=1.0,
                    metrics=metrics,
                    trainer=trainer,
                    training_time_seconds=training_time,
                )

                all_results[key] = result

                logger.info(
                    f"  val_f1={result.val_f1:.4f}, "
                    f"val_accuracy={result.val_accuracy:.4f}, "
                    f"time={training_time:.1f}s"
                )

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                raise

        return all_results

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        model_name: str,
        return_regimes: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, pd.Series]:
        """
        Generate predictions using regime-aware models.

        Args:
            X: Features array or DataFrame
            model_name: Name of the model to use
            return_regimes: If True, also return detected regimes

        Returns:
            Predictions array, or (predictions, regimes) if return_regimes=True
        """
        if not self._trainers:
            raise RuntimeError("No trained models. Call train() first.")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        # Create DataFrame for regime detection
        # This is a simplification - in practice, need original OHLCV
        regimes = self.detector.detect(X_df)

        # Initialize predictions
        predictions = np.full(len(X), np.nan)

        if self.config.train_separate_regime_models:
            # Use regime-specific models
            for regime_label in regimes.get_regime_labels():
                key = (model_name, regime_label)
                if key not in self._trainers:
                    logger.warning(f"No model for {model_name}/{regime_label}")
                    continue

                mask = regimes.get_mask(regime_label).values
                trainer = self._trainers[key]

                if mask.sum() > 0:
                    X_regime = X_df[mask].values
                    pred_output = trainer.model.predict(X_regime)
                    predictions[mask] = pred_output.class_predictions
        else:
            # Use combined model with regime features
            key = (model_name, "combined")
            if key not in self._trainers:
                raise RuntimeError(f"Combined model not found for {model_name}")

            # Add regime features
            regime_dummies = pd.get_dummies(regimes.regimes, prefix="regime")
            X_augmented = pd.concat(
                [X_df.reset_index(drop=True), regime_dummies.reset_index(drop=True)],
                axis=1,
            )

            trainer = self._trainers[key]
            pred_output = trainer.model.predict(X_augmented.values)
            predictions = pred_output.class_predictions

        if return_regimes:
            return predictions, regimes.regimes
        return predictions

    def _create_df_for_detection(self, prepared: PreparedData) -> pd.DataFrame:
        """Create DataFrame suitable for regime detection from PreparedData."""
        # If we have feature names that include price data, use them
        if prepared.data_rank == 2:
            df = pd.DataFrame(
                prepared.X_train,
                columns=prepared.feature_names,
            )
        else:
            # For 3D/4D data, we need to handle this differently
            # Use last timestep of sequence as proxy
            if prepared.data_rank == 3:
                # Shape: (n_samples, seq_len, n_features)
                # Use last timestep
                df = pd.DataFrame(
                    prepared.X_train[:, -1, :],
                    columns=prepared.feature_names,
                )
            else:
                # Shape: (n_samples, n_timeframes, seq_len, n_features)
                # Use first timeframe, last timestep
                df = pd.DataFrame(
                    prepared.X_train[:, 0, -1, :],
                    columns=prepared.feature_names,
                )

        return df

    def _create_df_for_detection_from_array(
        self,
        X: np.ndarray,
        prepared: PreparedData,
    ) -> pd.DataFrame:
        """Create DataFrame for regime detection from raw array."""
        if prepared.data_rank == 2:
            return pd.DataFrame(X, columns=prepared.feature_names)
        elif prepared.data_rank == 3:
            return pd.DataFrame(X[:, -1, :], columns=prepared.feature_names)
        else:
            return pd.DataFrame(X[:, 0, -1, :], columns=prepared.feature_names)

    def _compute_aggregated_metrics(
        self,
        results: dict[tuple[str, str], RegimeModelResult],
    ) -> dict[str, float]:
        """Compute aggregated metrics across all regime/model combinations."""
        if not results:
            return {}

        # Group by model
        model_metrics: dict[str, list[RegimeModelResult]] = {}
        for (model_name, regime), result in results.items():
            if model_name not in model_metrics:
                model_metrics[model_name] = []
            model_metrics[model_name].append(result)

        aggregated: dict[str, float] = {}

        # Compute weighted averages per model
        for model_name, model_results in model_metrics.items():
            total_samples = sum(r.n_samples for r in model_results)

            weighted_f1 = sum(r.val_f1 * r.n_samples / total_samples for r in model_results)
            weighted_accuracy = sum(
                r.val_accuracy * r.n_samples / total_samples for r in model_results
            )

            aggregated[f"{model_name}_weighted_f1"] = weighted_f1
            aggregated[f"{model_name}_weighted_accuracy"] = weighted_accuracy

        # Overall metrics
        all_results = list(results.values())
        total_samples = sum(r.n_samples for r in all_results)

        aggregated["overall_weighted_f1"] = sum(
            r.val_f1 * r.n_samples / total_samples for r in all_results
        )
        aggregated["overall_weighted_accuracy"] = sum(
            r.val_accuracy * r.n_samples / total_samples for r in all_results
        )
        aggregated["n_models_trained"] = len(results)
        aggregated["n_regimes"] = len(set(r.regime for r in all_results))

        return aggregated

    def _save_summary(self, result: RegimeTrainingResult) -> None:
        """Save training summary to JSON."""
        summary_path = self.output_dir / "regime_training_summary.json"

        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Summary saved to: {summary_path}")

    def get_trained_model(
        self,
        model_name: str,
        regime: str,
    ) -> Any | None:
        """Get a trained model by name and regime."""
        return self._trainers.get((model_name, regime))

    def get_all_regime_models(self, model_name: str) -> dict[str, Any]:
        """Get all regime models for a given model name."""
        return {
            regime: trainer
            for (name, regime), trainer in self._trainers.items()
            if name == model_name
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RegimeAwareTrainer",
    "RegimeTrainingResult",
    "RegimeModelResult",
]
