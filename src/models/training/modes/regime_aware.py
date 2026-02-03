"""
Regime-aware training mode.

Detects market regimes and trains separate models for each regime, or uses
regime as an additional feature. This improves performance by allowing
different strategies for different market conditions.

Regime types:
- Volatility: high_vol, low_vol
- Trend: trending, mean_reverting
- Composite: trending_low_vol, trending_high_vol, mean_reverting_low_vol, mean_reverting_high_vol
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer

from src.models import Trainer, TrainerConfig

from ..config import ExperimentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Supported regime types
REGIME_TYPES = Literal["volatility", "trend", "composite"]

# Composite regime labels
COMPOSITE_REGIMES = [
    "trending_low_vol",
    "trending_high_vol",
    "mean_reverting_low_vol",
    "mean_reverting_high_vol",
]

# Simple regime labels
VOLATILITY_REGIMES = ["low_vol", "high_vol"]
TREND_REGIMES = ["trending", "mean_reverting"]


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class RegimeAwareConfig:
    """
    Configuration for regime-aware training.

    Attributes:
        regime_type: Type of regime detection ("volatility", "trend", "composite")
        train_separate_models: If True, train separate model per regime
        use_regime_as_feature: If True, add regime label as feature to single model
        regimes_to_train: List of specific regimes to train on (None = all)
        min_samples_per_regime: Minimum samples required per regime
        adx_threshold: ADX threshold for trend detection (default 25)
        volatility_multiplier: ATR multiplier for high vol detection (default 1.5)
    """

    regime_type: REGIME_TYPES = "composite"
    train_separate_models: bool = True
    use_regime_as_feature: bool = False
    regimes_to_train: list[str] | None = None
    min_samples_per_regime: int = 100
    adx_threshold: float = 25.0
    volatility_multiplier: float = 1.5


@dataclass
class RegimeTrainingResult:
    """
    Results from regime-aware training.

    Attributes:
        regime: Regime label
        n_samples: Number of samples in this regime
        sample_fraction: Fraction of total samples
        val_f1: Validation F1 score
        val_accuracy: Validation accuracy
        model_path: Path to saved model
        training_time: Training time in seconds
    """

    regime: str
    n_samples: int
    sample_fraction: float
    val_f1: float
    val_accuracy: float
    model_path: Path | None = None
    training_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime,
            "n_samples": self.n_samples,
            "sample_fraction": self.sample_fraction,
            "val_f1": self.val_f1,
            "val_accuracy": self.val_accuracy,
            "model_path": str(self.model_path) if self.model_path else None,
            "training_time": self.training_time,
        }


@dataclass
class RegimeAwareTrainingResult:
    """
    Aggregated results from regime-aware training.

    Attributes:
        model_name: Name of the trained model
        horizon: Label horizon
        regime_type: Type of regime detection used
        regime_results: Results per regime
        aggregated_metrics: Overall aggregated metrics
        regime_distribution: Distribution of samples across regimes
        total_time: Total training time
    """

    model_name: str
    horizon: int
    regime_type: str
    regime_results: dict[str, RegimeTrainingResult] = field(default_factory=dict)
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    regime_distribution: dict[str, int] = field(default_factory=dict)
    total_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "regime_type": self.regime_type,
            "regime_results": {k: v.to_dict() for k, v in self.regime_results.items()},
            "aggregated_metrics": self.aggregated_metrics,
            "regime_distribution": self.regime_distribution,
            "total_time": self.total_time,
        }


# =============================================================================
# REGIME DETECTION
# =============================================================================


def detect_volatility_regime(
    df: pd.DataFrame,
    volatility_multiplier: float = 1.5,
    window: int = 100,
) -> pd.Series:
    """
    Detect volatility regime (high_vol vs low_vol).

    Uses ATR compared to rolling mean to classify volatility.

    Args:
        df: DataFrame with features (must include atr_14)
        volatility_multiplier: Threshold multiplier for high volatility
        window: Rolling window for mean calculation

    Returns:
        Series with regime labels ("high_vol" or "low_vol")
    """
    # Try to find ATR column
    atr_col = None
    for col in ["atr_14", "atr", "ATR_14", "ATR"]:
        if col in df.columns:
            atr_col = col
            break

    if atr_col is None:
        raise ValueError("ATR column not found. Required for volatility regime detection.")

    atr = df[atr_col]
    atr_mean = atr.rolling(window, min_periods=20).mean()

    high_vol = atr > (volatility_multiplier * atr_mean)

    regime = pd.Series("low_vol", index=df.index)
    regime[high_vol] = "high_vol"

    return regime


def detect_trend_regime(
    df: pd.DataFrame,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """
    Detect trend regime (trending vs mean_reverting).

    Uses ADX to classify trend strength. If ADX is not available,
    falls back to autocorrelation-based detection.

    Args:
        df: DataFrame with features
        adx_threshold: ADX threshold for trending classification

    Returns:
        Series with regime labels ("trending" or "mean_reverting")
    """
    # Try to find ADX column
    adx_col = None
    for col in ["adx_14", "adx", "ADX_14", "ADX"]:
        if col in df.columns:
            adx_col = col
            break

    if adx_col is not None:
        adx = df[adx_col]
        trending = adx > adx_threshold
    else:
        # Fallback: use price autocorrelation
        logger.warning("ADX not found, using autocorrelation-based trend detection")

        close_col = None
        for col in ["close", "Close", "CLOSE"]:
            if col in df.columns:
                close_col = col
                break

        if close_col is None:
            raise ValueError("Neither ADX nor close price found for trend detection")

        close_returns = df[close_col].pct_change()
        # FIX: pandas autocorr(lag=1) requires at least 3 data points (lag+2)
        # to compute covariance without "Degrees of freedom <= 0" warning
        abs_autocorr = close_returns.rolling(20).apply(
            lambda x: abs(x.autocorr(lag=1)) if len(x) >= 3 else 0, raw=False
        )
        trending = abs_autocorr > 0.3

    regime = pd.Series("mean_reverting", index=df.index)
    regime[trending] = "trending"

    return regime


def detect_composite_regime(
    df: pd.DataFrame,
    adx_threshold: float = 25.0,
    volatility_multiplier: float = 1.5,
    window: int = 100,
) -> pd.Series:
    """
    Detect composite regime combining trend and volatility.

    Classifies into 4 regimes:
    - trending_low_vol
    - trending_high_vol
    - mean_reverting_low_vol
    - mean_reverting_high_vol

    Args:
        df: DataFrame with features
        adx_threshold: ADX threshold for trending
        volatility_multiplier: ATR multiplier for high vol
        window: Rolling window for volatility mean

    Returns:
        Series with composite regime labels
    """
    trend_regime = detect_trend_regime(df, adx_threshold)
    vol_regime = detect_volatility_regime(df, volatility_multiplier, window)

    regime = pd.Series("unknown", index=df.index)

    trending = trend_regime == "trending"
    high_vol = vol_regime == "high_vol"

    regime[trending & ~high_vol] = "trending_low_vol"
    regime[trending & high_vol] = "trending_high_vol"
    regime[~trending & ~high_vol] = "mean_reverting_low_vol"
    regime[~trending & high_vol] = "mean_reverting_high_vol"

    return regime


def detect_regimes(
    df: pd.DataFrame,
    regime_type: REGIME_TYPES = "composite",
    adx_threshold: float = 25.0,
    volatility_multiplier: float = 1.5,
) -> pd.Series:
    """
    Unified regime detection function.

    Args:
        df: DataFrame with features
        regime_type: Type of regime detection
        adx_threshold: ADX threshold for trend
        volatility_multiplier: ATR multiplier for volatility

    Returns:
        Series with regime labels
    """
    if regime_type == "volatility":
        return detect_volatility_regime(df, volatility_multiplier)
    elif regime_type == "trend":
        return detect_trend_regime(df, adx_threshold)
    else:  # composite
        return detect_composite_regime(df, adx_threshold, volatility_multiplier)


# =============================================================================
# REGIME-AWARE TRAINER
# =============================================================================


class RegimeAwareTrainer:
    """
    Regime-aware model trainer.

    Trains separate models for different market regimes or uses regime
    as an additional feature. This allows different strategies for
    different market conditions.

    Example:
        >>> config = ExperimentConfig(symbol="MES", horizons=[20], models=["xgboost"])
        >>> trainer = RegimeAwareTrainer(config)
        >>> results = trainer.run(container)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        regime_config: RegimeAwareConfig | None = None,
    ) -> None:
        """
        Initialize regime-aware trainer.

        Args:
            config: Experiment configuration
            regime_config: Regime-specific configuration (optional)
        """
        self.config = config
        self.regime_config = regime_config or RegimeAwareConfig()
        self.output_dir = config.output_dir / "regime_aware"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Trained models storage (regime -> Trainer)
        self.trainers: dict[str, Trainer] = {}

        logger.info("Initialized RegimeAwareTrainer")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Regime type: {self.regime_config.regime_type}")
        logger.info(f"  Separate models: {self.regime_config.train_separate_models}")

    def run(
        self,
        container: TimeSeriesDataContainer | None = None,
        save_models: bool = True,
    ) -> dict[str, Any]:
        """
        Run regime-aware training.

        Args:
            container: TimeSeriesDataContainer with data. If None, loads from config.
            save_models: Whether to save trained models

        Returns:
            Dict with keys:
                - model_results: Dict[model_name, RegimeAwareTrainingResult]
                - summary: Aggregated summary
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
        logger.info("REGIME-AWARE TRAINING")
        logger.info("=" * 60)
        logger.info(f"Container: {container}")

        # Get model names - handle both string and ModelConfig types
        model_names = [m if isinstance(m, str) else m.name for m in self.config.models]
        logger.info(f"Models: {model_names}")
        logger.info(f"Regime type: {self.regime_config.regime_type}")

        # Run regime-aware training for each model
        all_results: dict[str, RegimeAwareTrainingResult] = {}

        for model_name in model_names:
            logger.info("-" * 60)
            logger.info(f"Training regime-aware: {model_name}")
            logger.info("-" * 60)

            try:
                if self.regime_config.train_separate_models:
                    result = self._train_separate_models(
                        container=container,
                        model_name=model_name,
                        save_models=save_models,
                    )
                else:
                    result = self._train_with_regime_feature(
                        container=container,
                        model_name=model_name,
                        save_models=save_models,
                    )
                all_results[model_name] = result

            except Exception as e:
                logger.error(f"Failed regime-aware training for {model_name}: {e}")
                raise

        # Build summary
        summary = self._build_summary(all_results)

        total_time = time.time() - start_time

        # Save summary
        if save_models:
            summary_path = self.output_dir / "regime_summary.json"
            with open(summary_path, "w") as f:
                json.dump(
                    {
                        "summary": summary,
                        "results": {k: v.to_dict() for k, v in all_results.items()},
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Summary saved to: {summary_path}")

        logger.info("=" * 60)
        logger.info("REGIME-AWARE TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")

        self._log_summary(summary)

        return {
            "model_results": all_results,
            "summary": summary,
            "output_path": str(self.output_dir),
            "total_time": total_time,
        }

    def _train_separate_models(
        self,
        container: TimeSeriesDataContainer,
        model_name: str,
        save_models: bool = True,
    ) -> RegimeAwareTrainingResult:
        """
        Train separate models for each regime.

        Args:
            container: Data container
            model_name: Model to train
            save_models: Whether to save models

        Returns:
            RegimeAwareTrainingResult
        """
        model_start = time.time()

        # Get training data - cast to pd.DataFrame/Series since return_df=True
        X_train_raw, y_train_raw, w_train_raw = container.get_sklearn_arrays(
            "train", return_df=True
        )
        X_val_raw, y_val_raw, _ = container.get_sklearn_arrays("val", return_df=True)
        X_train = cast(pd.DataFrame, X_train_raw)
        y_train = cast(pd.Series, y_train_raw)
        w_train = cast(pd.Series, w_train_raw) if w_train_raw is not None else None
        X_val = cast(pd.DataFrame, X_val_raw)
        y_val = cast(pd.Series, y_val_raw)

        # Detect regimes
        train_regimes = detect_regimes(
            X_train,
            regime_type=self.regime_config.regime_type,
            adx_threshold=self.regime_config.adx_threshold,
            volatility_multiplier=self.regime_config.volatility_multiplier,
        )
        val_regimes = detect_regimes(
            X_val,
            regime_type=self.regime_config.regime_type,
            adx_threshold=self.regime_config.adx_threshold,
            volatility_multiplier=self.regime_config.volatility_multiplier,
        )

        # Log regime distribution
        regime_dist = train_regimes.value_counts().to_dict()
        logger.info("  Regime distribution (train):")
        for regime, count in regime_dist.items():
            pct = 100 * count / len(train_regimes)
            logger.info(f"    {regime}: {count} ({pct:.1f}%)")

        # Determine regimes to train
        regimes_to_train = self.regime_config.regimes_to_train
        if regimes_to_train is None:
            regimes_to_train = list(train_regimes.unique())

        # Train model for each regime
        regime_results: dict[str, RegimeTrainingResult] = {}
        model_dir = self.output_dir / f"{model_name}_h{container.horizon}"
        model_dir.mkdir(parents=True, exist_ok=True)

        for regime in regimes_to_train:
            regime_start = time.time()

            # Get regime mask
            train_mask = train_regimes == regime
            val_mask = val_regimes == regime

            n_train = train_mask.sum()
            n_val = val_mask.sum()

            # Check minimum samples
            if n_train < self.regime_config.min_samples_per_regime:
                logger.warning(
                    f"  Skipping {regime}: insufficient training data "
                    f"({n_train} < {self.regime_config.min_samples_per_regime})"
                )
                continue

            if n_val < 10:
                logger.warning(f"  Skipping {regime}: insufficient validation data ({n_val})")
                continue

            logger.info(f"\n  Training on regime: {regime}")
            logger.info(f"    Train samples: {n_train} ({100*n_train/len(X_train):.1f}%)")
            logger.info(f"    Val samples: {n_val}")

            # Create regime-specific container
            from src.core.container import TimeSeriesDataContainer

            regime_container = TimeSeriesDataContainer.from_dataframes(
                train_df=pd.concat(
                    [
                        X_train[train_mask].reset_index(drop=True),
                        y_train[train_mask].reset_index(drop=True).to_frame(),
                        (
                            w_train[train_mask].reset_index(drop=True).to_frame()
                            if w_train is not None
                            else pd.DataFrame()
                        ),
                    ],
                    axis=1,
                ),
                val_df=pd.concat(
                    [
                        X_val[val_mask].reset_index(drop=True),
                        y_val[val_mask].reset_index(drop=True).to_frame(),
                    ],
                    axis=1,
                ),
                horizon=container.horizon,
                feature_columns=list(X_train.columns),
            )

            # Create trainer config
            trainer_config = TrainerConfig(
                model_name=model_name,
                horizon=container.horizon,
                output_dir=model_dir,
            )

            # Train
            trainer = Trainer(trainer_config)
            results = trainer.run(regime_container, skip_save=not save_models)

            # Store trainer
            self.trainers[regime] = trainer

            # Save model
            model_path = None
            if save_models:
                model_path = model_dir / f"model_{regime}.pkl"
                trainer.model.save(model_path)
                logger.info(f"    Saved model to: {model_path}")

            regime_time = time.time() - regime_start

            # Build result
            val_f1 = results["evaluation_metrics"].get("macro_f1", 0)
            val_acc = results["evaluation_metrics"].get("accuracy", 0)

            logger.info(f"    Val F1: {val_f1:.4f}")
            logger.info(f"    Val Accuracy: {val_acc:.4f}")

            regime_results[regime] = RegimeTrainingResult(
                regime=regime,
                n_samples=n_train,
                sample_fraction=n_train / len(X_train),
                val_f1=val_f1,
                val_accuracy=val_acc,
                model_path=model_path,
                training_time=regime_time,
            )

        # Build aggregated metrics (weighted by sample fraction)
        if regime_results:
            total_samples = sum(r.n_samples for r in regime_results.values())
            aggregated = {
                "weighted_f1": sum(
                    r.val_f1 * r.n_samples / total_samples for r in regime_results.values()
                ),
                "weighted_accuracy": sum(
                    r.val_accuracy * r.n_samples / total_samples for r in regime_results.values()
                ),
                "n_regimes_trained": len(regime_results),
            }
        else:
            aggregated = {}

        total_time = time.time() - model_start

        return RegimeAwareTrainingResult(
            model_name=model_name,
            horizon=container.horizon,
            regime_type=self.regime_config.regime_type,
            regime_results=regime_results,
            aggregated_metrics=aggregated,
            regime_distribution=regime_dist,
            total_time=total_time,
        )

    def _train_with_regime_feature(
        self,
        container: TimeSeriesDataContainer,
        model_name: str,
        save_models: bool = True,
    ) -> RegimeAwareTrainingResult:
        """
        Train single model with regime as additional feature.

        Args:
            container: Data container
            model_name: Model to train
            save_models: Whether to save model

        Returns:
            RegimeAwareTrainingResult
        """
        model_start = time.time()

        # Get training data - cast to pd.DataFrame/Series since return_df=True
        X_train_raw, y_train_raw, w_train_raw = container.get_sklearn_arrays(
            "train", return_df=True
        )
        X_val_raw, y_val_raw, _ = container.get_sklearn_arrays("val", return_df=True)
        X_train = cast(pd.DataFrame, X_train_raw)
        y_train = cast(pd.Series, y_train_raw)
        w_train = cast(pd.Series, w_train_raw) if w_train_raw is not None else None
        X_val = cast(pd.DataFrame, X_val_raw)
        y_val = cast(pd.Series, y_val_raw)

        # Detect regimes
        train_regimes = detect_regimes(
            X_train,
            regime_type=self.regime_config.regime_type,
            adx_threshold=self.regime_config.adx_threshold,
            volatility_multiplier=self.regime_config.volatility_multiplier,
        )
        val_regimes = detect_regimes(
            X_val,
            regime_type=self.regime_config.regime_type,
            adx_threshold=self.regime_config.adx_threshold,
            volatility_multiplier=self.regime_config.volatility_multiplier,
        )

        # Log regime distribution
        regime_dist = train_regimes.value_counts().to_dict()
        logger.info("  Regime distribution (train):")
        for regime, count in regime_dist.items():
            pct = 100 * count / len(train_regimes)
            logger.info(f"    {regime}: {count} ({pct:.1f}%)")

        # Add regime as one-hot encoded features
        regime_dummies_train = pd.get_dummies(train_regimes, prefix="regime")
        regime_dummies_val = pd.get_dummies(val_regimes, prefix="regime")

        # Ensure same columns in train and val
        for col in regime_dummies_train.columns:
            if col not in regime_dummies_val.columns:
                regime_dummies_val[col] = 0
        for col in regime_dummies_val.columns:
            if col not in regime_dummies_train.columns:
                regime_dummies_train[col] = 0

        regime_dummies_train = regime_dummies_train[sorted(regime_dummies_train.columns)]
        regime_dummies_val = regime_dummies_val[sorted(regime_dummies_val.columns)]

        # Concatenate regime features
        X_train_augmented = pd.concat(
            [X_train.reset_index(drop=True), regime_dummies_train.reset_index(drop=True)],
            axis=1,
        )
        X_val_augmented = pd.concat(
            [X_val.reset_index(drop=True), regime_dummies_val.reset_index(drop=True)],
            axis=1,
        )

        logger.info(
            f"  Added {len(regime_dummies_train.columns)} regime features "
            f"(total: {X_train_augmented.shape[1]})"
        )

        # Create container with augmented features
        from src.core.container import TimeSeriesDataContainer

        augmented_container = TimeSeriesDataContainer.from_dataframes(
            train_df=pd.concat(
                [
                    X_train_augmented,
                    y_train.reset_index(drop=True).to_frame(),
                    (
                        w_train.reset_index(drop=True).to_frame()
                        if w_train is not None
                        else pd.DataFrame()
                    ),
                ],
                axis=1,
            ),
            val_df=pd.concat(
                [
                    X_val_augmented,
                    y_val.reset_index(drop=True).to_frame(),
                ],
                axis=1,
            ),
            horizon=container.horizon,
            feature_columns=list(X_train_augmented.columns),
        )

        # Create trainer config
        model_dir = self.output_dir / f"{model_name}_h{container.horizon}"
        model_dir.mkdir(parents=True, exist_ok=True)

        trainer_config = TrainerConfig(
            model_name=model_name,
            horizon=container.horizon,
            output_dir=model_dir,
        )

        # Train
        trainer = Trainer(trainer_config)
        results = trainer.run(augmented_container, skip_save=not save_models)

        # Store trainer
        self.trainers["combined"] = trainer

        # Save model
        model_path = None
        if save_models:
            model_path = model_dir / "model_regime_feature.pkl"
            trainer.model.save(model_path)
            logger.info(f"  Saved model to: {model_path}")

        total_time = time.time() - model_start

        val_f1 = results["evaluation_metrics"].get("macro_f1", 0)
        val_acc = results["evaluation_metrics"].get("accuracy", 0)

        logger.info(f"  Val F1: {val_f1:.4f}")
        logger.info(f"  Val Accuracy: {val_acc:.4f}")

        # Build single result for combined model
        regime_results = {
            "combined": RegimeTrainingResult(
                regime="combined",
                n_samples=len(X_train),
                sample_fraction=1.0,
                val_f1=val_f1,
                val_accuracy=val_acc,
                model_path=model_path,
                training_time=total_time,
            )
        }

        return RegimeAwareTrainingResult(
            model_name=model_name,
            horizon=container.horizon,
            regime_type=self.regime_config.regime_type,
            regime_results=regime_results,
            aggregated_metrics={"val_f1": val_f1, "val_accuracy": val_acc},
            regime_distribution=regime_dist,
            total_time=total_time,
        )

    def _build_summary(self, results: dict[str, RegimeAwareTrainingResult]) -> dict[str, Any]:
        """Build summary from all results."""
        summary: dict[str, Any] = {
            "n_models": len(results),
            "regime_type": self.regime_config.regime_type,
            "train_separate_models": self.regime_config.train_separate_models,
            "models": {},
        }

        for model_name, result in results.items():
            summary["models"][model_name] = {
                "n_regimes_trained": len(result.regime_results),
                "aggregated_metrics": result.aggregated_metrics,
                "total_time": result.total_time,
            }

        return summary

    def _log_summary(self, summary: dict[str, Any]) -> None:
        """Log summary to console."""
        logger.info("Regime-Aware Training Summary:")
        logger.info(f"  Regime type: {summary['regime_type']}")
        logger.info(f"  Separate models: {summary['train_separate_models']}")

        for model_name, metrics in summary["models"].items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Regimes trained: {metrics['n_regimes_trained']}")
            if metrics["aggregated_metrics"]:
                for metric_name, value in metrics["aggregated_metrics"].items():
                    if isinstance(value, float):
                        logger.info(f"    {metric_name}: {value:.4f}")
                    else:
                        logger.info(f"    {metric_name}: {value}")
            logger.info(f"    Time: {metrics['total_time']:.1f}s")

    def predict(
        self,
        X: pd.DataFrame,
        return_regime: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, pd.Series]:
        """
        Generate predictions using regime-aware models.

        Args:
            X: Features DataFrame
            return_regime: If True, also return detected regimes

        Returns:
            Predictions array, or (predictions, regimes) if return_regime=True
        """
        if not self.trainers:
            raise RuntimeError("No trained models. Call run() first.")

        # Detect regimes
        regimes = detect_regimes(
            X,
            regime_type=self.regime_config.regime_type,
            adx_threshold=self.regime_config.adx_threshold,
            volatility_multiplier=self.regime_config.volatility_multiplier,
        )

        # Initialize predictions
        predictions = np.full(len(X), np.nan)

        if self.regime_config.train_separate_models:
            # Use regime-specific models
            for regime in regimes.unique():
                if regime not in self.trainers:
                    logger.warning(f"No model for regime {regime}, using fallback")
                    continue

                mask = regimes == regime
                trainer = self.trainers[regime]
                regime_preds = trainer.model.predict(X[mask].values)
                predictions[mask] = regime_preds.class_predictions
        else:
            # Use combined model with regime features
            if "combined" not in self.trainers:
                raise RuntimeError("Combined model not found")

            # Add regime features
            regime_dummies = pd.get_dummies(regimes, prefix="regime")
            X_augmented = pd.concat([X.reset_index(drop=True), regime_dummies], axis=1)

            trainer = self.trainers["combined"]
            pred_output = trainer.model.predict(X_augmented.values)
            predictions = pred_output.class_predictions

        if return_regime:
            return predictions, regimes
        return predictions
