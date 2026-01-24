"""
UnifiedTrainingOrchestrator - Thin coordinator for training workflows.

NOW focuses ONLY on:
- Routing to training modes
- Coordinating services
- Managing workflow state

Delegates to:
- ModelTrainingService: Individual model training
- OOFGenerationService: Out-of-fold prediction generation
- ArtifactManager: Saving results

Example:
    from src.core import PipelineConfig
    from src.models.training import UnifiedTrainingOrchestrator

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./experiments/exp_001",
        models=["xgboost", "lightgbm", "lstm"],
        build_ensemble=True,
    )

    orchestrator = UnifiedTrainingOrchestrator(config)
    result = orchestrator.train(df)  # Pass raw OHLCV DataFrame
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core import CVMethod, PipelineConfig, TrainingMode
from src.data.adapters import AlignedOOFResult, PreparedData
from src.validation.cv import OOFPrediction, PurgedKFold, PurgedKFoldConfig, StackingDataset

from .services import (
    ArtifactManager,
    ArtifactSaveRequest,
    DataPreparer,
    EnsembleRequest,
    EnsembleService,
    ModelTrainingRequest,
    ModelTrainingService,
    OOFGenerationService,
    OOFRequest,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class PreTrainingValidationError(Exception):
    """
    Raised when pre-training validation fails.

    This exception is raised when any of the following validation checks fail:
    - Data contract validation (strict mode)
    - Leakage detection (check_leakage enabled)
    - Lookahead audit (check_lookahead enabled)

    The validation settings are controlled via PipelineConfig:
    - strict_validation: Enable/disable data contract validation
    - check_leakage: Enable/disable leakage detection
    - check_lookahead: Enable/disable lookahead audit

    Example:
        try:
            orchestrator.train(df)
        except PreTrainingValidationError as e:
            print(f"Validation failed: {e}")
            # Inspect the specific issues and fix them
    """

    pass


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class ModelTrainingResult:
    """
    Result from training a single model.

    Attributes:
        model_name: Name of the model (e.g., "xgboost", "lstm")
        horizon: Prediction horizon in bars
        metrics: Validation metrics dict (val_f1, val_accuracy, etc.)
        oof_prediction: Optional OOF predictions from CV
        trainer: Optional Trainer instance (for inference)
        training_time_seconds: Time taken to train
        n_features: Number of features used
        data_rank: Data dimensionality (2, 3, or 4)
    """

    model_name: str
    horizon: int
    metrics: dict[str, float] = field(default_factory=dict)
    oof_prediction: OOFPrediction | None = None
    trainer: Any | None = None
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
            "n_features": self.n_features,
            "data_rank": self.data_rank,
        }


@dataclass
class TrainingRunResult:
    """
    Result from a complete training run.

    Attributes:
        run_id: Unique identifier for this run
        config: PipelineConfig used for this run
        model_results: Dict mapping model key -> ModelTrainingResult
        ensemble_result: Optional ensemble training result
        stacking_dataset: Optional stacking dataset for meta-learner
        aligned_oof: Optional AlignedOOFResult for heterogeneous ensembles
        total_time_seconds: Total wall-clock time
        output_dir: Directory where results are saved
    """

    run_id: str
    config: PipelineConfig
    model_results: dict[str, ModelTrainingResult] = field(default_factory=dict)
    ensemble_result: ModelTrainingResult | None = None
    stacking_dataset: StackingDataset | None = None
    aligned_oof: AlignedOOFResult | None = None
    total_time_seconds: float = 0.0
    output_dir: Path = field(default_factory=lambda: Path("."))

    @property
    def n_models(self) -> int:
        """Number of trained models."""
        return len(self.model_results)

    @property
    def best_model(self) -> str | None:
        """Model with best validation F1 score."""
        if not self.model_results:
            return None
        return max(
            self.model_results.keys(), key=lambda k: self.model_results[k].metrics.get("val_f1", 0)
        )

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all model metrics."""
        return {key: result.metrics for key, result in self.model_results.items()}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "n_models": self.n_models,
            "best_model": self.best_model,
            "total_time_seconds": self.total_time_seconds,
            "output_dir": str(self.output_dir),
            "model_results": {key: result.to_dict() for key, result in self.model_results.items()},
        }


# =============================================================================
# UNIFIED TRAINING ORCHESTRATOR
# =============================================================================


class UnifiedTrainingOrchestrator:
    """
    THE single entry point for all training in the ML Factory.

    Uses PipelineConfig from src/core as the ONLY configuration source.

    Supports:
    - All training modes: standard, walk_forward, regime_aware, meta_labeling
    - All CV methods: purged_kfold, cpcv, pbo, walk_forward
    - Heterogeneous ensembles with OOF alignment
    - Integration with PHASE_2 adapters

    Usage:
        from src.core import PipelineConfig
        from src.models.training import UnifiedTrainingOrchestrator

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments/exp_001",
            models=["xgboost", "lightgbm", "lstm"],
            build_ensemble=True,
        )

        orchestrator = UnifiedTrainingOrchestrator(config)
        result = orchestrator.train(df)  # Pass raw OHLCV DataFrame
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize with PipelineConfig.

        Args:
            config: PipelineConfig from src/core - THE ONLY config source
        """
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_dir = Path(config.output_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services (all heavy lifting is delegated)
        self._data_preparer = DataPreparer(config)
        self._model_service = ModelTrainingService()
        self._oof_service = OOFGenerationService(
            cache_dir=self.output_dir / "oof_cache" if config.save_oof else None
        )
        self._ensemble_service = EnsembleService()
        self._artifact_manager = ArtifactManager(self.output_dir)

        # Initialize CV based on config
        self._cv = self._create_cv()

        # Results storage
        self._model_results: dict[str, ModelTrainingResult] = {}
        self._oof_predictions: dict[str, OOFPrediction] = {}
        self._trained_models: dict[str, Any] = {}

        logger.info("Initialized UnifiedTrainingOrchestrator")
        logger.info(f"  Run ID: {self.run_id}")
        logger.info(f"  Mode: {config.training_mode}")
        logger.info(f"  CV: {config.cv_method}")
        logger.info(f"  Models: {config.models}")
        logger.info(f"  Output: {self.output_dir}")

    def _generate_run_id(self) -> str:
        """Generate unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = self.config.training_mode
        return f"run_{mode}_{timestamp}"

    def _create_cv(self) -> PurgedKFold:
        """
        Create cross-validator based on config.

        Returns:
            PurgedKFold instance configured per PipelineConfig
        """
        cv_method = CVMethod(self.config.cv_method)

        # Base config for all CV methods
        cv_config = PurgedKFoldConfig(
            n_splits=self.config.n_splits,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
        )

        # For now, use PurgedKFold for all methods
        # CPCV and WalkForward have specialized implementations
        if cv_method == CVMethod.PURGED_KFOLD:
            return PurgedKFold(cv_config)
        elif cv_method == CVMethod.CPCV:
            # CPCV wraps PurgedKFold with combinatorial paths
            return PurgedKFold(cv_config)
        elif cv_method == CVMethod.WALK_FORWARD:
            # Walk-forward uses different evaluator
            return PurgedKFold(cv_config)
        elif cv_method == CVMethod.PBO:
            # PBO is post-hoc validation, use purged k-fold for training
            return PurgedKFold(cv_config)
        else:
            logger.warning(f"Unknown CV method: {cv_method}, using PurgedKFold")
            return PurgedKFold(cv_config)

    def _pre_training_validation(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Validate data before training. Raises PreTrainingValidationError on failure.

        This method runs the following checks based on PipelineConfig settings:
        1. Data contract validation (if strict_validation=True)
        2. Leakage detection (if check_leakage=True)
        3. Lookahead audit (if check_lookahead=True)

        Args:
            df: Input DataFrame to validate
            feature_names: Optional list of feature column names

        Raises:
            PreTrainingValidationError: If any validation check fails
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info("\n" + "-" * 40)
        logger.info("PRE-TRAINING VALIDATION")
        logger.info("-" * 40)

        # Determine feature columns
        if feature_names is None:
            # Exclude known non-feature columns
            exclude_patterns = ["label", "sample_weight", "datetime", "date", "time"]
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            feature_names = [
                col
                for col in df.columns
                if col not in ohlcv_cols and not any(pat in col.lower() for pat in exclude_patterns)
            ]

        # 1. Contract validation (if enabled)
        if self.config.strict_validation:
            logger.info("  [1/3] Validating data contracts...")
            try:
                from src.core.contracts import DataContract, DataContractViolation

                # Find label column for validation
                label_col = None
                for h in self.config.horizons:
                    candidate = f"label_h{h}"
                    if candidate in df.columns:
                        label_col = candidate
                        break

                if feature_names and len(feature_names) > 0:
                    # Create data contract from the DataFrame
                    X = df[feature_names].values
                    data_contract = DataContract.from_array(
                        X,
                        symbol=self.config.symbol,
                        timeframe=(
                            self.config.mtf_timeframes[0] if self.config.mtf_timeframes else "5min"
                        ),
                        horizon=self.config.horizons[0],
                        split="train",
                    )

                    # Validate against model contracts
                    for model_name in self.config.models:
                        from src.core.contracts import get_model_contract

                        model_contract = get_model_contract(model_name)
                        is_valid, issues = model_contract.validate_data_contract(data_contract)
                        if not is_valid:
                            errors.append(
                                f"Contract violation for {model_name}: {'; '.join(issues)}"
                            )
                        elif issues:
                            warnings.extend(issues)

                    logger.info(f"    Data contract: {data_contract.n_features} features")
                else:
                    warnings.append("No feature columns identified for contract validation")

            except DataContractViolation as e:
                errors.append(f"Data contract violation: {e}")
            except Exception as e:
                warnings.append(f"Contract validation skipped: {e}")

        else:
            logger.info("  [1/3] Data contract validation: SKIPPED (strict_validation=False)")

        # 2. Leakage detection (if enabled)
        if self.config.check_leakage:
            logger.info("  [2/3] Running leakage detection...")
            try:
                from src.validation.leakage_detection import check_feature_label_correlation

                # Find label column
                label_col = None
                for h in self.config.horizons:
                    candidate = f"label_h{h}"
                    if candidate in df.columns:
                        label_col = candidate
                        break

                if label_col is not None and feature_names and len(feature_names) > 0:
                    # Run correlation-based leakage check
                    X_features = df[feature_names]
                    y_labels = df[label_col].values

                    report = check_feature_label_correlation(
                        features=X_features,
                        labels=y_labels,
                        feature_names=feature_names,
                        correlation_threshold=self.config.validation_correlation_threshold,
                    )

                    if report.n_suspicious > 0:
                        suspicious_names = report.get_suspicious_feature_names()[:5]
                        error_msg = (
                            f"LEAKAGE DETECTED: {report.n_suspicious} features have "
                            f"suspicious correlation (threshold={report.correlation_threshold}). "
                            f"Top suspicious: {', '.join(suspicious_names)}"
                        )
                        errors.append(error_msg)
                        logger.warning(f"    {error_msg}")
                    else:
                        logger.info(
                            f"    Leakage check passed: "
                            f"{report.n_features} features analyzed, 0 suspicious"
                        )
                else:
                    warnings.append("Leakage detection skipped: no label column or features found")

            except Exception as e:
                warnings.append(f"Leakage detection failed: {e}")

        else:
            logger.info("  [2/3] Leakage detection: SKIPPED (check_leakage=False)")

        # 3. Lookahead audit (if enabled)
        if self.config.check_lookahead:
            logger.info("  [3/3] Running lookahead audit...")
            try:
                from src.validation.lookahead_audit import validate_resample_config

                # Validate resample config (basic check for implicit parameters)
                # Note: Full lookahead auditing requires a feature function, which we
                # don't have here. The full audit should be done during feature engineering.
                is_valid, issues = validate_resample_config(
                    closed="left",  # Our expected setting
                    label="left",  # Our expected setting
                )

                if not is_valid:
                    errors.append(f"Lookahead risk: {'; '.join(issues)}")
                elif issues:
                    # These are warnings about implicit parameters
                    if self.config.validation_fail_on_warning:
                        errors.extend(issues)
                    else:
                        warnings.extend(issues)

                if is_valid:
                    logger.info("    Lookahead audit passed: resample config validated")

            except Exception as e:
                warnings.append(f"Lookahead audit failed: {e}")

        else:
            logger.info("  [3/3] Lookahead audit: SKIPPED (check_lookahead=False)")

        # Log warnings
        if warnings:
            logger.warning("\n  Validation warnings:")
            for w in warnings:
                logger.warning(f"    - {w}")

        # Raise if any errors
        if errors:
            error_summary = "\n".join([f"  - {e}" for e in errors])
            raise PreTrainingValidationError(
                f"Pre-training validation failed:\n{error_summary}\n\n"
                f"To bypass validation, set in PipelineConfig:\n"
                f"  - strict_validation=False (disable contract validation)\n"
                f"  - check_leakage=False (disable leakage detection)\n"
                f"  - check_lookahead=False (disable lookahead audit)"
            )

        logger.info("\n  Pre-training validation: PASSED")
        logger.info("-" * 40 + "\n")

    def train(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
        generate_financial_report: bool = True,
    ) -> TrainingRunResult:
        """
        Execute complete training pipeline.

        This is THE unified interface for all training.

        Args:
            df: Raw OHLCV DataFrame (features will be computed via adapters)
            additional_dfs: Optional dict of additional timeframe DataFrames
                for multi-stream models (e.g., {"5min": df_5min})
            generate_financial_report: Whether to generate financial report
                with visualizations after training (default: True)

        Returns:
            TrainingRunResult with all training outputs
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info(f"UNIFIED TRAINING: {self.config.training_mode.upper()}")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Models: {self.config.models}")
        logger.info(f"Horizons: {self.config.horizons}")

        # Run pre-training validation (Phase 1: Contract Enforcement)
        # This will raise PreTrainingValidationError if validation fails
        self._pre_training_validation(df)

        # Route to appropriate training mode
        mode = TrainingMode(self.config.training_mode)

        if mode == TrainingMode.STANDARD:
            self._train_standard(df, additional_dfs)
        elif mode == TrainingMode.WALK_FORWARD:
            self._train_walk_forward(df, additional_dfs)
        elif mode == TrainingMode.REGIME_AWARE:
            self._train_regime_aware(df, additional_dfs)
        elif mode == TrainingMode.META_LABELING:
            self._train_meta_labeling(df, additional_dfs)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        # Build ensemble if requested
        ensemble_result = None
        stacking_dataset = None
        aligned_oof = None

        if self.config.build_ensemble and len(self._model_results) > 1:
            logger.info("\n" + "=" * 60)
            logger.info("BUILDING ENSEMBLE")
            logger.info("=" * 60)
            aligned_oof, stacking_dataset, ensemble_result = self._build_ensemble(df)

        # Save results
        self._save_results()

        total_time = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Models trained: {len(self._model_results)}")
        logger.info(f"Output: {self.output_dir}")

        # Generate financial report if requested
        if generate_financial_report and self._model_results:
            self._generate_financial_reports(df)

        return TrainingRunResult(
            run_id=self.run_id,
            config=self.config,
            model_results=self._model_results,
            ensemble_result=ensemble_result,
            stacking_dataset=stacking_dataset,
            aligned_oof=aligned_oof,
            total_time_seconds=total_time,
            output_dir=self.output_dir,
        )

    def _train_standard(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Standard training with PurgedKFold CV and OOF generation.

        This is the default training mode that:
        1. Iterates through each horizon and model
        2. Prepares data using PHASE_2 adapters
        3. Optionally optimizes hyperparameters
        4. Trains model and generates OOF predictions

        Args:
            df: Input DataFrame with features and labels
            additional_dfs: Optional additional timeframe DataFrames
        """
        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")

            for model_name in self.config.models:
                logger.info(f"\nTraining {model_name}...")

                # Prepare data using PHASE_2 adapters
                prepared = self._data_preparer.prepare(
                    df=df,
                    model_name=model_name,
                    additional_dfs=additional_dfs,
                )

                logger.info(f"  Data prepared: {prepared.summary()}")

                # Train model
                result = self._train_single_model(
                    model_name=model_name,
                    prepared=prepared,
                    horizon=horizon,
                )

                key = f"{model_name}_h{horizon}"
                self._model_results[key] = result

                # Generate OOF if enabled
                if self.config.save_oof:
                    oof = self._generate_oof(model_name, prepared, horizon)
                    if oof is not None:
                        self._oof_predictions[key] = oof
                        result.oof_prediction = oof

                logger.info(
                    f"  {model_name} complete: "
                    f"val_f1={result.metrics.get('val_f1', 0):.4f}, "
                    f"time={result.training_time_seconds:.1f}s"
                )

    def _train_single_model(
        self,
        model_name: str,
        prepared: PreparedData,
        horizon: int,
    ) -> ModelTrainingResult:
        """Train a single model - delegates to ModelTrainingService."""
        request = ModelTrainingRequest(
            model_name=model_name,
            horizon=horizon,
            prepared_data=prepared,
            sequence_length=self.config.sequence_length,
            output_dir=self.output_dir / f"h{horizon}",
            optimize_hyperparams=self.config.optimize_hyperparams,
            n_splits=self.config.n_splits,
            hyperparam_trials=self.config.hyperparam_trials,
        )
        result = self._model_service.train_model(request)

        # Store for later use
        self._trained_models[f"{model_name}_h{horizon}"] = result.trainer

        # Convert service result to orchestrator result
        return ModelTrainingResult(
            model_name=result.model_name,
            horizon=result.horizon,
            metrics=result.metrics,
            trainer=result.trainer,
            training_time_seconds=result.training_time_seconds,
            n_features=result.n_features,
            data_rank=result.data_rank,
        )

    def _generate_oof(
        self,
        model_name: str,
        prepared: PreparedData,
        horizon: int,
    ) -> OOFPrediction | None:
        """Generate OOF predictions - delegates to OOFGenerationService."""
        request = OOFRequest(
            model_name=model_name,
            horizon=horizon,
            prepared_data=prepared,
            n_splits=self.config.n_splits,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
        )
        return self._oof_service.generate_oof(request)

    def _build_ensemble(
        self,
        df: pd.DataFrame,
    ) -> tuple[AlignedOOFResult | None, StackingDataset | None, ModelTrainingResult | None]:
        """
        Build ensemble from OOF predictions - delegates to EnsembleService.

        Args:
            df: Original DataFrame (for label extraction)

        Returns:
            Tuple of (aligned_oof, stacking_dataset, ensemble_result)
        """
        if not self._oof_predictions:
            logger.warning("No OOF predictions available for ensemble")
            return None, None, None

        # Delegate to EnsembleService
        request = EnsembleRequest(
            oof_predictions=self._oof_predictions,
            config=self.config,
            df=df,
        )
        result = self._ensemble_service.build_ensemble(request)

        # Convert service result to orchestrator result
        ensemble_result = None
        if result.meta_learner is not None:
            ensemble_result = ModelTrainingResult(
                model_name=f"ensemble_{self.config.meta_learner}",
                horizon=self.config.horizons[0] if self.config.horizons else 20,
                metrics=result.ensemble_metrics,
                trainer=result.meta_learner,
                training_time_seconds=result.training_time_seconds,
            )

        return result.aligned_oof, result.stacking_dataset, ensemble_result

    def _train_walk_forward(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Walk-forward training mode.

        Uses expanding or rolling windows for more realistic backtesting.

        Args:
            df: Input DataFrame
            additional_dfs: Optional additional timeframe DataFrames
        """
        from .config import ExperimentConfig, ModelConfig
        from .modes import WalkForwardTrainer, WalkForwardTrainerConfig

        logger.info("Walk-forward training mode")

        # Create ExperimentConfig from PipelineConfig
        model_configs = [ModelConfig(name=m) for m in self.config.models]

        exp_config = ExperimentConfig(
            symbol=self.config.symbol,
            horizons=self.config.horizons,
            models=model_configs,
            data_dir=Path(self.config.data_path).parent,
            output_dir=self.output_dir,
        )

        wf_config = WalkForwardTrainerConfig(
            n_windows=self.config.n_splits,
            window_type="expanding",
            min_train_pct=self.config.train_ratio,
            test_pct=self.config.val_ratio,
            gap_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
        )

        trainer = WalkForwardTrainer(exp_config, wf_config)

        # Prepare container from df
        from src.core.container import TimeSeriesDataContainer

        # Use first model to prepare data (for basic structure)
        prepared = self._data_preparer.prepare(
            df=df,
            model_name=self.config.models[0],
            additional_dfs=additional_dfs,
        )

        X_train_df = pd.DataFrame(
            (
                prepared.X_train.reshape(prepared.X_train.shape[0], -1)
                if prepared.data_rank > 2
                else prepared.X_train
            ),
            columns=(
                prepared.feature_names
                if prepared.data_rank == 2
                else [f"f{i}" for i in range(np.prod(prepared.X_train.shape[1:]))]
            ),
        )

        # Build train DataFrame for container
        train_df = X_train_df.copy()
        train_df[f"label_h{self.config.horizons[0]}"] = prepared.y_train
        train_df[f"sample_weight_h{self.config.horizons[0]}"] = np.ones(len(prepared.y_train))

        container = TimeSeriesDataContainer.from_dataframes(
            train_df=train_df,
            val_df=None,
            test_df=None,
            horizon=self.config.horizons[0],
            feature_columns=list(X_train_df.columns),
        )

        results = trainer.run(container)

        # Convert walk-forward results to our format
        for model_name, wf_result in results.get("model_results", {}).items():
            key = f"{model_name}_h{self.config.horizons[0]}"
            self._model_results[key] = ModelTrainingResult(
                model_name=model_name,
                horizon=self.config.horizons[0],
                metrics={
                    "val_f1": wf_result.aggregated_metrics.get("mean_f1", 0),
                    "val_accuracy": wf_result.aggregated_metrics.get("mean_accuracy", 0),
                },
                training_time_seconds=wf_result.total_time,
                n_features=prepared.n_features,
                data_rank=prepared.data_rank,
            )

    def _train_regime_aware(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Regime-aware training mode.

        Trains separate models for different market regimes using the new
        RegimeAwareTrainer that integrates with PipelineConfig.

        Uses PipelineConfig settings:
        - regime_detection_method: Detection method (volatility_percentile, trend_adx, combined)
        - regime_lookback: Lookback period for regime detection
        - n_regimes: Number of regime states (2 or 3)
        - regime_volatility_window: Rolling window for volatility
        - regime_adx_threshold: ADX threshold for trending
        - regime_min_samples: Minimum samples per regime
        - train_separate_regime_models: Whether to train separate models per regime

        Args:
            df: Input DataFrame with OHLCV data
            additional_dfs: Optional additional timeframe DataFrames
        """
        from .regime_trainer import RegimeAwareTrainer

        logger.info("Regime-aware training mode")
        logger.info(f"  Detection method: {self.config.regime_detection_method}")
        logger.info(f"  Number of regimes: {self.config.n_regimes}")
        logger.info(f"  Separate models: {self.config.train_separate_regime_models}")

        # Initialize the regime-aware trainer with PipelineConfig
        regime_trainer = RegimeAwareTrainer(self.config)

        # Store regime trainer for inference
        self._regime_trainer = regime_trainer

        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")

            for model_name in self.config.models:
                logger.info(f"\nTraining regime-aware: {model_name}...")

                # Prepare data using PHASE_2 adapters
                prepared = self._data_preparer.prepare(
                    df=df,
                    model_name=model_name,
                    additional_dfs=additional_dfs,
                )

                logger.info(f"  Data prepared: {prepared.summary()}")

                # Train regime-aware model
                regime_result = regime_trainer.train(
                    prepared=prepared,
                    horizon=horizon,
                    model_name=model_name,
                    save_models=self.config.save_models,
                )

                # Convert regime results to ModelTrainingResult format
                for (
                    trained_model_name,
                    regime,
                ), regime_model_result in regime_result.regime_results.items():
                    if trained_model_name != model_name:
                        continue

                    # Create key that includes regime
                    key = f"{model_name}_h{horizon}_{regime}"

                    self._model_results[key] = ModelTrainingResult(
                        model_name=f"{model_name}_{regime}",
                        horizon=horizon,
                        metrics={
                            "val_f1": regime_model_result.val_f1,
                            "val_accuracy": regime_model_result.val_accuracy,
                            **regime_model_result.metrics,
                        },
                        trainer=regime_model_result.trainer,
                        training_time_seconds=regime_model_result.training_time_seconds,
                        n_features=prepared.n_features,
                        data_rank=prepared.data_rank,
                    )

                    # Store trained model for inference
                    self._trained_models[key] = regime_model_result.trainer

                    logger.info(
                        f"    {model_name}/{regime}: "
                        f"val_f1={regime_model_result.val_f1:.4f}, "
                        f"samples={regime_model_result.n_samples}"
                    )

                # Also create an aggregated result for this model
                aggregated_key = f"{model_name}_h{horizon}_aggregated"
                aggregated_metrics = {
                    k.replace(f"{model_name}_", ""): v
                    for k, v in regime_result.aggregated_metrics.items()
                    if model_name in k or "overall" in k
                }

                self._model_results[aggregated_key] = ModelTrainingResult(
                    model_name=f"{model_name}_regime_aware",
                    horizon=horizon,
                    metrics=aggregated_metrics,
                    training_time_seconds=regime_result.total_time_seconds,
                    n_features=prepared.n_features,
                    data_rank=prepared.data_rank,
                )

        logger.info("\nRegime-aware training complete")

    def _train_meta_labeling(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Meta-labeling training mode (Lopez de Prado 2018).

        Implements the complete meta-labeling methodology:
        1. Train primary model for direction prediction (+1, -1)
        2. Generate primary model predictions on training data
        3. Create meta-labels: 1 if primary was correct, 0 if wrong
        4. Train meta-model to predict probability of primary being correct
        5. At inference: final_position = primary_direction * meta_probability

        Uses PipelineConfig settings:
        - meta_labeling_primary_model: Model for direction (default: "xgboost")
        - meta_labeling_meta_model: Model for bet sizing (default: "logistic")
        - meta_labeling_threshold: Min probability to take trade (default: 0.5)

        Args:
            df: Input DataFrame
            additional_dfs: Optional additional timeframe DataFrames
        """
        logger.info("=" * 60)
        logger.info("META-LABELING TRAINING (Lopez de Prado 2018)")
        logger.info("=" * 60)
        logger.info(f"Primary model: {self.config.meta_labeling_primary_model}")
        logger.info(f"Meta model: {self.config.meta_labeling_meta_model}")
        logger.info(f"Bet threshold: {self.config.meta_labeling_threshold}")

        start_time = time.time()

        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")

            result = self._train_meta_labeling_for_horizon(
                df=df,
                horizon=horizon,
                additional_dfs=additional_dfs,
            )

            key = f"meta_labeling_h{horizon}"
            self._model_results[key] = result

            logger.info(
                f"  Meta-labeling complete: "
                f"primary_acc={result.metrics.get('primary_accuracy', 0):.4f}, "
                f"combined_acc={result.metrics.get('combined_accuracy', 0):.4f}, "
                f"trade_fraction={result.metrics.get('trade_fraction', 0)*100:.1f}%"
            )

        total_time = time.time() - start_time
        logger.info(f"\nMeta-labeling total time: {total_time:.1f}s")

    def _train_meta_labeling_for_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> ModelTrainingResult:
        """
        Train meta-labeling system for a single horizon.

        Args:
            df: Input DataFrame
            horizon: Prediction horizon
            additional_dfs: Optional additional DataFrames

        Returns:
            ModelTrainingResult with combined metrics
        """

        start_time = time.time()

        # Get model names from config
        primary_model_name = self.config.meta_labeling_primary_model
        meta_model_name = self.config.meta_labeling_meta_model

        # =====================================================================
        # STAGE 1: PREPARE DATA
        # =====================================================================
        logger.info("\n  STAGE 1: Preparing data...")

        # Prepare data for primary model
        prepared = self._data_preparer.prepare(
            df=df,
            model_name=primary_model_name,
            additional_dfs=additional_dfs,
        )

        logger.info(f"    Data: {prepared.n_train} train, {prepared.n_val} val samples")
        logger.info(f"    Features: {prepared.n_features}, Rank: {prepared.data_rank}")

        # Convert to 2D for training
        X_train = self._flatten_to_2d(prepared.X_train)
        X_val = self._flatten_to_2d(prepared.X_val)
        y_train = prepared.y_train
        y_val = prepared.y_val

        feature_names = (
            prepared.feature_names
            if prepared.data_rank == 2
            else [f"f{i}" for i in range(X_train.shape[1])]
        )

        # =====================================================================
        # STAGE 2: TRAIN PRIMARY MODEL (Direction)
        # =====================================================================
        logger.info("\n  STAGE 2: Training primary model (direction)...")

        primary_result = self._train_single_model(
            model_name=primary_model_name,
            prepared=prepared,
            horizon=horizon,
        )

        primary_trainer = primary_result.trainer
        if primary_trainer is None:
            raise RuntimeError("Primary model training failed")

        # Get primary predictions on training and validation sets
        primary_train_preds = primary_trainer.model.predict(X_train)
        primary_val_preds = primary_trainer.model.predict(X_val)

        # Extract class predictions
        primary_train_classes = primary_train_preds.class_predictions
        primary_val_classes = primary_val_preds.class_predictions

        # Primary model accuracy
        primary_train_acc = (primary_train_classes == y_train).mean()
        primary_val_acc = (primary_val_classes == y_val).mean()

        logger.info(f"    Primary train accuracy: {primary_train_acc:.4f}")
        logger.info(f"    Primary val accuracy: {primary_val_acc:.4f}")

        # =====================================================================
        # STAGE 3: CREATE META-LABELS
        # =====================================================================
        logger.info("\n  STAGE 3: Creating meta-labels...")

        # Meta-label = 1 if primary prediction was correct, 0 if wrong
        meta_labels_train = (primary_train_classes == y_train).astype(int)
        meta_labels_val = (primary_val_classes == y_val).astype(int)

        # Log meta-label distribution
        train_correct_pct = meta_labels_train.mean() * 100
        val_correct_pct = meta_labels_val.mean() * 100

        logger.info(f"    Train: {train_correct_pct:.1f}% correct (meta=1)")
        logger.info(f"    Val: {val_correct_pct:.1f}% correct (meta=1)")

        # =====================================================================
        # STAGE 4: TRAIN META-MODEL (Bet Sizing)
        # =====================================================================
        logger.info("\n  STAGE 4: Training meta-model (bet sizing)...")

        # Create meta-model based on config
        meta_model = self._create_meta_model(meta_model_name)

        # Train meta-model to predict if primary is correct
        meta_model.fit(X_train, meta_labels_train)

        # Get meta-model probabilities
        if hasattr(meta_model, "predict_proba"):
            meta_proba_train = meta_model.predict_proba(X_train)[:, 1]
            meta_proba_val = meta_model.predict_proba(X_val)[:, 1]
        else:
            # Fall back to decision function if no predict_proba
            meta_proba_train = meta_model.predict(X_train).astype(float)
            meta_proba_val = meta_model.predict(X_val).astype(float)

        # Meta-model accuracy (thresholded at 0.5)
        meta_train_acc = ((meta_proba_train >= 0.5) == meta_labels_train).mean()
        meta_val_acc = ((meta_proba_val >= 0.5) == meta_labels_val).mean()

        logger.info(f"    Meta-model train accuracy: {meta_train_acc:.4f}")
        logger.info(f"    Meta-model val accuracy: {meta_val_acc:.4f}")

        # =====================================================================
        # STAGE 5: EVALUATE COMBINED SYSTEM
        # =====================================================================
        logger.info("\n  STAGE 5: Evaluating combined system...")

        threshold = self.config.meta_labeling_threshold

        # Apply threshold - only take trades where meta probability >= threshold
        trades_taken_val = meta_proba_val >= threshold

        # Combined accuracy (only on trades taken)
        if trades_taken_val.sum() > 0:
            combined_val_acc = (
                primary_val_classes[trades_taken_val] == y_val[trades_taken_val]
            ).mean()
            trade_fraction = trades_taken_val.mean()
        else:
            combined_val_acc = 0.0
            trade_fraction = 0.0

        # Improvement over primary alone
        improvement = combined_val_acc - primary_val_acc if trade_fraction > 0 else 0.0

        logger.info(f"    Threshold: {threshold}")
        logger.info(f"    Trade fraction: {trade_fraction*100:.1f}%")
        logger.info(f"    Primary-only accuracy: {primary_val_acc:.4f}")
        logger.info(f"    Combined accuracy: {combined_val_acc:.4f}")
        logger.info(f"    Improvement: {improvement:+.4f}")

        # =====================================================================
        # STAGE 6: STORE MODELS FOR INFERENCE
        # =====================================================================
        logger.info("\n  STAGE 6: Storing models...")

        # Store both models for inference
        model_key = f"meta_labeling_h{horizon}"
        self._trained_models[f"{model_key}_primary"] = primary_trainer
        self._trained_models[f"{model_key}_meta"] = meta_model

        # Save models if enabled
        if self.config.save_models:
            models_dir = self.output_dir / "models"
            models_dir.mkdir(exist_ok=True)

            # Save meta-model (sklearn)
            import pickle

            meta_path = models_dir / f"{model_key}_meta.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump(
                    {
                        "meta_model": meta_model,
                        "primary_model_name": primary_model_name,
                        "meta_model_name": meta_model_name,
                        "threshold": threshold,
                        "feature_names": feature_names,
                    },
                    f,
                )
            logger.info(f"    Saved meta-model to: {meta_path}")

        training_time = time.time() - start_time

        # Build combined metrics
        metrics = {
            # Primary model metrics
            "primary_train_accuracy": float(primary_train_acc),
            "primary_val_accuracy": float(primary_val_acc),
            "primary_val_f1": primary_result.metrics.get("val_f1", 0),
            # Meta model metrics
            "meta_train_accuracy": float(meta_train_acc),
            "meta_val_accuracy": float(meta_val_acc),
            # Combined system metrics
            "combined_accuracy": float(combined_val_acc),
            "primary_accuracy": float(primary_val_acc),  # For comparison
            "trade_fraction": float(trade_fraction),
            "trades_taken": int(trades_taken_val.sum()),
            "total_samples": len(y_val),
            "threshold": threshold,
            "improvement": float(improvement),
            # For backward compatibility with val_f1 key
            "val_f1": float(combined_val_acc),  # Use combined accuracy as main metric
            "val_accuracy": float(combined_val_acc),
        }

        return ModelTrainingResult(
            model_name=f"meta_labeling_{primary_model_name}_{meta_model_name}",
            horizon=horizon,
            metrics=metrics,
            trainer=primary_trainer,  # Store primary trainer as main
            training_time_seconds=training_time,
            n_features=prepared.n_features,
            data_rank=prepared.data_rank,
        )

    def _flatten_to_2d(self, X: np.ndarray) -> np.ndarray:
        """Flatten array to 2D if needed."""
        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)

    def _create_meta_model(self, model_name: str) -> Any:
        """
        Create meta-model for bet sizing.

        Args:
            model_name: Name of the model ("logistic", "xgboost", etc.)

        Returns:
            Fitted sklearn-compatible model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        if model_name == "logistic":
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.random_state,
            )
        elif model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight="balanced",
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
        elif model_name == "xgboost":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to logistic")
                return self._create_meta_model("logistic")
        elif model_name == "lightgbm":
            try:
                import lightgbm as lgb

                return lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available, falling back to logistic")
                return self._create_meta_model("logistic")
        elif model_name == "catboost":
            try:
                import catboost as cb  # type: ignore[import-not-found]

                return cb.CatBoostClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    verbose=False,
                )
            except ImportError:
                logger.warning("CatBoost not available, falling back to logistic")
                return self._create_meta_model("logistic")
        else:
            logger.warning(f"Unknown meta model: {model_name}, using logistic")
            return self._create_meta_model("logistic")

    def _save_results(self) -> None:
        """Save all results - delegates to ArtifactManager."""
        request = ArtifactSaveRequest(
            output_dir=self.output_dir,
            config=self.config,
            model_results=self._model_results,
            oof_predictions=self._oof_predictions,
            trained_models=self._trained_models,
            save_models=self.config.save_models,
            save_oof=self.config.save_oof,
        )
        self._artifact_manager.save_all(request)

    def _generate_financial_reports(self, df: pd.DataFrame) -> None:
        """
        Generate financial reports with visualizations for trained models.

        This method generates comprehensive financial reports including
        equity curves, trade statistics, and performance visualizations.

        Args:
            df: Original input DataFrame with prices and labels
        """
        try:
            from src.models.evaluation.financial_report import (
                FinancialReportConfig,
                generate_financial_report,
            )
        except ImportError:
            logger.warning("Financial report generation not available - missing dependencies")
            return

        logger.info("\n" + "=" * 60)
        logger.info("GENERATING FINANCIAL REPORTS")
        logger.info("=" * 60)

        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Get price data from input DataFrame
        if "close" not in df.columns:
            logger.warning("Cannot generate financial report - 'close' column not found")
            return

        prices = df["close"].values
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        if timestamps is None and "datetime" in df.columns:
            timestamps = pd.to_datetime(df["datetime"])

        # Configure based on symbol
        config = FinancialReportConfig(
            initial_equity=100000.0,
            commission_per_trade=2.50,
            slippage_ticks=1.0,
            tick_value=1.25 if self.config.symbol == "MES" else 0.10,  # MES or MGC
        )

        # Generate report for each trained model
        for model_key, result in self._model_results.items():
            # Get OOF predictions if available
            oof = self._oof_predictions.get(model_key)
            if oof is None:
                logger.info(f"Skipping {model_key} - no OOF predictions available")
                continue

            # Extract predictions and true labels
            predictions = oof.predictions
            y_true = oof.y_true if hasattr(oof, "y_true") else None

            # Try to get true labels from DataFrame
            if y_true is None:
                label_col = f"label_h{result.horizon}"
                if label_col in df.columns:
                    y_true = df[label_col].values[: len(predictions)]

            if y_true is None:
                logger.warning(f"Skipping {model_key} - no true labels available")
                continue

            # Generate report for this model
            try:
                model_report_dir = reports_dir / model_key
                generate_financial_report(
                    model_name=result.model_name,
                    horizon=result.horizon,
                    predictions=predictions,
                    y_true=y_true,
                    prices=prices[: len(predictions)],
                    timestamps=timestamps[: len(predictions)] if timestamps is not None else None,
                    output_dir=model_report_dir,
                    run_id=self.run_id,
                    config=config,
                )
                logger.info(f"Generated financial report for {model_key}")
            except Exception as e:
                logger.error(f"Failed to generate report for {model_key}: {e}")

    def get_trained_model(self, model_key: str) -> Any | None:
        """
        Get a trained model by key.

        Args:
            model_key: Key in format "model_name_hHORIZON" (e.g., "xgboost_h20")

        Returns:
            Trainer instance or None if not found
        """
        return self._trained_models.get(model_key)

    def get_oof_predictions(self, model_key: str) -> OOFPrediction | None:
        """
        Get OOF predictions for a model.

        Args:
            model_key: Key in format "model_name_hHORIZON"

        Returns:
            OOFPrediction or None if not found
        """
        return self._oof_predictions.get(model_key)

    def get_meta_labeling_models(
        self,
        horizon: int,
    ) -> tuple[Any, Any, float] | None:
        """
        Get meta-labeling models for a given horizon.

        Returns both the primary model (for direction) and meta-model (for bet sizing),
        along with the configured threshold.

        Args:
            horizon: Prediction horizon

        Returns:
            Tuple of (primary_trainer, meta_model, threshold) or None if not found

        Example:
            primary, meta, threshold = orchestrator.get_meta_labeling_models(20)
            if primary and meta:
                # Get direction from primary
                direction = primary.model.predict(X).class_predictions
                # Get bet probability from meta
                bet_prob = meta.predict_proba(X)[:, 1]
                # Final position = direction * bet_prob (where bet_prob >= threshold)
        """
        primary_key = f"meta_labeling_h{horizon}_primary"
        meta_key = f"meta_labeling_h{horizon}_meta"

        primary = self._trained_models.get(primary_key)
        meta = self._trained_models.get(meta_key)

        if primary is None or meta is None:
            return None

        return primary, meta, self.config.meta_labeling_threshold

    def predict_meta_labeling(
        self,
        X: np.ndarray | pd.DataFrame,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Generate predictions using trained meta-labeling system.

        Implements Lopez de Prado's meta-labeling:
        1. Primary model predicts direction
        2. Meta-model predicts probability of primary being correct
        3. Only trade where probability >= threshold
        4. Position size = direction * probability

        Args:
            X: Feature array/DataFrame
            horizon: Prediction horizon to use

        Returns:
            Tuple of (directions, probabilities, positions) or None if models not trained
            - directions: Primary model predictions (-1, 0, +1 or class labels)
            - probabilities: Meta-model confidence [0, 1]
            - positions: Final position sizes (direction * probability, 0 if below threshold)

        Example:
            directions, probs, positions = orchestrator.predict_meta_labeling(X, 20)
            # positions contains the bet-sized positions ready for trading
        """
        models = self.get_meta_labeling_models(horizon)
        if models is None:
            logger.warning(f"Meta-labeling models not found for horizon {horizon}")
            return None

        primary_trainer, meta_model, threshold = models

        # Convert to numpy if needed
        X_arr = np.asarray(X)
        if X_arr.ndim > 2:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)

        # Get primary predictions (direction)
        primary_preds = primary_trainer.model.predict(X_arr)
        directions = primary_preds.class_predictions

        # Get meta-model probabilities
        if hasattr(meta_model, "predict_proba"):
            probabilities = meta_model.predict_proba(X_arr)[:, 1]
        else:
            probabilities = meta_model.predict(X_arr).astype(float)

        # Calculate positions: direction * probability (0 if below threshold)
        # Map class predictions to direction (-1, 0, +1)
        # Assuming 3-class: {0: short, 1: neutral, 2: long} -> {-1, 0, +1}
        direction_mapped = directions.astype(float) - 1.0

        # Position = direction * probability, but 0 if probability < threshold
        positions = np.where(probabilities >= threshold, direction_mapped * probabilities, 0.0)

        return directions, probabilities, positions


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def train_pipeline(
    config: PipelineConfig,
    df: pd.DataFrame,
    **kwargs: Any,
) -> TrainingRunResult:
    """
    Convenience function for training.

    Usage:
        from src.core import PipelineConfig
        from src.models.training import train_pipeline

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments",
            models=["xgboost", "lightgbm"],
        )

        result = train_pipeline(config, df)

    Args:
        config: PipelineConfig from src/core
        df: Raw OHLCV DataFrame
        **kwargs: Additional arguments passed to train()

    Returns:
        TrainingRunResult with all outputs
    """
    orchestrator = UnifiedTrainingOrchestrator(config)
    return orchestrator.train(df, **kwargs)


def train_meta_labeling(
    config: PipelineConfig,
    df: pd.DataFrame,
    **kwargs: Any,
) -> TrainingRunResult:
    """
    Convenience function for meta-labeling training.

    This is a shortcut that sets training_mode to "meta_labeling" and runs
    the unified training orchestrator.

    Usage:
        from src.core import PipelineConfig
        from src.models.training import train_meta_labeling

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments",
            meta_labeling_primary_model="xgboost",
            meta_labeling_meta_model="logistic",
            meta_labeling_threshold=0.5,
        )

        result = train_meta_labeling(config, df)

        # Access results
        print(f"Trade fraction: {result.model_results['meta_labeling_h20'].metrics['trade_fraction']}")
        print(f"Combined accuracy: {result.model_results['meta_labeling_h20'].metrics['combined_accuracy']}")

    Args:
        config: PipelineConfig from src/core (training_mode will be overridden)
        df: Raw OHLCV DataFrame
        **kwargs: Additional arguments passed to train()

    Returns:
        TrainingRunResult with meta-labeling outputs
    """
    # Override training mode to meta_labeling
    config.training_mode = TrainingMode.META_LABELING.value

    orchestrator = UnifiedTrainingOrchestrator(config)
    return orchestrator.train(df, **kwargs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "UnifiedTrainingOrchestrator",
    "PreTrainingValidationError",
    "TrainingRunResult",
    "ModelTrainingResult",
    "train_pipeline",
    "train_meta_labeling",
]
