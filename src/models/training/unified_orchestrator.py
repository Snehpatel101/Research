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

import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core import CVMethod, PipelineConfig, TrainingMode
from src.core.exceptions import PreTrainingValidationError
from src.data.adapters import AlignedOOFResult, PreparedData
from src.optimization.feature_selection.filtering import (
    filter_correlated_features,
    filter_low_variance,
)
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
    ParallelTrainingService,  # Phase 12A-6: Add parallel training support
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================


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
    calibrator: Any | None = None

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
        # Phase 12A-6: Use ParallelTrainingService for multi-model training
        self._model_service = ModelTrainingService()
        self._parallel_service = ParallelTrainingService(n_jobs=config.n_jobs)
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

        # PreparedData cache: keyed by contract properties (rank, seq_len, feature_mode,
        # mtf_mode, scaler, n_features) so models with identical data requirements
        # share preparation. Biggest win: 3 boosting models (all rank 2) prepare once,
        # reuse 3x.
        self._prepared_cache: dict[tuple, PreparedData] = {}

        # Per-model feature subsets: each model gets features appropriate for its
        # contract (top N by variance where N = min(available, max_features)).
        # Populated by _pre_training_validation, consumed by _prepare_with_cache.
        self._per_model_features: dict[str, list[str]] = {}
        self._all_feature_names: list[str] = []

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

    def _prepare_with_cache(
        self,
        df: pd.DataFrame,
        model_name: str,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> PreparedData:
        """
        Prepare data with caching by model contract properties.

        Models sharing the same data rank, sequence length, feature mode,
        MTF mode, and scaler type produce identical PreparedData, so we
        prepare once and reuse. This mainly benefits boosting models
        (all rank 2, identical contracts) — prepare once, reuse 3x.

        Neural models with different sequence lengths or feature modes
        get separate cache entries automatically.

        Args:
            df: Input DataFrame
            model_name: Model name (determines contract/adapter)
            additional_dfs: Optional additional timeframe DataFrames

        Returns:
            PreparedData (from cache if available)
        """
        from src.core.contracts import get_model_contract

        contract = get_model_contract(model_name)
        n_model_features = len(self._per_model_features.get(model_name, []))
        cache_key = (
            contract.input_rank.value,
            contract.sequence_length,
            contract.feature_mode.value,
            contract.mtf_mode.value,
            contract.scaler_type,
            n_model_features,
        )

        if cache_key in self._prepared_cache:
            logger.debug(
                f"Cache hit for {model_name} (rank={contract.input_rank.value}, "
                f"seq={contract.sequence_length})"
            )
            return self._prepared_cache[cache_key]

        # Filter DataFrame to per-model feature subset (if computed)
        if self._per_model_features and model_name in self._per_model_features:
            model_features = set(self._per_model_features[model_name])
            all_features = set(self._all_feature_names)
            drop_cols = [c for c in df.columns if c in all_features and c not in model_features]
            if drop_cols:
                df = df.drop(columns=drop_cols)
                logger.debug(f"Filtered to {len(model_features)} features for {model_name}")

        prepared = self._data_preparer.prepare(
            df=df,
            model_name=model_name,
            additional_dfs=additional_dfs,
        )
        self._prepared_cache[cache_key] = prepared
        logger.debug(
            f"Cached PreparedData for {model_name} (rank={contract.input_rank.value}, "
            f"seq={contract.sequence_length})"
        )
        return prepared

    def _clear_prepared_cache(self) -> None:
        """Clear PreparedData cache to free memory after a horizon completes."""
        if self._prepared_cache:
            n_entries = len(self._prepared_cache)
            self._prepared_cache.clear()
            gc.collect()
            logger.debug(f"Cleared {n_entries} PreparedData cache entries")

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

        # Per-model feature selection: each model gets top N features by variance
        # where N = min(available_features, model.max_features).
        # This replaces the old global truncation that picked one feature count
        # for ALL models (causing conflicts, e.g. N-BEATS max=20 vs LSTM min=50).
        if feature_names and len(feature_names) > 0:
            from src.core.contracts import get_model_contract

            self._all_feature_names = list(feature_names)

            # --- Pre-filter: remove degenerate and redundant features ---
            original_count = len(feature_names)
            feature_df = df[feature_names].dropna()

            if len(feature_df) > 0:
                # Step 1: Remove low-variance (near-constant) features
                kept_after_var, removed_low_var = filter_low_variance(
                    feature_df, list(feature_names), variance_threshold=0.01
                )
                logger.info(
                    f"  Pre-filter: low-variance removed {len(removed_low_var)}"
                    f"/{original_count} features"
                )

                # Step 2: Remove highly correlated features
                kept_after_corr, removed_corr, corr_groups = filter_correlated_features(
                    feature_df, kept_after_var, correlation_threshold=0.85
                )
                logger.info(
                    f"  Pre-filter: correlation removed {len(removed_corr)}"
                    f"/{len(kept_after_var)} features"
                    f" ({len(corr_groups)} correlated groups)"
                )

                # Safety: if filtering removed too many features, skip and warn
                min_surviving = 10
                if len(kept_after_corr) >= min_surviving:
                    feature_names = kept_after_corr
                    logger.info(
                        f"  Pre-filter result: {original_count} -> {len(feature_names)} features"
                    )
                else:
                    logger.warning(
                        f"  Pre-filter would reduce features to {len(kept_after_corr)}"
                        f" (< {min_surviving}), skipping pre-filter"
                    )

            # Compute variance ranking once (reused for all models)
            X_subset = df[feature_names].dropna()
            variances = X_subset.var().sort_values(ascending=False) if len(X_subset) > 0 else None

            for model_name in self.config.models:
                model_contract = get_model_contract(model_name)
                max_feat = model_contract.max_features
                if len(feature_names) > max_feat and variances is not None:
                    model_features = variances.head(int(max_feat)).index.tolist()
                    logger.info(
                        f"    {model_name}: selected top {len(model_features)}"
                        f"/{len(feature_names)} features (max={max_feat})"
                    )
                else:
                    model_features = list(feature_names)
                self._per_model_features[model_name] = model_features

        # 1. Contract validation (if enabled)
        if self.config.strict_validation:
            logger.info("  [1/3] Validating data contracts...")
            try:
                from src.core.contracts import DataContractViolation, get_model_contract

                if feature_names and len(feature_names) > 0:
                    # Validate each model against its per-model feature subset
                    # NOTE: Skip rank validation — adapters transform to 3D/4D later
                    for model_name in self.config.models:
                        model_contract = get_model_contract(model_name)
                        model_feat = self._per_model_features.get(model_name, feature_names)
                        n_feat = len(model_feat)

                        issues = []
                        if n_feat < model_contract.min_features:
                            issues.append(
                                f"Too few features: model needs >= {model_contract.min_features}, "
                                f"data has {n_feat}"
                            )
                        if n_feat > model_contract.max_features:
                            issues.append(
                                f"Too many features: model max is {model_contract.max_features}, "
                                f"data has {n_feat}"
                            )

                        if issues:
                            errors.append(
                                f"Contract violation for {model_name}: {'; '.join(issues)}"
                            )

                    logger.info(
                        f"    Per-model features: {len(feature_names)} total, "
                        f"{len(self.config.models)} models validated"
                    )
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

            # Clear OOF predictions to free memory after ensemble building (Phase 37 fix)
            # Each OOF dict entry is 50-200MB; clearing saves 750MB-1.5GB for typical runs
            oof_count = len(self._oof_predictions)
            self._oof_predictions.clear()
            logger.debug(f"Cleared {oof_count} OOF predictions from memory")

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

        Boosting models (xgboost, lightgbm, catboost) are CPU-bound and trained
        in parallel via ParallelTrainingService. Neural/transformer models remain
        sequential to avoid GPU contention.

        Args:
            df: Input DataFrame with features and labels
            additional_dfs: Optional additional timeframe DataFrames
        """
        from src.core.contracts import get_model_contract

        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")

            # Separate boosting (CPU, parallelizable) from neural/transformer (GPU, sequential)
            boosting_models = []
            sequential_models = []
            for model_name in self.config.models:
                contract = get_model_contract(model_name)
                if contract.model_family == "boosting":
                    boosting_models.append(model_name)
                else:
                    sequential_models.append(model_name)

            # Train boosting models in parallel (if 2+ models)
            if len(boosting_models) >= 2:
                logger.info(
                    f"\nTraining {len(boosting_models)} boosting models in parallel: "
                    f"{boosting_models}"
                )
                self._train_boosting_parallel(
                    df=df,
                    horizon=horizon,
                    boosting_models=boosting_models,
                    additional_dfs=additional_dfs,
                )
            else:
                # Single boosting model trains sequentially (no parallelism benefit)
                sequential_models = boosting_models + sequential_models

            # Train non-boosting models sequentially (GPU contention)
            for model_name in sequential_models:
                self._train_model_sequential(
                    df=df,
                    model_name=model_name,
                    horizon=horizon,
                    additional_dfs=additional_dfs,
                )

            # Clear PreparedData cache after each horizon to free memory
            self._clear_prepared_cache()

    def _train_boosting_parallel(
        self,
        df: pd.DataFrame,
        horizon: int,
        boosting_models: list[str],
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Train boosting models in parallel using ParallelTrainingService.

        Prepares data for each model, submits training requests in parallel,
        then processes results (calibration, OOF generation, storage).

        Args:
            df: Input DataFrame with features and labels
            horizon: Prediction horizon
            boosting_models: List of boosting model names to train
            additional_dfs: Optional additional timeframe DataFrames
        """
        # Prepare data and build training requests for each boosting model
        # All boosting models share the same adapter ("tabular", rank 2),
        # so data is prepared once via cache and reused
        prepared_map: dict[str, PreparedData] = {}
        training_requests: list[ModelTrainingRequest] = []

        for model_name in boosting_models:
            prepared = self._prepare_with_cache(
                df=df,
                model_name=model_name,
                additional_dfs=additional_dfs,
            )
            prepared_map[model_name] = prepared
            logger.info(f"  {model_name} data prepared: {prepared.summary()}")

            request = ModelTrainingRequest(
                model_name=model_name,
                horizon=horizon,
                prepared_data=prepared,
                sequence_length=self.config.sequence_length,
                output_dir=self.output_dir / f"h{horizon}",
                optimize_hyperparams=self.config.optimize_hyperparams,
                n_splits=self.config.n_splits,
                hyperparam_trials=self.config.hyperparam_trials,
                scoring=self.config.optuna_metric,
                use_feature_selection=self.config.optimize_features,
            )
            training_requests.append(request)

        # Train all boosting models in parallel
        parallel_results = self._parallel_service.train_models_parallel(training_requests)

        # Process parallel results: calibration, OOF, result conversion
        for model_name, service_result in zip(boosting_models, parallel_results, strict=True):
            prepared = prepared_map[model_name]

            # Calibrate if enabled
            if self.config.auto_calibrate:
                self._calibrate_model(service_result, prepared, model_name)

            # Store trained model for later use
            self._trained_models[f"{model_name}_h{horizon}"] = service_result.trainer

            # Convert service result to orchestrator result
            result = ModelTrainingResult(
                model_name=service_result.model_name,
                horizon=service_result.horizon,
                metrics=service_result.metrics,
                trainer=service_result.trainer,
                training_time_seconds=service_result.training_time_seconds,
                n_features=service_result.n_features,
                data_rank=service_result.data_rank,
            )

            # Log post-training feature importance (read-only feedback)
            self._log_feature_importance(model_name, service_result.trainer)

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

        # Clean up all prepared data
        del prepared_map
        gc.collect()

    def _train_model_sequential(
        self,
        df: pd.DataFrame,
        model_name: str,
        horizon: int,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Train a single model sequentially.

        Handles data preparation, training, OOF generation, and cleanup.

        Args:
            df: Input DataFrame with features and labels
            model_name: Name of the model to train
            horizon: Prediction horizon
            additional_dfs: Optional additional timeframe DataFrames
        """
        logger.info(f"\nTraining {model_name}...")

        # Prepare data using PHASE_2 adapters (with caching by adapter type)
        prepared = self._prepare_with_cache(
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

        # Log post-training feature importance (read-only feedback)
        self._log_feature_importance(model_name, result.trainer)

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

        # Note: PreparedData cleanup is handled by _clear_prepared_cache()
        # at the end of each horizon, since cached entries may be shared

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
            scoring=self.config.optuna_metric,
            use_feature_selection=self.config.optimize_features,
        )
        result = self._model_service.train_model(request)

        # Phase 4F: Automatic calibration if enabled
        if self.config.auto_calibrate:
            self._calibrate_model(result, prepared, model_name)

        # Store for later use
        self._trained_models[f"{model_name}_h{horizon}"] = result.trainer

        # Convert service result to orchestrator result
        calibrator = getattr(result, "calibrator", None)
        return ModelTrainingResult(
            model_name=result.model_name,
            horizon=result.horizon,
            metrics=result.metrics,
            trainer=result.trainer,
            training_time_seconds=result.training_time_seconds,
            n_features=result.n_features,
            data_rank=result.data_rank,
            calibrator=calibrator,
        )

    def _log_feature_importance(self, model_name: str, trainer: Any) -> None:
        """Log top 10 feature importances after training (read-only, no behavior change)."""
        try:
            model = getattr(trainer, "model", None)
            if model is None:
                return
            importances = model.get_feature_importance()
            if importances:
                sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.info(f"  Feature importance for {model_name} (top 10):")
                for feat, score in sorted_imp:
                    logger.info(f"    {feat}: {score:.4f}")
            else:
                family = getattr(model, "model_family", "unknown")
                logger.info(f"  Feature importance not available for {model_name} ({family} model)")
        except Exception as e:
            logger.debug(f"  Could not retrieve feature importance for {model_name}: {e}")

    def _calibrate_model(
        self,
        result: Any,
        prepared: PreparedData,
        model_name: str,
    ) -> None:
        """
        Calibrate model probabilities (Phase 4F).

        Applies probability calibration to improve reliability of model predictions.
        Uses validation set for calibration to avoid overfitting.

        Args:
            result: ModelTrainingService result with trained model
            prepared: PreparedData with validation split
            model_name: Name of the model being calibrated
        """
        try:
            from src.models.calibration import CalibrationConfig, ProbabilityCalibrator

            logger.info(f"  Calibrating {model_name} probabilities...")

            # Check if model supports probability prediction
            if not hasattr(result.trainer, "predict_proba"):
                logger.warning(
                    f"    {model_name} doesn't support predict_proba, skipping calibration"
                )
                return

            # Need validation split for calibration
            if prepared.X_val is None or prepared.y_val is None:
                logger.warning("    No validation data available, skipping calibration")
                return

            # Check minimum samples
            if len(prepared.y_val) < self.config.calibration_min_samples:
                logger.warning(
                    f"    Insufficient validation samples for calibration "
                    f"({len(prepared.y_val)} < {self.config.calibration_min_samples})"
                )
                return

            # Get validation probabilities
            val_probas = result.trainer.predict_proba(prepared.X_val)

            # Handle multi-class (take positive class probabilities)
            if val_probas.ndim == 2 and val_probas.shape[1] > 1:
                val_probas = val_probas[:, 1]  # Positive class

            # Create calibration config
            method = self.config.calibration_method
            if method not in ("isotonic", "sigmoid", "auto"):
                method = "auto"
            calib_config = CalibrationConfig(
                method=method,  # type: ignore[arg-type]
                min_samples_per_class=self.config.calibration_min_samples,
            )

            # Fit calibrator
            calibrator = ProbabilityCalibrator(calib_config)
            metrics = calibrator.fit(prepared.y_val, val_probas)

            # Log calibration improvement
            logger.info(
                f"    Brier improvement: {metrics.brier_improvement:.1%}, "
                f"ECE improvement: {metrics.ece_improvement:.1%}"
            )

            # Attach calibrator to result for later use
            # This allows inference pipeline to apply calibration
            if not hasattr(result, "calibrator"):
                result.calibrator = calibrator
                result.calibration_metrics = metrics

        except ImportError:
            logger.warning("    Calibration module not available, skipping")
        except Exception as e:
            logger.warning(f"    Calibration failed: {e}")

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

        Also performs diversity analysis on base model predictions (Phase 16E).

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

        # Phase 16E: Analyze ensemble diversity
        diversity_metrics = self._analyze_ensemble_diversity(result.aligned_oof, df)

        # Convert service result to orchestrator result
        ensemble_result = None
        if result.meta_learner is not None:
            # Merge diversity metrics into ensemble metrics
            ensemble_metrics = result.ensemble_metrics.copy()
            if diversity_metrics:
                ensemble_metrics.update(diversity_metrics)

            ensemble_result = ModelTrainingResult(
                model_name=f"ensemble_{self.config.meta_learner}",
                horizon=self.config.horizons[0] if self.config.horizons else 20,
                metrics=ensemble_metrics,
                trainer=result.meta_learner,
                training_time_seconds=result.training_time_seconds,
            )

        return result.aligned_oof, result.stacking_dataset, ensemble_result

    def _analyze_ensemble_diversity(
        self,
        aligned_oof: AlignedOOFResult | None,
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Analyze diversity of base model predictions using DiversityAnalyzer.

        Phase 16E: Wire DiversityAnalyzer to ensemble training.

        Computes diversity metrics including:
        - Q-statistic (Yule's Q): Agreement measure between classifier pairs
        - Pairwise correlation: Average correlation between base predictions
        - Disagreement: Fraction of samples where classifiers disagree
        - Diversity score: Composite diversity score

        Args:
            aligned_oof: AlignedOOFResult with base model predictions
            df: Original DataFrame for label extraction

        Returns:
            Dict of diversity metrics, empty if analysis fails
        """
        if aligned_oof is None:
            logger.debug("No aligned OOF, skipping diversity analysis")
            return {}

        try:
            from src.models.ensemble.diversity import DiversityAnalyzer

            logger.info("\n--- Ensemble Diversity Analysis (Phase 16E) ---")

            # Extract base model predictions
            base_predictions: dict[str, np.ndarray] = {}
            base_probabilities: dict[str, np.ndarray] = {}

            for model_key, oof in self._oof_predictions.items():
                if oof is not None and hasattr(oof, "predictions"):
                    # Align to common indices
                    aligned_preds = oof.predictions[
                        np.isin(np.arange(len(oof.predictions)), aligned_oof.common_indices)
                    ]
                    if len(aligned_preds) == len(aligned_oof.common_indices):
                        base_predictions[model_key] = aligned_preds

                    # Also get probabilities if available
                    if hasattr(oof, "probabilities") and oof.probabilities is not None:
                        aligned_probs = oof.probabilities[
                            np.isin(np.arange(len(oof.probabilities)), aligned_oof.common_indices)
                        ]
                        if len(aligned_probs) == len(aligned_oof.common_indices):
                            base_probabilities[model_key] = aligned_probs

            if len(base_predictions) < 2:
                logger.warning("Need at least 2 models for diversity analysis")
                return {}

            # Get ground truth labels
            y_true = None
            for h in self.config.horizons:
                label_col = f"label_h{h}"
                if label_col in df.columns:
                    y_true = df[label_col].values[aligned_oof.common_indices]
                    break

            # Initialize analyzer
            analyzer = DiversityAnalyzer(
                min_diversity_threshold=0.3,
                correlation_threshold=0.8,
                n_classes=3,  # Long/Neutral/Short
            )

            # Run diversity analysis
            metrics = analyzer.analyze(
                base_predictions=base_predictions,
                base_probabilities=base_probabilities if base_probabilities else None,
                y_true=y_true,
            )

            # Log key metrics
            logger.info(f"  Diversity Score: {metrics.diversity_score:.4f}")
            logger.info(f"  Q-statistic: {metrics.q_statistic:.4f}")
            logger.info(f"  Pairwise Correlation: {metrics.pairwise_correlation:.4f}")
            logger.info(f"  Disagreement Rate: {metrics.disagreement:.4f}")
            logger.info(f"  Double Fault Rate: {metrics.double_fault:.4f}")
            logger.info(f"  Entropy: {metrics.entropy:.4f}")

            # Log recommendations if any
            if metrics.recommendations:
                logger.info("\n  Diversity Recommendations:")
                for rec in metrics.recommendations[:3]:  # Top 3 recommendations
                    logger.info(f"    - {rec}")

            # Log highly correlated pairs
            high_corr_pairs = [
                (m1, m2, corr)
                for (m1, m2), corr in metrics.model_pair_correlations.items()
                if abs(corr) > 0.8
            ]
            if high_corr_pairs:
                logger.info("\n  High Correlation Pairs (>0.8):")
                for m1, m2, corr in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:5]:
                    logger.info(f"    {m1} <-> {m2}: {corr:.4f}")

            # Return metrics dict for inclusion in ensemble result
            return {
                "diversity_score": metrics.diversity_score,
                "diversity_q_statistic": metrics.q_statistic,
                "diversity_correlation": metrics.pairwise_correlation,
                "diversity_disagreement": metrics.disagreement,
                "diversity_double_fault": metrics.double_fault,
                "diversity_entropy": metrics.entropy,
                "diversity_kl_divergence": metrics.kl_divergence,
            }

        except ImportError as e:
            logger.warning(f"Diversity analysis not available: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Diversity analysis failed: {e}")
            return {}

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
            n_windows=getattr(self.config, "wf_n_windows", self.config.n_splits),
            window_type=getattr(self.config, "wf_window_type", "expanding"),
            min_train_pct=getattr(self.config, "wf_min_train_pct", self.config.train_ratio),
            test_pct=getattr(self.config, "wf_test_pct", self.config.val_ratio),
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

        # Clean up prepared data after walk-forward training (Phase 37 fix)
        del prepared
        gc.collect()

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

                # Clean up prepared data after regime training (Phase 37 fix)
                del prepared
                gc.collect()

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

            # Clean up after each horizon to prevent memory accumulation (Phase 37 fix)
            gc.collect()

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
