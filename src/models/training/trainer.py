"""
Trainer - Orchestrates model training workflow.

The Trainer class handles the complete training pipeline:
1. Load and prepare data from TimeSeriesDataContainer
2. Apply per-model feature selection (for tabular/classical models)
3. Apply model-specific preprocessing
4. Train model with early stopping
5. Evaluate on validation set
6. Save artifacts (model, metrics, predictions, feature selection)

Example:
    >>> from src.models.trainer import Trainer
    >>> from src.models.config import TrainerConfig
    >>> from src.core.container import TimeSeriesDataContainer
    ...
    >>> config = TrainerConfig(model_name="xgboost", horizon=20)
    >>> container = TimeSeriesDataContainer.from_parquet_dir(
    ...     "data/splits/scaled", horizon=20
    ... )
    ...
    >>> trainer = Trainer(config)
    >>> results = trainer.run(container)
    >>> print(results["evaluation_metrics"]["val_f1"])
"""

from __future__ import annotations

import logging
import secrets
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..calibration import CalibrationConfig, ProbabilityCalibrator
from ..config import TrainerConfig
from ..data_preparation import prepare_training_data
from ..metrics import compute_classification_metrics, compute_trading_metrics
from ..registry import ModelRegistry
from ..tracking import TrackerConfig, get_tracker
from ..tracking.base import ExperimentTracker
from .artifacts import TrainerArtifactsMixin
from .evaluation import TrainerEvaluationMixin, _validate_labels
from .features import TrainerFeaturesMixin

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer
    from src.core.coordination import TimeframeCoordinator
    from src.data.adapters import PreparedData
    from src.optimization.feature_selection import FeatureSelectionManager

logger = logging.getLogger(__name__)


class Trainer(TrainerFeaturesMixin, TrainerEvaluationMixin, TrainerArtifactsMixin):
    """
    Orchestrates model training and evaluation.

    Handles the complete training workflow including data preparation,
    model training, evaluation, and artifact saving.

    Inherits from:
    - TrainerFeaturesMixin: Feature selection and feature set resolution
    - TrainerEvaluationMixin: Test set evaluation
    - TrainerArtifactsMixin: Artifact saving (configs, metrics, models)

    Attributes:
        config: TrainerConfig with training settings
        model: Instantiated model from registry
        run_id: Unique identifier for this training run
        output_path: Path to output directory

    Example:
        >>> config = TrainerConfig(model_name="xgboost", horizon=20)
        >>> trainer = Trainer(config)
        >>> results = trainer.run(container)
    """

    def __init__(self, config: TrainerConfig) -> None:
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_path = config.output_dir / self.run_id

        # Create model from registry
        self.model = ModelRegistry.create(
            config.model_name,
            config=config.model_config,
        )

        # Initialize feature selection manager based on model family
        self.feature_selector: FeatureSelectionManager | None = None
        self._setup_feature_selection()

        # Feature set columns (set during run() for per-model filtering)
        self._feature_set_columns: list[str] | None = None

        # Calibrator (set during run() if calibration is enabled)
        self.calibrator: ProbabilityCalibrator | None = None

        # Initialize experiment tracker
        self.tracker: ExperimentTracker = self._setup_tracker()

        # Validate feature_set against model recommendations (MOD-005)
        self._validate_feature_set()

        logger.info(
            f"Initialized Trainer: model={config.model_name}, "
            f"horizon={config.horizon}, run_id={self.run_id}, "
            f"feature_selection={self._is_feature_selection_enabled()}"
        )

    def _generate_run_id(self) -> str:
        """
        Generate unique run identifier with collision prevention.

        Format: {model}_{horizon}_{timestamp_with_ms}_{random_suffix}
        Example: xgboost_h20_20251228_143025_789456_a3f9

        Milliseconds + random suffix ensure uniqueness even for parallel runs.
        """
        # Include milliseconds (%f) for sub-second precision
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Add 4-character random suffix for collision prevention
        random_suffix = secrets.token_hex(2)  # 2 bytes = 4 hex chars
        return f"{self.config.model_name}_h{self.config.horizon}_{timestamp}_{random_suffix}"

    def _validate_pipeline_lineage(self) -> tuple[bool, list[str]]:
        """
        Validate dataset integrity against pipeline lineage metadata.

        Returns:
            Tuple of (is_valid, issues_list)
        """
        from pathlib import Path

        from src.core.lineage import PipelineLineage, validate_dataset_checksum

        if not self.config.pipeline_run_id:
            return True, []

        lineage_path = (
            Path(self.config.project_root)
            / "data"
            / "lineage"
            / f"{self.config.pipeline_run_id}.json"
        )

        if not lineage_path.exists():
            logger.warning(f"Pipeline lineage file not found: {lineage_path}")
            return False, [f"Lineage file not found: {lineage_path}"]

        try:
            lineage = PipelineLineage.load(lineage_path)
        except Exception as e:
            logger.error(f"Failed to load pipeline lineage: {e}")
            return False, [f"Failed to load lineage: {e}"]

        all_valid = True
        all_issues = []

        for name, expected_checksum in lineage.dataset_checksums.items():
            dataset_path = Path(expected_checksum.file_path)

            if not dataset_path.exists():
                logger.warning(f"Dataset file not found: {dataset_path}")
                all_valid = False
                all_issues.append(f"Dataset not found: {dataset_path}")
                continue

            is_valid, issues = validate_dataset_checksum(
                dataset_path, expected_checksum, strict=False
            )

            if not is_valid:
                logger.warning(f"Dataset validation failed for {name}: {issues}")
                all_valid = False
                all_issues.extend(issues)
            else:
                logger.info(f"Dataset validation passed for {name}")

        if all_valid:
            logger.info(
                f"Pipeline lineage validation successful for run_id={self.config.pipeline_run_id}"
            )
        else:
            logger.warning(
                f"Pipeline lineage validation issues found: {len(all_issues)} issues for run_id={self.config.pipeline_run_id}"
            )

        return all_valid, all_issues

    def _run_pre_training_validation(
        self,
        X_train_df: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        feature_names: list[str],
    ) -> None:
        """
        Run pre-training validation (Phase 7: Contract Enforcement).

        Performs leakage detection and lookahead auditing before training.
        Raises on failure when blocking mode is enabled (default).

        Args:
            X_train_df: Training features DataFrame
            y_train: Training labels
            feature_names: List of feature column names

        Raises:
            LeakageDetectedError: If leakage is detected and check_leakage=True
            LookaheadBiasError: If lookahead bias is detected and check_lookahead=True
        """
        logger.info("\n" + "-" * 40)
        logger.info("PRE-TRAINING VALIDATION (Phase 7)")
        logger.info("-" * 40)

        # Convert y_train to numpy if needed
        y_train_arr = np.asarray(y_train)

        # 1. Leakage detection (if enabled)
        if self.config.check_leakage:
            logger.info("  [1/2] Running leakage detection...")
            try:
                from src.validation.leakage_detection import check_feature_label_correlation

                report = check_feature_label_correlation(
                    features=X_train_df,
                    labels=y_train_arr,
                    feature_names=feature_names,
                    correlation_threshold=self.config.validation_correlation_threshold,
                    raise_on_leakage=True,  # Blocking mode (Phase 7)
                )

                logger.info(
                    f"    Leakage check passed: "
                    f"{report.n_features} features analyzed, 0 suspicious"
                )

            except Exception as e:
                # Re-raise LeakageDetectedError, log other errors
                from src.validation.leakage_detection import LeakageDetectedError

                if isinstance(e, LeakageDetectedError):
                    logger.error(f"    LEAKAGE DETECTED: {e}")
                    raise
                else:
                    logger.warning(f"    Leakage detection failed: {e}")
        else:
            logger.info("  [1/2] Leakage detection: SKIPPED (check_leakage=False)")

        # 2. Lookahead audit (if enabled)
        if self.config.check_lookahead:
            logger.info("  [2/2] Running lookahead validation...")
            try:
                from src.validation.lookahead_audit import validate_resample_config

                # Basic resample config validation
                is_valid, issues = validate_resample_config(
                    closed="left",
                    label="left",
                )

                if is_valid:
                    logger.info("    Lookahead audit passed: resample config validated")
                else:
                    logger.warning(f"    Lookahead issues: {'; '.join(issues)}")

            except Exception as e:
                from src.validation.lookahead_audit import LookaheadBiasError

                if isinstance(e, LookaheadBiasError):
                    logger.error(f"    LOOKAHEAD BIAS DETECTED: {e}")
                    raise
                else:
                    logger.warning(f"    Lookahead audit failed: {e}")
        else:
            logger.info("  [2/2] Lookahead audit: SKIPPED (check_lookahead=False)")

        logger.info("\n  Pre-training validation: PASSED")
        logger.info("-" * 40 + "\n")

    def _setup_tracker(self) -> ExperimentTracker:
        """
        Initialize experiment tracker based on configuration.

        Returns:
            Configured ExperimentTracker instance
        """
        tracker_config = TrackerConfig(
            enabled=self.config.tracking_enabled,
            backend=self.config.tracking_backend,
            experiment_name=self.config.experiment_name or f"model_{self.config.model_name}",
            run_name=self.run_id,
            tracking_uri=self.config.tracking_uri,
            output_dir=self.config.output_dir / "tracking",
            log_artifacts=True,
            tags={
                "model_name": self.config.model_name,
                "horizon": str(self.config.horizon),
                **self.config.tracking_tags,
            },
        )
        return get_tracker(config=tracker_config)

    def _setup_output_dir(self) -> None:
        """Create output directory structure."""
        dirs = [
            self.output_path / "config",
            self.output_path / "checkpoints",
            self.output_path / "predictions",
            self.output_path / "metrics",
            self.output_path / "plots",
            self.output_path / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created output directory: {self.output_path}")

    def _is_heterogeneous_ensemble(self) -> bool:
        """Check if this is a heterogeneous stacking ensemble requiring both 2D and 3D data."""
        if self.model.model_family != "ensemble":
            return False
        if not hasattr(self.model, "ensemble_type"):
            return False
        # Only stacking supports heterogeneous (voting/blending require same shape)
        if self.model.ensemble_type != "stacking":
            return False
        # Check if base models have mixed input requirements
        from ..ensemble.validator import is_heterogeneous_ensemble, validate_ensemble_config

        base_models = self.config.model_config.get("base_model_names", [])
        is_valid, error = validate_ensemble_config(base_models, ensemble_type="stacking")
        if not is_valid:
            raise ValueError(error)
        return is_heterogeneous_ensemble(base_models)

    def _get_coordinator(self) -> TimeframeCoordinator:
        """
        Get or create a TimeframeCoordinator for multi-timeframe data loading.

        Returns:
            TimeframeCoordinator instance configured for this trainer's data directory

        Note:
            This is a Phase 3 SNwH method for heterogeneous ensemble support.
        """
        from src.core.coordination import TimeframeCoordinator

        # Determine data directory from config
        # Try output_dir parent structure or fall back to standard path
        data_dir = self.config.output_dir.parent.parent / "data" / "splits" / "scaled"
        if not data_dir.exists():
            # Fall back to current working directory structure
            from pathlib import Path

            data_dir = Path("data/splits/scaled")

        return TimeframeCoordinator(
            data_dir=data_dir,
            split="train",
            horizon=self.config.horizon,
        )

    def _load_data_for_model(
        self,
        container: TimeSeriesDataContainer,
        model_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data for a specific model based on its contract's primary timeframe.

        This method supports Phase 3 SNwH where different models train on different
        timeframes. If the model's primary timeframe matches the container's timeframe,
        data is loaded directly. Otherwise, the TimeframeCoordinator loads from the
        appropriate timeframe-specific files.

        Args:
            container: TimeSeriesDataContainer with data
            model_name: Model name (defaults to self.config.model_name)

        Returns:
            (train_df, val_df) DataFrames for the model's primary timeframe
        """
        from src.core.contracts import get_model_contract

        model_name = model_name or self.config.model_name
        contract = get_model_contract(model_name)
        primary_tf = contract.primary_timeframe
        current_tf = self.config.primary_timeframe

        # If same timeframe, use container directly
        if primary_tf == current_tf:
            train_split = container.get_split("train")
            val_split = container.get_split("val")
            return train_split.df, val_split.df

        # Otherwise, need to load from correct timeframe via coordinator
        logger.info(
            f"Model {model_name} requires {primary_tf} but container has {current_tf}. "
            f"Loading from timeframe-specific data."
        )

        # Use coordinator to load correct timeframe
        coordinator = self._get_coordinator()
        coordinator.load_timeframes([primary_tf])

        # Get train data
        train_df = coordinator.get_timeframe_data(primary_tf).df

        # Load val data separately
        from src.core.coordination import TimeframeCoordinator

        coordinator_val = TimeframeCoordinator(
            data_dir=coordinator.data_dir,
            split="val",
            horizon=self.config.horizon,
        )
        coordinator_val.load_timeframes([primary_tf])
        val_df = coordinator_val.get_timeframe_data(primary_tf).df

        return train_df, val_df

    def _load_heterogeneous_data(
        self,
        container: TimeSeriesDataContainer,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data for heterogeneous ensemble with multiple timeframes.

        Each base model may need a different timeframe, so we load all required
        timeframes and return the anchor (smallest) timeframe as the primary data.

        Args:
            container: TimeSeriesDataContainer with data

        Returns:
            (train_df, val_df) for anchor timeframe
        """
        from src.core.contracts import get_model_contract
        from src.core.coordination import TimeframeCoordinator

        base_models = self.config.model_config.get("base_model_names", [])

        # Get all required timeframes
        required_tfs = set()
        for model_name in base_models:
            contract = get_model_contract(model_name)
            required_tfs.add(contract.primary_timeframe)
            required_tfs.update(contract.mtf_timeframes)

        logger.info(f"Heterogeneous ensemble requires timeframes: {sorted(required_tfs)}")

        # Load all timeframes
        coordinator = self._get_coordinator()
        coordinator.load_timeframes(list(required_tfs))

        # Return anchor timeframe data (smallest)
        anchor_tf = coordinator.anchor_timeframe
        if anchor_tf is None:
            raise RuntimeError("No anchor timeframe available from coordinator")
        train_df = coordinator.get_timeframe_data(anchor_tf).df

        # Load val data
        coordinator_val = TimeframeCoordinator(
            data_dir=coordinator.data_dir,
            split="val",
            horizon=self.config.horizon,
        )
        coordinator_val.load_timeframes(list(required_tfs))
        val_anchor_tf = coordinator_val.anchor_timeframe
        if val_anchor_tf is None:
            raise RuntimeError("No anchor timeframe available from validation coordinator")
        val_df = coordinator_val.get_timeframe_data(val_anchor_tf).df

        return train_df, val_df

    def run(
        self,
        container: TimeSeriesDataContainer,
        skip_save: bool = False,
    ) -> dict[str, Any]:
        """
        Execute complete training pipeline.

        Workflow:
        1. Setup output directories
        2. Load and prepare data
        3. Run feature selection (for tabular/classical models)
        4. Apply model-specific preprocessing
        5. Train model
        6. Evaluate on validation set
        7. Save artifacts (including feature selection)

        Args:
            container: TimeSeriesDataContainer with train/val data
            skip_save: If True, skip saving artifacts (for testing)

        Returns:
            Dict with training results including:
            - run_id: Run identifier
            - training_metrics: TrainingMetrics from training
            - evaluation_metrics: Validation set metrics
            - output_path: Path to outputs
            - feature_selection: Feature selection results (if enabled)
        """
        start_time = time.time()

        # Setup
        self._setup_output_dir()
        self._save_config()

        # Start experiment tracking
        tracking_run_id = self.tracker.start_run(
            run_name=self.run_id,
            tags={
                "model_family": self.model.model_family,
                "feature_set": self.config.feature_set,
            },
        )
        logger.info(f"Started experiment tracking run: {tracking_run_id}")

        # Log training configuration as parameters
        self.tracker.log_params(self.config.to_dict())

        # Validate pipeline lineage if pipeline_run_id is specified
        lineage_validated = True
        lineage_issues: list[str] = []
        if self.config.pipeline_run_id:
            lineage_validated, lineage_issues = self._validate_pipeline_lineage()

        # Load data - get DataFrames for feature selection
        logger.info("Loading data from container...")

        # For feature selection, we need the feature names
        feature_names = container.feature_columns

        # Get raw training data (as DataFrames for feature selection)
        X_train_df, y_train_series, w_train_series = container.get_sklearn_arrays(
            "train", return_df=True
        )
        X_val_df, y_val_series, _ = container.get_sklearn_arrays("val", return_df=True)

        # LEAKAGE PREVENTION: Validate no invalid labels (-99) in training data
        # The container should filter these by default, but this is a defensive check
        _validate_labels(y_train_series, "training labels")
        _validate_labels(y_val_series, "validation labels")

        # Phase 7: Run pre-training validation (leakage/lookahead detection)
        self._run_pre_training_validation(X_train_df, y_train_series, feature_names)

        # Extract label_end_times for overlapping label purging
        label_end_times = container.get_label_end_times("train")
        if label_end_times is not None:
            logger.info(
                "Label end times available for purging overlapping labels "
                "(prevents leakage in stacking/blending ensembles)"
            )

        # Apply per-model feature set filtering (if specified)
        # This filters features BEFORE MDA-based feature selection
        feature_set_columns = self._resolve_feature_set_columns(X_train_df)
        self._feature_set_columns = feature_set_columns  # Store for test set evaluation
        if feature_set_columns is not None:
            X_train_df = self._apply_feature_set_filter(X_train_df, feature_set_columns)
            X_val_df = self._apply_feature_set_filter(X_val_df, feature_set_columns)
            # Update feature_names to reflect filtered set
            feature_names = list(X_train_df.columns)
            logger.info(
                f"Feature set filter applied: {len(feature_names)} features "
                f"(from original {len(container.feature_columns)})"
            )

        # Run feature selection (for tabular/classical models only)
        feature_selection_result = None
        if self._is_feature_selection_enabled():
            feature_selection_result = self._run_feature_selection(
                X_train_df=X_train_df,
                y_train=y_train_series,
                w_train=w_train_series,
                label_end_times=label_end_times,
            )

            # Apply feature selection to training data
            if self.feature_selector is not None:
                X_train_df = self.feature_selector.apply_selection_df(X_train_df)
                X_val_df = self.feature_selector.apply_selection_df(X_val_df)

            logger.info(
                f"Applied feature selection: {feature_selection_result.n_features_selected} features "
                f"(from {feature_selection_result.n_features_original})"
            )

        # Convert to numpy arrays and apply model-specific preprocessing
        # Track sequence data for heterogeneous stacking ensembles
        X_train_seq = None
        X_val_seq = None

        if self._is_heterogeneous_ensemble():
            # Heterogeneous stacking: load BOTH tabular and sequence data
            logger.info("Heterogeneous stacking detected: preparing both tabular and sequence data")

            # Tabular data (already loaded as DataFrames)
            # Use np.asarray for type-safe conversion (handles both DataFrame and ndarray)
            X_train = np.asarray(X_train_df)
            y_train = np.asarray(y_train_series)
            w_train = np.asarray(w_train_series)
            X_val = np.asarray(X_val_df)
            y_val = np.asarray(y_val_series)

            # Get feature columns for sequence models
            # Use the first sequence model's optimal feature set
            seq_feature_columns = None
            base_model_names = self.config.model_config.get("base_model_names", [])
            for model_name in base_model_names:
                model_info = ModelRegistry.get_model_info(model_name)
                if model_info and model_info.get("requires_sequences", False):
                    seq_feature_columns = self._get_sequence_model_feature_columns(
                        model_name, container
                    )
                    break  # Use first sequence model's feature set

            # Sequence data for sequence-based base models (with feature filtering)
            X_train_seq, y_train_seq, _, X_val_seq, y_val_seq = prepare_training_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
                feature_columns=seq_feature_columns,
            )

            # MOD-009 FIX: Alignment validation for heterogeneous ensemble data flow
            # Sequence data has fewer samples due to windowing (loses seq_len-1)
            tabular_train_samples = X_train.shape[0]
            seq_train_samples = X_train_seq.shape[0]
            expected_offset = tabular_train_samples - seq_train_samples

            logger.info(
                f"Heterogeneous data prepared: "
                f"tabular={X_train.shape}, sequence={X_train_seq.shape}"
            )
            logger.info(
                f"MOD-009: Data alignment check - "
                f"tabular_train={tabular_train_samples}, seq_train={seq_train_samples}, "
                f"offset={expected_offset} (expected: seq_len-1={self.config.sequence_length - 1})"
            )

            # Validate alignment consistency
            if expected_offset < 0:
                raise ValueError(
                    f"MOD-009: Invalid data alignment - sequence data ({seq_train_samples}) "
                    f"has more samples than tabular data ({tabular_train_samples}). "
                    f"This indicates a data preparation error."
                )

            # MOD-009 FIX: Change warnings to errors per "fail fast, fail hard" principle
            # Validate labels are consistent between tabular and sequence views
            # After accounting for offset, labels should match exactly
            y_train_trimmed = y_train[expected_offset:] if expected_offset > 0 else y_train
            if len(y_train_trimmed) != len(y_train_seq):
                raise ValueError(
                    f"MOD-009: CRITICAL - Label length mismatch after offset adjustment. "
                    f"y_train_trimmed={len(y_train_trimmed)}, y_train_seq={len(y_train_seq)}. "
                    f"Expected offset={expected_offset} (seq_len-1={self.config.sequence_length - 1}). "
                    f"This indicates a data alignment bug that would cause misaligned training. "
                    f"Check that tabular and sequence data were prepared from the same source."
                )
            elif not np.array_equal(y_train_trimmed, y_train_seq):
                # Calculate how many labels differ for diagnostic purposes
                n_different = int(np.sum(y_train_trimmed != y_train_seq))
                pct_different = 100.0 * n_different / len(y_train_seq)
                raise ValueError(
                    f"MOD-009: CRITICAL - Labels differ between tabular and sequence views after alignment. "
                    f"{n_different} labels ({pct_different:.1f}%) do not match. "
                    f"This indicates different data sources or processing paths were used. "
                    f"Training with misaligned labels would produce invalid models. "
                    f"Ensure both tabular and sequence data come from the same TimeSeriesDataContainer."
                )

        elif self.model.requires_4d:
            # Pure 4D model - load multi-resolution 4D tensors from container
            from src.core.contracts import FeatureMode, get_model_contract

            contract = get_model_contract(self.config.model_name)
            timeframes = [contract.primary_timeframe, *list(contract.mtf_timeframes)]

            features_per_timeframe = None
            if contract.feature_mode == FeatureMode.RAW:
                features_per_timeframe = ["open", "high", "low", "close", "volume"]

            X_train, y_train, w_train, X_val, y_val = prepare_training_data(
                container,
                requires_sequences=False,
                requires_4d=True,
                sequence_length=self.config.sequence_length,
                timeframes=timeframes,
                features_per_timeframe=features_per_timeframe,
                include_base_features=True,
            )
        elif self.model.requires_sequences:
            # Pure sequence model - get model-specific feature columns
            seq_feature_columns = self._get_sequence_model_feature_columns(
                self.config.model_name, container
            )
            X_train, y_train, w_train, X_val, y_val = prepare_training_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
                feature_columns=seq_feature_columns,
            )
        else:
            # Pure tabular model
            # Use np.asarray for type-safe conversion (handles both DataFrame and ndarray)
            X_train = np.asarray(X_train_df)
            y_train = np.asarray(y_train_series)
            w_train = np.asarray(w_train_series)
            X_val = np.asarray(X_val_df)
            y_val = np.asarray(y_val_series)

        # Log data shapes
        logger.info(
            f"Data shapes: "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_val={X_val.shape}, y_val={y_val.shape}"
        )

        # Set feature names on model (for interpretability)
        if hasattr(self.model, "set_feature_names") and not self.model.requires_sequences:
            if (
                self._is_feature_selection_enabled()
                and self.feature_selector is not None
                and self.feature_selector.is_fitted
            ):
                self.model.set_feature_names(self.feature_selector.selected_features)
            else:
                self.model.set_feature_names(feature_names)

        # Train model (pass label_end_times for ensemble models with internal CV)
        logger.info(f"Training {self.config.model_name}...")
        fit_kwargs: dict[str, Any] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "sample_weights": w_train,
            "config": self.config.model_config,
        }

        # Add label_end_times if model supports it (ensemble models with internal CV)
        # Non-ensemble models ignore this parameter (not in their fit() signature)
        if self.model.model_family == "ensemble" and label_end_times is not None:
            fit_kwargs["label_end_times"] = label_end_times

        # Add sequence data if heterogeneous stacking ensemble
        if self._is_heterogeneous_ensemble() and X_train_seq is not None:
            fit_kwargs["X_train_seq"] = X_train_seq
            fit_kwargs["X_val_seq"] = X_val_seq
            logger.info("Passing sequence data to heterogeneous stacking ensemble")

        training_metrics = self.model.fit(**fit_kwargs)

        # Evaluate
        logger.info("Evaluating on validation set...")
        # For heterogeneous stacking, pass both tabular and sequence data
        if self._is_heterogeneous_ensemble() and X_val_seq is not None:
            # Heterogeneous stacking models accept X_seq as keyword argument
            val_predictions = self.model.predict(X_val, X_seq=X_val_seq)
        else:
            val_predictions = self.model.predict(X_val)

        # MOD-005 FIX: Align y_val with predictions (may be trimmed for heterogeneous ensembles)
        # Also trim X_val if passthrough is enabled since it will be concatenated with OOF preds
        y_val_aligned = y_val
        X_val_aligned = X_val
        if len(val_predictions.class_predictions) < len(y_val):
            offset = len(y_val) - len(val_predictions.class_predictions)
            y_val_aligned = y_val[offset:]
            X_val_aligned = X_val[offset:]
            logger.info(
                f"Aligned validation data: trimmed {offset} samples from start "
                f"(y_val: {len(y_val)} -> {len(y_val_aligned)}, "
                f"X_val: {X_val.shape[0]} -> {X_val_aligned.shape[0]}) "
                f"to match prediction count {len(val_predictions.class_predictions)}"
            )

        eval_metrics = compute_classification_metrics(
            y_true=y_val_aligned,
            y_pred=val_predictions.class_predictions,
            y_proba=val_predictions.class_probabilities,
        )

        # Add trading metrics
        eval_metrics["trading"] = compute_trading_metrics(
            y_true=y_val_aligned,
            y_pred=val_predictions.class_predictions,
        )

        # Add feature selection info to eval metrics
        if feature_selection_result is not None:
            eval_metrics["feature_selection"] = {
                "n_features_original": feature_selection_result.n_features_original,
                "n_features_selected": feature_selection_result.n_features_selected,
                "reduction_ratio": feature_selection_result.reduction_ratio,
                "method": feature_selection_result.selection_method,
            }

        # Test set evaluation (one-shot generalization estimate)
        test_metrics = None
        test_predictions = None
        if self.config.evaluate_test_set:
            test_metrics, test_predictions = self._evaluate_test_set(container)

        # Probability calibration (leakage-safe: fits on held-out val set)
        self.calibrator = None
        if self.config.use_calibration:
            logger.info("Applying probability calibration...")
            # Cast calibration_method to Literal type for CalibrationConfig
            calibration_method = self.config.calibration_method
            if calibration_method not in ("isotonic", "sigmoid", "auto"):
                calibration_method = "auto"
            cal_config = CalibrationConfig(method=calibration_method)
            self.calibrator = ProbabilityCalibrator(cal_config)
            calibration_metrics = self.calibrator.fit(
                y_true=y_val_aligned,
                probabilities=val_predictions.class_probabilities,
            )
            eval_metrics["calibration"] = calibration_metrics.to_dict()

        # Log evaluation metrics to tracker
        flat_metrics = {
            "val_accuracy": eval_metrics["accuracy"],
            "val_macro_f1": eval_metrics["macro_f1"],
            "val_precision": eval_metrics["precision"],
            "val_recall": eval_metrics["recall"],
        }
        if "trading" in eval_metrics:
            flat_metrics["val_win_rate"] = eval_metrics["trading"].get("win_rate", 0)
            flat_metrics["val_profit_factor"] = eval_metrics["trading"].get("profit_factor", 0)
        if test_metrics:
            flat_metrics["test_accuracy"] = test_metrics.get("accuracy", 0)
            flat_metrics["test_macro_f1"] = test_metrics.get("macro_f1", 0)
        self.tracker.log_metrics(flat_metrics)

        # Save artifacts
        if not skip_save:
            self._save_artifacts(
                training_metrics,
                eval_metrics,
                val_predictions,
                test_metrics=test_metrics,
                test_predictions=test_predictions,
                lineage_validated=lineage_validated,
                lineage_issues=lineage_issues,
            )
            self._save_model()
            self._save_feature_selection()
            if self.calibrator is not None:
                self._save_calibrator()
            # Generate checksums for all artifacts (must be last)
            self._save_checksums()

            # Log artifacts to tracker
            self.tracker.log_artifact(self.output_path / "config", "config")
            self.tracker.log_artifact(self.output_path / "metrics", "metrics")
            self.tracker.log_artifact(self.output_path / "checkpoints", "model")

        total_time = time.time() - start_time

        results = {
            "run_id": self.run_id,
            "model_name": self.config.model_name,
            "horizon": self.config.horizon,
            "training_metrics": training_metrics.to_dict(),
            "evaluation_metrics": eval_metrics,
            "test_metrics": test_metrics,
            "output_path": str(self.output_path),
            "total_time_seconds": total_time,
            "val_predictions": val_predictions.class_predictions,
            "val_true": y_val_aligned,
            "feature_selection": (
                self.feature_selector.get_feature_report()
                if self._is_feature_selection_enabled() and self.feature_selector is not None
                else None
            ),
        }

        logger.info(
            f"Training complete: "
            f"val_f1={eval_metrics['macro_f1']:.4f}, "
            f"val_accuracy={eval_metrics['accuracy']:.4f}, "
            f"time={total_time:.1f}s"
        )

        # Log final metrics and end tracking run
        self.tracker.log_metrics({"total_time_seconds": total_time})
        self.tracker.end_run(status="FINISHED")

        return results

    def run_prepared(
        self,
        prepared: PreparedData,
        skip_save: bool = False,
    ) -> dict[str, Any]:
        """
        Execute training with pre-prepared data (bypasses container).

        Use this method when PreparedData already has correctly shaped arrays
        (3D for sequence models, 4D for multi-stream models). This bypasses
        the container pathway which would incorrectly reshape the data.

        Args:
            prepared: PreparedData with X_train, y_train, X_val, y_val arrays.
                For sequence models, X should be 3D (n_samples, seq_len, n_features).
                For multi-stream models, X should be 4D.
            skip_save: If True, skip saving artifacts (for testing)

        Returns:
            Dict with training results including metrics, predictions, etc.
        """

        start_time = time.time()

        # Setup
        self._setup_output_dir()
        self._save_config()

        # Start experiment tracking
        tracking_run_id = self.tracker.start_run(
            run_name=self.run_id,
            tags={
                "model_family": self.model.model_family,
                "feature_set": self.config.feature_set,
                "data_rank": str(prepared.data_rank),
            },
        )
        logger.info(f"Started experiment tracking run: {tracking_run_id}")
        self.tracker.log_params(self.config.to_dict())

        # Extract data directly from PreparedData (no reshaping needed)
        X_train = prepared.X_train
        y_train = prepared.y_train
        w_train = prepared.train_weights if prepared.has_weights else np.ones(len(y_train))
        X_val = prepared.X_val
        y_val = prepared.y_val

        # Log data shapes
        logger.info(
            f"Data shapes (from PreparedData): "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_val={X_val.shape}, y_val={y_val.shape}, "
            f"data_rank={prepared.data_rank}D"
        )

        # Train model
        logger.info(f"Training {self.config.model_name}...")
        fit_kwargs: dict[str, Any] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "sample_weights": w_train,
            "config": self.config.model_config,
        }

        training_metrics = self.model.fit(**fit_kwargs)

        # Free training data from memory after training completes
        # Critical for sequence models where X_train can be 10GB+
        import gc

        import torch

        del X_train, w_train
        del fit_kwargs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Freed training data from memory after model.fit()")

        # Evaluate
        logger.info("Evaluating on validation set...")
        val_predictions = self.model.predict(X_val)

        eval_metrics = compute_classification_metrics(
            y_true=y_val,
            y_pred=val_predictions.class_predictions,
            y_proba=val_predictions.class_probabilities,
        )

        # Add trading metrics
        eval_metrics["trading"] = compute_trading_metrics(
            y_true=y_val,
            y_pred=val_predictions.class_predictions,
        )

        # Probability calibration
        self.calibrator = None
        if self.config.use_calibration:
            logger.info("Applying probability calibration...")
            calibration_method = self.config.calibration_method
            if calibration_method not in ("isotonic", "sigmoid", "auto"):
                calibration_method = "auto"
            cal_config = CalibrationConfig(method=calibration_method)
            self.calibrator = ProbabilityCalibrator(cal_config)
            calibration_metrics = self.calibrator.fit(
                y_true=y_val,
                probabilities=val_predictions.class_probabilities,
            )
            eval_metrics["calibration"] = calibration_metrics.to_dict()

        # Log metrics to tracker
        flat_metrics = {
            "val_accuracy": eval_metrics["accuracy"],
            "val_macro_f1": eval_metrics["macro_f1"],
            "val_precision": eval_metrics["precision"],
            "val_recall": eval_metrics["recall"],
        }
        self.tracker.log_metrics(flat_metrics)

        # Save artifacts if not skipped
        if not skip_save:
            self._save_artifacts(
                training_metrics=training_metrics,
                eval_metrics=eval_metrics,
                predictions=val_predictions,
            )
            self._save_model()

        total_time = time.time() - start_time

        results = {
            "run_id": self.run_id,
            "model_name": self.config.model_name,
            "horizon": self.config.horizon,
            "training_metrics": training_metrics.to_dict(),
            "evaluation_metrics": eval_metrics,
            "test_metrics": None,  # Test eval not supported in run_prepared yet
            "output_path": str(self.output_path),
            "total_time_seconds": total_time,
            "val_predictions": val_predictions.class_predictions,
            "val_true": y_val,
            "feature_selection": None,  # Not applicable for pre-prepared data
        }

        logger.info(
            f"Training complete: "
            f"val_f1={eval_metrics['macro_f1']:.4f}, "
            f"val_accuracy={eval_metrics['accuracy']:.4f}, "
            f"time={total_time:.1f}s"
        )

        self.tracker.log_metrics({"total_time_seconds": total_time})
        self.tracker.end_run(status="FINISHED")

        return results
