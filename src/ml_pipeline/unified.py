"""
Unified ML Pipeline - Single entry point for entire ML workflow.

DEPRECATED: This module is deprecated in favor of src.factory.MLFactory.
MLFactory is now the ONE canonical entry point that internally delegates to
PipelineRunner for Phase 1 data preparation.

Migration Guide:
    # OLD (deprecated)
    from src.ml_pipeline import MLPipeline, MLConfig
    config = MLConfig(symbol="MES", horizons=[20], models=["xgboost"])
    pipeline = MLPipeline(config)
    results = pipeline.run()

    # NEW (recommended)
    from src.factory import MLFactory
    from src.core import PipelineConfig
    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./experiments",
        models=["xgboost"],
        horizons=[20],
    )
    factory = MLFactory(config)
    results = factory.run(df)

This module is kept for backward compatibility and will be removed in a future version.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import MLConfig
from .state import PipelineState

if TYPE_CHECKING:
    from src.pipeline.data_config import PipelineConfig
    from src.training.config import ExperimentConfig

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Unified ML pipeline orchestrating entire workflow.

    DEPRECATED: Use src.factory.MLFactory instead.
    MLFactory is now the ONE canonical entry point that internally delegates to
    PipelineRunner for Phase 1 data preparation.

    Migration:
        # OLD
        from src.ml_pipeline import MLPipeline, MLConfig
        pipeline = MLPipeline(config)

        # NEW
        from src.factory import MLFactory
        factory = MLFactory(config)

    Coordinates three main phases:
    1. Data Pipeline (Phase 1-5): Raw OHLCV → Labeled Features
    2. Training (Phase 6): Model training with various modes
    3. Evaluation (Phase 7): CV, walk-forward, CPCV-PBO

    Supports checkpointing and resumption from any phase.
    """

    def __init__(self, config: MLConfig | dict | str | Path):
        """
        Initialize ML pipeline.

        DEPRECATED: Use src.factory.MLFactory instead.

        Args:
            config: MLConfig object, dict, YAML path, or YAML string

        Raises:
            ValueError: If config type is invalid
            FileNotFoundError: If YAML path doesn't exist
        """
        import warnings
        warnings.warn(
            "MLPipeline is deprecated. Use src.factory.MLFactory instead. "
            "MLFactory is now the ONE canonical entry point that internally "
            "delegates to PipelineRunner for Phase 1 data preparation.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.config = self._normalize_config(config)
        self.state = PipelineState.load_or_create(self.config.run_id)

        # Setup logging
        self._setup_logging()

        logger.info(f"Initialized MLPipeline (run_id={self.config.run_id})")
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Horizons: {self.config.horizons}")
        logger.info(f"Models: {[m.name if hasattr(m, 'name') else m for m in self.config.models]}")
        logger.info(f"Training mode: {self.config.training_mode}")

    def _setup_logging(self) -> None:
        """Configure logging for unified pipeline."""
        log_dir = self.state.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "unified_pipeline.log"

        # Add file handler if not already present
        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            root_logger.addHandler(handler)

        logger.info(f"Logging to {log_file}")

    def _normalize_config(self, config: MLConfig | dict | str | Path) -> MLConfig:
        """
        Convert various config formats to MLConfig.

        Args:
            config: Config in various formats

        Returns:
            Normalized MLConfig object

        Raises:
            ValueError: If config type is invalid
        """
        if isinstance(config, MLConfig):
            return config
        elif isinstance(config, dict):
            return MLConfig(**config)
        elif isinstance(config, (str, Path)):
            path = Path(config)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            return MLConfig.from_yaml(path)
        else:
            raise ValueError(
                f"Invalid config type: {type(config)}. " f"Expected MLConfig, dict, str, or Path."
            )

    def run(self) -> dict[str, Any]:
        """
        Run full pipeline: data → training → evaluation.

        Returns:
            Dict with keys:
                - data: Data pipeline results
                - training: Training results
                - evaluation: Evaluation results (if enabled)
                - run_id: Run identifier
                - output_dir: Path to outputs

        Raises:
            Exception: If any phase fails
        """
        logger.info("\n" + "=" * 70)
        logger.info("UNIFIED ML PIPELINE - FULL RUN")
        logger.info("=" * 70)

        results = {
            "run_id": self.config.run_id,
            "output_dir": str(self.state.output_dir),
        }

        try:
            # Phase 1-5: Data pipeline
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 1-5: DATA PIPELINE")
            logger.info("=" * 70)
            results["data"] = self.run_data()

            # Phase 6: Training
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 6: MODEL TRAINING")
            logger.info("=" * 70)
            results["training"] = self.run_training()

            # Phase 7: Evaluation (optional)
            if self.config.evaluation_methods:
                logger.info("\n" + "=" * 70)
                logger.info("PHASE 7: MODEL EVALUATION")
                logger.info("=" * 70)
                results["evaluation"] = self.run_evaluation()

            # Mark pipeline complete
            self.state.status = "completed"
            self.state.save()

            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Run ID: {self.config.run_id}")
            logger.info(f"Output: {self.state.output_dir}")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.state.status = "failed"
            self.state.error_message = str(e)
            self.state.save()
            raise

    def run_data(self) -> dict[str, Any]:
        """
        Run Phase 1-5: Raw OHLCV → Labeled Features.

        Executes the data pipeline:
        1. Ingest raw OHLCV
        2. Clean and resample
        3. Engineer features
        4. Generate labels (triple-barrier)
        5. Create splits with purge/embargo
        6. Scale features

        Returns:
            Dict with keys:
                - success: Whether pipeline completed successfully
                - labeled_data_paths: Paths to labeled data by timeframe
                - features_info: Feature metadata
                - split_info: Train/val/test split information
                - run_id: Pipeline run identifier

        Raises:
            Exception: If pipeline execution fails
        """
        phase_name = "data"

        # Check if already complete
        if self.state.is_complete(phase_name):
            logger.info(f"Phase '{phase_name}' already complete, loading results")
            return self.state.get_result(phase_name)

        logger.info(f"Starting phase: {phase_name}")
        self.state.mark_start(phase_name)

        try:
            # Convert MLConfig → PipelineConfig
            from src.pipeline.data_config import PipelineConfig

            pipeline_config = self._mlconfig_to_pipeline_config()

            # Import and run pipeline
            from src.pipeline.runner import PipelineRunner

            runner = PipelineRunner(pipeline_config, resume=False)
            success = runner.run()

            if not success:
                raise RuntimeError("Data pipeline failed")

            # Extract results
            results = {
                "success": success,
                "run_id": pipeline_config.run_id,
                "labeled_data_paths": self._get_labeled_data_paths(pipeline_config),
                "features_info": self._get_features_info(pipeline_config),
                "split_info": self._get_split_info(pipeline_config),
                "manifest_path": str(runner.manifest.manifest_path),
            }

            # Mark complete and save
            self.state.mark_complete(phase_name, results)

            logger.info(f"Phase '{phase_name}' completed successfully")
            return results

        except Exception as e:
            logger.error(f"Phase '{phase_name}' failed: {e}", exc_info=True)
            self.state.mark_failed(phase_name, str(e))
            raise

    def run_training(self) -> dict[str, Any]:
        """
        Run Phase 6: Model Training.

        Executes training based on training_mode:
        - standard: Basic training with TrainingOrchestrator
        - walk_forward: Walk-forward training (TODO: extract from scripts)
        - regime_aware: Regime-specific training (TODO: extract from scripts)
        - meta_labeling: Meta-labeling approach (TODO: extract from scripts)

        Returns:
            Dict with keys:
                - model_paths: Paths to trained models
                - performance_metrics: Model performance metrics
                - ensemble_path: Path to ensemble (if built)
                - training_mode: Which training mode was used

        Raises:
            RuntimeError: If data phase not complete
            Exception: If training fails
        """
        phase_name = "training"

        # Verify data phase complete
        if not self.state.is_complete("data"):
            raise RuntimeError("Cannot run training: data phase not complete")

        # Check if already complete
        if self.state.is_complete(phase_name):
            logger.info(f"Phase '{phase_name}' already complete, loading results")
            return self.state.get_result(phase_name)

        logger.info(f"Starting phase: {phase_name}")
        self.state.mark_start(phase_name)

        try:
            # Dispatch based on training mode
            if self.config.training_mode == "standard":
                results = self._run_standard_training()
            elif self.config.training_mode == "walk_forward":
                results = self._run_walk_forward_training()
            elif self.config.training_mode == "regime_aware":
                results = self._run_regime_aware_training()
            elif self.config.training_mode == "meta_labeling":
                results = self._run_meta_labeling_training()
            else:
                raise ValueError(f"Unknown training mode: {self.config.training_mode}")

            # Add training mode to results
            results["training_mode"] = self.config.training_mode

            # Mark complete and save
            self.state.mark_complete(phase_name, results)

            logger.info(f"Phase '{phase_name}' completed successfully")
            return results

        except Exception as e:
            logger.error(f"Phase '{phase_name}' failed: {e}", exc_info=True)
            self.state.mark_failed(phase_name, str(e))
            raise

    def run_evaluation(self) -> dict[str, Any]:
        """
        Run Phase 7: Model Evaluation.

        Executes evaluation methods specified in config.evaluation_methods:
        - cv: Cross-validation with PurgedKFold
        - walk_forward: Walk-forward validation
        - cpcv_pbo: CPCV-PBO validation

        Returns:
            Dict with keys for each evaluation method:
                - cv: Cross-validation results
                - walk_forward: Walk-forward results
                - cpcv_pbo: CPCV-PBO results

        Raises:
            RuntimeError: If training phase not complete
            Exception: If evaluation fails
        """
        phase_name = "evaluation"

        # Verify training phase complete
        if not self.state.is_complete("training"):
            raise RuntimeError("Cannot run evaluation: training phase not complete")

        # Check if already complete
        if self.state.is_complete(phase_name):
            logger.info(f"Phase '{phase_name}' already complete, loading results")
            return self.state.get_result(phase_name)

        logger.info(f"Starting phase: {phase_name}")
        self.state.mark_start(phase_name)

        try:
            results = {}

            for method in self.config.evaluation_methods:
                logger.info(f"Running evaluation method: {method}")

                if method == "cv":
                    results["cv"] = self._run_cv_evaluation()
                elif method == "walk_forward":
                    results["walk_forward"] = self._run_walk_forward_evaluation()
                elif method == "cpcv_pbo":
                    results["cpcv_pbo"] = self._run_cpcv_pbo_evaluation()
                else:
                    logger.warning(f"Unknown evaluation method: {method}")

            # Mark complete and save
            self.state.mark_complete(phase_name, results)

            logger.info(f"Phase '{phase_name}' completed successfully")
            return results

        except Exception as e:
            logger.error(f"Phase '{phase_name}' failed: {e}", exc_info=True)
            self.state.mark_failed(phase_name, str(e))
            raise

    def resume(self) -> dict[str, Any]:
        """
        Resume pipeline from last checkpoint.

        Checks state to find last incomplete phase and continues from there.

        Returns:
            Dict with results from all phases executed

        Raises:
            Exception: If any phase fails
        """
        logger.info(f"Resuming pipeline from run_id: {self.config.run_id}")
        logger.info(f"Current status: {self.state.status}")

        # Get summary
        summary = self.state.get_summary()
        logger.info(f"Completed phases: {summary['completed_phases']}")
        logger.info(f"Pending phases: {summary['pending_phases']}")

        results = {
            "run_id": self.config.run_id,
            "resumed": True,
        }

        # Run incomplete phases in order
        if not self.state.is_complete("data"):
            results["data"] = self.run_data()

        if not self.state.is_complete("training"):
            results["training"] = self.run_training()

        if self.config.evaluation_methods and not self.state.is_complete("evaluation"):
            results["evaluation"] = self.run_evaluation()

        logger.info("Resume completed")
        return results

    # ========================================================================
    # PRIVATE METHODS - Config Conversion
    # ========================================================================

    def _mlconfig_to_pipeline_config(self) -> "PipelineConfig":
        """
        Convert MLConfig to PipelineConfig for Phase 1 pipeline.

        Returns:
            PipelineConfig object

        Note:
            Some MLConfig fields may not map directly to PipelineConfig.
            Unmapped fields will use PipelineConfig defaults.
        """
        from src.pipeline.data_config import PipelineConfig

        # Use MLConfig.run_id if available, otherwise generate new one
        pipeline_run_id = self.config.run_id

        # Build feature_toggles dict from MLConfig feature flags
        feature_toggles = {
            "wavelets": self.config.enable_wavelets,
            "microstructure": self.config.enable_microstructure,
            "volume": self.config.enable_volume,
            "volatility": self.config.enable_volatility,
        }

        # Build barrier_overrides if specified
        barrier_overrides = None
        if self.config.k_up is not None or self.config.k_down is not None:
            barrier_overrides = {}
            if self.config.k_up is not None:
                barrier_overrides["k_up"] = self.config.k_up
            if self.config.k_down is not None:
                barrier_overrides["k_down"] = self.config.k_down

        config = PipelineConfig(
            # Run identification
            run_id=pipeline_run_id,
            # Data configuration
            symbols=[self.config.symbol] if self.config.symbol else self.config.symbols or ["MES"],
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            # Timeframe configuration
            target_timeframe=self.config.target_timeframe,
            output_timeframes=self.config.output_timeframes,
            process_all_timeframes=self.config.process_all_timeframes,
            # MTF
            mtf_mode=self.config.mtf_mode,
            mtf_timeframes=self.config.mtf_timeframes or [],
            # Labeling
            label_horizons=self.config.horizons,
            # Splits
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
            # Scaling
            scaler_type=self.config.scaler_type,
            # Random seed
            random_seed=self.config.random_seed,
            # Feature toggles (wavelets, microstructure, etc.)
            feature_toggles=feature_toggles,
            # Barrier overrides (k_up, k_down)
            barrier_overrides=barrier_overrides,
            # Paths (use MLConfig data_dir)
            project_root=self.config.data_dir.parent if self.config.data_dir else Path("."),
        )

        return config

    def _mlconfig_to_experiment_config(self) -> "ExperimentConfig":
        """
        Convert MLConfig to ExperimentConfig for training orchestration.

        Returns:
            ExperimentConfig object
        """
        from src.training.config import ExperimentConfig

        # Get data paths from completed data phase
        data_result = self.state.get_result("data")
        if not data_result:
            raise RuntimeError("Data phase results not found")

        config = ExperimentConfig(
            symbol=self.config.symbol,
            horizons=self.config.horizons,
            models=self.config.models,
            data_dir=self.config.data_dir,
            output_dir=self.config.output_dir / self.config.run_id,
            cross_validate=self.config.cross_validate,
            cv_splits=self.config.cv_splits,
            build_ensemble=self.config.build_ensemble,
            ensemble_method=self.config.ensemble_method,
            meta_learner=self.config.meta_learner,
            global_feature_optimization=self.config.global_feature_optimization,
            global_hyperparam_optimization=self.config.global_hyperparam_optimization,
            parallel_training=self.config.parallel_training,
            n_jobs=self.config.n_jobs,
        )

        return config

    # ========================================================================
    # PRIVATE METHODS - Training Mode Dispatchers
    # ========================================================================

    def _run_standard_training(self) -> dict[str, Any]:
        """
        Run standard training mode using TrainingOrchestrator.

        Returns:
            Training results dict
        """
        from src.training.orchestrator import TrainingOrchestrator

        experiment_config = self._mlconfig_to_experiment_config()

        orchestrator = TrainingOrchestrator(experiment_config)
        results = orchestrator.run()

        return {
            "model_paths": results.get("model_paths", {}),
            "performance_metrics": results.get("metrics", {}),
            "ensemble_path": results.get("ensemble_path"),
            "output_dir": str(orchestrator.output_dir),
        }

    def _run_walk_forward_training(self) -> dict[str, Any]:
        """
        Run walk-forward training mode.

        Returns:
            Training results dict
        """
        from src.training.modes import WalkForwardTrainer

        experiment_config = self._mlconfig_to_experiment_config()

        trainer = WalkForwardTrainer(experiment_config)
        results = trainer.run()

        return {
            "model_paths": results.get("model_paths", {}),
            "performance_metrics": results.get("metrics", {}),
            "fold_results": results.get("fold_results", []),
            "output_dir": str(trainer.output_dir),
        }

    def _run_regime_aware_training(self) -> dict[str, Any]:
        """
        Run regime-aware training mode.

        Returns:
            Training results dict
        """
        from src.training.modes import RegimeAwareTrainer

        experiment_config = self._mlconfig_to_experiment_config()

        trainer = RegimeAwareTrainer(experiment_config)
        results = trainer.run()

        return {
            "model_paths": results.get("model_paths", {}),
            "performance_metrics": results.get("metrics", {}),
            "regime_info": results.get("regime_info", {}),
            "output_dir": str(trainer.output_dir),
        }

    def _run_meta_labeling_training(self) -> dict[str, Any]:
        """
        Run meta-labeling training mode.

        Returns:
            Training results dict
        """
        from src.training.modes import MetaLabelingTrainer

        experiment_config = self._mlconfig_to_experiment_config()

        trainer = MetaLabelingTrainer(experiment_config)
        results = trainer.run()

        return {
            "base_model_path": results.get("base_model_path"),
            "meta_model_path": results.get("meta_model_path"),
            "performance_metrics": results.get("metrics", {}),
            "base_metrics": results.get("base_metrics", {}),
            "meta_metrics": results.get("meta_metrics", {}),
            "output_dir": str(trainer.output_dir),
        }

    # ========================================================================
    # PRIVATE METHODS - Evaluation Dispatchers
    # ========================================================================

    def _run_cv_evaluation(self) -> dict[str, Any]:
        """
        Run cross-validation evaluation.

        Returns:
            CV results dict
        """
        from src.evaluation import CVEvaluator

        training_result = self.state.get_result("training")

        eval_config = {
            "output_dir": self.state.output_dir / "evaluation" / "cv",
            "model_paths": training_result.get("model_paths", {}),
            "cv_splits": self.config.cv_splits,
            "purge_bars": self.config.cv_purge_bars,
            "embargo_bars": self.config.cv_embargo_bars,
        }

        evaluator = CVEvaluator(eval_config)
        results = evaluator.run()

        return {
            "fold_metrics": results.get("fold_metrics", {}),
            "aggregate_metrics": results.get("aggregate_metrics", {}),
            "oof_predictions": results.get("oof_predictions"),
        }

    def _run_walk_forward_evaluation(self) -> dict[str, Any]:
        """
        Run walk-forward evaluation.

        Returns:
            Walk-forward results dict
        """
        from src.evaluation import WalkForwardEvaluator

        training_result = self.state.get_result("training")

        eval_config = {
            "output_dir": self.state.output_dir / "evaluation" / "walk_forward",
            "model_paths": training_result.get("model_paths", {}),
            "window_size": self.config.walk_forward_window_size,
            "step_size": self.config.walk_forward_step_size,
            "min_train_size": self.config.walk_forward_min_train_size,
        }

        evaluator = WalkForwardEvaluator(eval_config)
        results = evaluator.run()

        return {
            "window_metrics": results.get("window_metrics", {}),
            "aggregate_metrics": results.get("aggregate_metrics", {}),
            "predictions": results.get("predictions"),
        }

    def _run_cpcv_pbo_evaluation(self) -> dict[str, Any]:
        """
        Run CPCV-PBO evaluation.

        Returns:
            CPCV-PBO results dict
        """
        from src.evaluation import CPCVPBOEvaluator

        training_result = self.state.get_result("training")

        eval_config = {
            "output_dir": self.state.output_dir / "evaluation" / "cpcv_pbo",
            "model_paths": training_result.get("model_paths", {}),
            "n_splits": self.config.cpcv_pbo_n_splits,
            "n_tests": self.config.cpcv_pbo_n_tests,
        }

        evaluator = CPCVPBOEvaluator(eval_config)
        results = evaluator.run()

        return {
            "pbo_score": results.get("pbo_score"),
            "performance_degradation": results.get("performance_degradation"),
            "lambda_values": results.get("lambda_values"),
            "combinatorial_metrics": results.get("combinatorial_metrics", {}),
        }

    # ========================================================================
    # PRIVATE METHODS - Data Extraction Helpers
    # ========================================================================

    def _get_labeled_data_paths(self, pipeline_config: "PipelineConfig") -> dict[str, str]:
        """
        Get paths to labeled data by timeframe.

        Args:
            pipeline_config: Completed pipeline config

        Returns:
            Dict mapping timeframe to labeled data path
        """
        paths = {}

        # Determine which timeframes were processed
        if pipeline_config.process_all_timeframes or pipeline_config.output_timeframes:
            timeframes = pipeline_config.output_timeframes or [pipeline_config.target_timeframe]
        else:
            timeframes = [pipeline_config.target_timeframe]

        for tf in timeframes:
            features_path = pipeline_config.data_features_dir / f"features_{tf}.parquet"
            if features_path.exists():
                paths[tf] = str(features_path)

        return paths

    def _get_features_info(self, pipeline_config: "PipelineConfig") -> dict[str, Any]:
        """
        Extract feature information from pipeline.

        Args:
            pipeline_config: Completed pipeline config

        Returns:
            Dict with feature metadata
        """
        import pandas as pd

        info = {}

        # Get feature list from any timeframe
        tf = pipeline_config.target_timeframe
        features_path = pipeline_config.data_features_dir / f"features_{tf}.parquet"

        if features_path.exists():
            df = pd.read_parquet(features_path, columns=[])  # Just get column names
            feature_cols = [c for c in df.columns if c not in ["timestamp", "target", "quality"]]

            info["n_features"] = len(feature_cols)
            info["feature_names"] = feature_cols
            info["timeframe"] = tf

        return info

    def _get_split_info(self, pipeline_config: "PipelineConfig") -> dict[str, Any]:
        """
        Extract split information from pipeline.

        Args:
            pipeline_config: Completed pipeline config

        Returns:
            Dict with split metadata
        """
        info = {
            "train_ratio": pipeline_config.train_ratio,
            "val_ratio": pipeline_config.val_ratio,
            "test_ratio": pipeline_config.test_ratio,
            "purge_bars": pipeline_config.purge_bars,
            "embargo_bars": pipeline_config.embargo_bars,
        }

        # Try to get actual split sizes
        tf = pipeline_config.target_timeframe
        scaled_dir = pipeline_config.data_splits_dir / "scaled" / tf

        for split in ["train", "val", "test"]:
            path = scaled_dir / f"{split}_scaled.parquet"
            if path.exists():
                import pandas as pd

                df = pd.read_parquet(path, columns=[])
                info[f"{split}_size"] = len(df)

        return info
