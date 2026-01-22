"""
DEPRECATED: TrainingOrchestrator is deprecated in favor of UnifiedTrainingOrchestrator.

This module is maintained for backward compatibility only.
Please migrate to:
    from src.models.training import UnifiedTrainingOrchestrator

Migration Guide:
    # Old usage (deprecated)
    from src.models.training import TrainingOrchestrator
    orchestrator = TrainingOrchestrator(config_dict)
    results = orchestrator.run()

    # New usage (preferred)
    from src.core import PipelineConfig
    from src.models.training import UnifiedTrainingOrchestrator

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./experiments",
        models=["xgboost", "lightgbm"],
    )
    orchestrator = UnifiedTrainingOrchestrator(config)
    result = orchestrator.train(df)

This module will be removed in v2.0.0.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.container import TimeSeriesDataContainer

from .config import ExperimentConfig, ModelConfig
from .config_loader import ConfigLoader
from .feature_selector import FeatureSelector
from .model_factory import ModelFactory
from .trainer import Trainer

logger = logging.getLogger(__name__)


def _emit_deprecation_warning():
    """Emit deprecation warning for TrainingOrchestrator."""
    warnings.warn(
        "TrainingOrchestrator is deprecated. Use UnifiedTrainingOrchestrator instead. "
        "This class will be removed in v2.0.0. "
        "See src/models/training/orchestrator.py for migration guide.",
        DeprecationWarning,
        stacklevel=3,
    )


class TrainingOrchestrator:
    """
    DEPRECATED: Use UnifiedTrainingOrchestrator instead.

    Master controller for unified ML training system.
    This class is maintained for backward compatibility only.

    Migration:
        # Old (deprecated)
        orchestrator = TrainingOrchestrator(config_dict)
        results = orchestrator.run()

        # New (preferred)
        from src.core import PipelineConfig
        from src.models.training import UnifiedTrainingOrchestrator
        config = PipelineConfig(...)
        orchestrator = UnifiedTrainingOrchestrator(config)
        result = orchestrator.train(df)
    """

    def __init__(self, config: dict[str, Any] | ExperimentConfig):
        """
        Initialize orchestrator with configuration.

        Args:
            config: Validated configuration dict or ExperimentConfig
        """
        _emit_deprecation_warning()

        self.experiment_config: ExperimentConfig | None
        self.config: dict[str, Any] | None

        if isinstance(config, ExperimentConfig):
            self.experiment_config = config
            self.config = None
            self.experiment_name = f"{config.symbol}_Experiment"
            self.output_dir = config.output_dir / self._generate_run_id()
        else:
            ConfigLoader.validate(config)
            self.experiment_config = None
            self.config = config
            self.experiment_name = config["experiment"]["name"]
            self.output_dir = Path(config["output"]["save_dir"]) / self._generate_run_id()

        self.results: dict[str, dict[str, Any]] = {}
        self.trained_models: dict[str, Any] = {}
        self.feature_selector = FeatureSelector()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[DEPRECATED] Initialized TrainingOrchestrator: {self.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.warning(
            "TrainingOrchestrator is deprecated. Please migrate to UnifiedTrainingOrchestrator."
        )

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def run(self) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING EXPERIMENT: {self.experiment_name}")
        logger.info(f"{'='*60}\n")

        if self.experiment_config:
            return self._run_with_experiment_config()
        else:
            return self._run_with_dict_config()

    def _run_with_experiment_config(self) -> dict[str, Any]:
        cfg = self.experiment_config
        if cfg is None:
            raise RuntimeError("experiment_config is None")

        for horizon in cfg.horizons:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING HORIZON: {horizon}")
            logger.info(f"{'='*60}")

            horizon_results: dict[str, Any] = {}

            # Convert models to ModelConfig if they are strings
            model_configs: list[ModelConfig] = []
            for m in cfg.models:
                if isinstance(m, str):
                    model_configs.append(ModelConfig(name=m))
                else:
                    model_configs.append(m)

            for model_cfg in model_configs:
                logger.info(f"\nTraining model: {model_cfg.name}")

                timeframe = model_cfg.timeframe or "5min"
                container = self._load_data_for_timeframe(horizon, timeframe)

                # Apply feature set filtering if specified (from smart_config)
                if model_cfg.features:
                    logger.info(f"  Applying feature set: {model_cfg.features}")
                    container = self._filter_container_by_feature_mode(
                        container, model_cfg.features, horizon
                    )

                if model_cfg.optimize_features:
                    from src.data.features.optimization import optimize_features_for_model

                    logger.info(f"  Optimizing features for {model_cfg.name}...")

                    X_train_df, y_train, _ = container.get_sklearn_arrays("train", return_df=True)
                    X_val_df, y_val, _ = container.get_sklearn_arrays("val", return_df=True)

                    opt_result = optimize_features_for_model(
                        model_cfg.name,
                        X_train_df,
                        y_train,
                        X_val_df,
                        y_val,
                        n_trials=model_cfg.feature_opt_trials,
                    )
                    logger.info(
                        f"  Optimized features: {len(opt_result.optimized_features)} (from {len(opt_result.baseline_features)})"
                    )

                    container = self._filter_container_to_features(
                        container, opt_result.optimized_features, horizon
                    )

                model_results = self._train_model_simple(container, model_cfg, horizon)
                horizon_results[model_cfg.name] = model_results
                self.trained_models[f"{model_cfg.name}_h{horizon}"] = model_results["trainer"]

            if cfg.build_ensemble and len(horizon_results) > 1:
                logger.info(f"\nBuilding ensemble for horizon {horizon}")
                ensemble_container = self._load_data_for_timeframe(horizon, "5min")
                ensemble_results = self._build_ensemble_simple(
                    ensemble_container, horizon_results, horizon, cfg
                )
                horizon_results["ensemble"] = ensemble_results

            self.results[f"horizon_{horizon}"] = horizon_results

        self._save_results()

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT COMPLETE: {self.experiment_name}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*60}\n")

        return self.results

    def _filter_container_by_feature_mode(
        self,
        container: TimeSeriesDataContainer,
        feature_mode: str,
        horizon: int,
    ) -> TimeSeriesDataContainer:
        """
        Filter container based on smart_config feature mode.

        Args:
            container: Original container with all features
            feature_mode: Feature mode from MODEL_DEFAULTS ("full", "standard", "sequence", "raw", etc.)
            horizon: Label horizon

        Returns:
            New container with filtered features
        """
        import pandas as pd

        X_train_result, y_train, w_train = container.get_sklearn_arrays("train", return_df=True)
        # Ensure we have a DataFrame
        X_train_df = (
            X_train_result
            if isinstance(X_train_result, pd.DataFrame)
            else pd.DataFrame(X_train_result)
        )

        # Use feature selector to filter
        X_train_filtered = self.feature_selector.select_features(X_train_df, mode=feature_mode)
        feature_list = list(X_train_filtered.columns)

        if len(feature_list) == len(X_train_df.columns):
            logger.info(f"  Feature mode '{feature_mode}': using all {len(feature_list)} features")
            return container

        logger.info(
            f"  Feature mode '{feature_mode}': {len(feature_list)} features (from {len(X_train_df.columns)})"
        )

        return self._filter_container_to_features(container, feature_list, horizon)

    def _filter_container_to_features(
        self,
        container: TimeSeriesDataContainer,
        feature_list: list[str],
        horizon: int,
    ) -> TimeSeriesDataContainer:
        """
        Filter container to use only specified features.

        Args:
            container: Original container with all features
            feature_list: Subset of features to keep
            horizon: Label horizon

        Returns:
            New container with filtered feature set
        """
        import pandas as pd

        from src.core.container import TimeSeriesDataContainer

        split_dfs: dict[str, pd.DataFrame] = {}

        for split in container.available_splits:
            X_result, y, w = container.get_sklearn_arrays(split, return_df=True)
            # Ensure we have a DataFrame
            X_df = X_result if isinstance(X_result, pd.DataFrame) else pd.DataFrame(X_result)

            X_filtered = X_df[feature_list]

            df_filtered = X_filtered.copy()
            df_filtered[f"label_h{horizon}"] = y
            df_filtered[f"sample_weight_h{horizon}"] = w

            split_dfs[split] = df_filtered

        filtered_container = TimeSeriesDataContainer.from_dataframes(
            train_df=split_dfs.get("train"),
            val_df=split_dfs.get("val"),
            test_df=split_dfs.get("test"),
            horizon=horizon,
            feature_columns=feature_list,
        )

        logger.info(
            f"Filtered container: {len(feature_list)} features (from {container.n_features})"
        )

        return filtered_container

    def _load_data_for_timeframe(self, horizon: int, timeframe: str) -> TimeSeriesDataContainer:
        """
        Load data for specific timeframe and horizon.

        Args:
            horizon: Label horizon (bars ahead)
            timeframe: Timeframe string (e.g., '5min', '15min', '1h')

        Returns:
            TimeSeriesDataContainer with train/val/test splits
        """
        cfg = self.experiment_config
        if cfg is None:
            raise RuntimeError("experiment_config is None")
        data_dir = cfg.data_dir / timeframe

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Run pipeline with --process-all-timeframes or --output-timeframes {timeframe}"
            )

        logger.info(f"Loading data: {data_dir}, horizon={horizon}")
        container = TimeSeriesDataContainer.from_parquet_dir(data_dir, horizon=horizon)

        # Get shapes from the new container API
        train_split = container.get_split("train")
        val_split = container.get_split("val")
        test_split = container.get_split("test")
        logger.info(
            f"Data loaded: Train={train_split.n_samples}x{train_split.n_features}, "
            f"Val={val_split.n_samples}x{val_split.n_features}, "
            f"Test={test_split.n_samples}x{test_split.n_features}"
        )

        return container

    def _train_model_simple(
        self,
        container: TimeSeriesDataContainer,
        model_cfg: ModelConfig,
        horizon: int,
    ) -> dict[str, Any]:
        """
        Train single model with new ModelConfig interface.

        Args:
            container: Data container
            model_cfg: Model configuration
            horizon: Label horizon

        Returns:
            Dict with trainer, results, and config
        """
        from src.models.config import TrainerConfig

        # Build trainer config with proper type handling
        trainer_kwargs: dict[str, Any] = {
            "model_name": model_cfg.name,
            "horizon": horizon,
        }
        if model_cfg.sequence_length is not None:
            trainer_kwargs["sequence_length"] = model_cfg.sequence_length

        trainer_config = TrainerConfig(**trainer_kwargs)

        if model_cfg.optimize_hyperparams:
            logger.info(f"  Hyperparameter optimization enabled for {model_cfg.name}")
            trainer_config = self._optimize_hyperparams_simple(container, trainer_config, model_cfg)

        trainer = Trainer(trainer_config)
        training_results = trainer.run(container)

        logger.info(
            f"  {model_cfg.name} Val F1: {training_results['evaluation_metrics']['val_f1']:.4f}"
        )

        return {
            "trainer": trainer,
            "results": training_results,
            "config": trainer_config,
        }

    def _optimize_hyperparams_simple(
        self,
        container: TimeSeriesDataContainer,
        trainer_config: Any,
        model_cfg: ModelConfig,
    ) -> Any:
        """
        Run hyperparameter optimization for ModelConfig-based training.

        Args:
            container: Data container
            trainer_config: Base trainer config
            model_cfg: Model configuration with optimization settings

        Returns:
            Updated trainer config with best params
        """
        import pandas as pd

        logger.info("    Running Optuna hyperparameter optimization...")

        from src.validation.cv.cv_tuner import TimeSeriesOptunaTuner
        from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

        # Get training data from container using the new API
        X_train_result, y_train_result, _ = container.get_sklearn_arrays("train", return_df=True)

        # Ensure DataFrame/Series types for tuner
        X_train = (
            X_train_result
            if isinstance(X_train_result, pd.DataFrame)
            else pd.DataFrame(X_train_result)
        )
        y_train = (
            y_train_result
            if isinstance(y_train_result, pd.Series)
            else pd.Series(y_train_result)
        )

        # Create purged CV splitter
        cv_config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=10)
        cv = PurgedKFold(cv_config)

        tuner = TimeSeriesOptunaTuner(
            model_name=trainer_config.model_name,
            cv=cv,
            n_trials=model_cfg.hyperparam_opt_trials,
        )

        # Use tune() method instead of optimize()
        result = tuner.tune(X=X_train, y=y_train)
        best_params = result.get("best_params", {})

        logger.info(f"    Best params: {best_params}")

        for param, value in best_params.items():
            setattr(trainer_config, param, value)

        return trainer_config

    def _build_ensemble_simple(
        self,
        container: TimeSeriesDataContainer,
        horizon_results: dict[str, Any],
        horizon: int,
        cfg: ExperimentConfig,
    ) -> dict[str, Any]:
        """
        Build ensemble from trained models using ExperimentConfig.

        Args:
            container: Data container for ensemble training
            horizon_results: Results from base models
            horizon: Label horizon
            cfg: Experiment configuration

        Returns:
            Dict with ensemble trainer, results, and config
        """
        from src.models.config import TrainerConfig

        ensemble_method = cfg.ensemble_method or "stacking"
        meta_learner = cfg.meta_learner or "ridge_meta"

        logger.info(f"  Ensemble method: {ensemble_method}, meta-learner: {meta_learner}")

        base_model_names = list(horizon_results.keys())

        if ensemble_method == "stacking":
            # Pass ensemble-specific config via model_config dict
            ensemble_config = TrainerConfig(
                model_name="stacking",
                horizon=horizon,
                model_config={
                    "base_models": base_model_names,
                    "meta_learner": meta_learner,
                },
            )

            ensemble_trainer = Trainer(ensemble_config)
            ensemble_results = ensemble_trainer.run(container)

            logger.info(
                f"  Ensemble Val F1: {ensemble_results['evaluation_metrics']['val_f1']:.4f}"
            )

            return {
                "trainer": ensemble_trainer,
                "results": ensemble_results,
                "method": ensemble_method,
            }

        return {}

    def _run_with_dict_config(self) -> dict[str, Any]:
        if self.config is None:
            raise RuntimeError("config is None")

        for horizon in self.config["data"]["horizons"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING HORIZON: {horizon}")
            logger.info(f"{'='*60}")

            container = self._load_data(horizon)
            container = self._apply_feature_selection(container)

            horizon_results: dict[str, Any] = {}

            for model_config in self.config["models"]["model_list"]:
                model_name = model_config["name"]
                logger.info(f"\nTraining model: {model_name}")

                model_results = self._train_single_model(container, model_config, horizon)

                horizon_results[model_name] = model_results
                self.trained_models[f"{model_name}_h{horizon}"] = model_results["trainer"]

            if self.config["ensemble"].get("enabled", False) and len(horizon_results) > 1:
                logger.info(f"\nBuilding ensemble for horizon {horizon}")
                ensemble_results = self._build_ensemble(container, horizon_results, horizon)
                horizon_results["ensemble"] = ensemble_results

            self.results[f"horizon_{horizon}"] = horizon_results

        self._save_results()

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT COMPLETE: {self.experiment_name}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*60}\n")

        return self.results

    def _load_data(self, horizon: int) -> TimeSeriesDataContainer:
        """Load data for given horizon."""
        if self.config is None:
            raise RuntimeError("config is None")

        data_dir = self.config["data"]["data_dir"]
        logger.info(f"Loading data from: {data_dir}, horizon: {horizon}")

        container = TimeSeriesDataContainer.from_parquet_dir(data_dir, horizon=horizon)

        # Get shapes from the new container API
        train_split = container.get_split("train")
        val_split = container.get_split("val")
        test_split = container.get_split("test")
        logger.info(
            f"Data loaded: Train={train_split.n_samples}x{train_split.n_features}, "
            f"Val={val_split.n_samples}x{val_split.n_features}, "
            f"Test={test_split.n_samples}x{test_split.n_features}"
        )

        return container

    def _apply_feature_selection(
        self, container: TimeSeriesDataContainer
    ) -> TimeSeriesDataContainer:
        """Apply feature selection based on config mode."""
        import pandas as pd

        if self.config is None:
            raise RuntimeError("config is None")

        feature_mode = self.config["features"]["mode"]
        mtf_strategy = self.config["features"].get("mtf_strategy", "indicators")

        logger.info(f"Applying feature selection: mode={feature_mode}, mtf_strategy={mtf_strategy}")

        # Get data using the new API
        X_train_result, y_train, w_train = container.get_sklearn_arrays("train", return_df=True)
        X_val_result, y_val, w_val = container.get_sklearn_arrays("val", return_df=True)
        X_test_result, y_test, w_test = container.get_sklearn_arrays("test", return_df=True)

        # Ensure DataFrames
        X_train_df = X_train_result if isinstance(X_train_result, pd.DataFrame) else pd.DataFrame(X_train_result)
        X_val_df = X_val_result if isinstance(X_val_result, pd.DataFrame) else pd.DataFrame(X_val_result)
        X_test_df = X_test_result if isinstance(X_test_result, pd.DataFrame) else pd.DataFrame(X_test_result)

        X_train_selected = self.feature_selector.select_features(
            X_train_df, mode=feature_mode, mtf_strategy=mtf_strategy
        )
        X_val_selected = self.feature_selector.select_features(
            X_val_df, mode=feature_mode, mtf_strategy=mtf_strategy
        )
        X_test_selected = self.feature_selector.select_features(
            X_test_df, mode=feature_mode, mtf_strategy=mtf_strategy
        )

        logger.info(
            f"Features selected: {len(X_train_selected.columns)} (from {len(X_train_df.columns)})"
        )

        # Build split DataFrames with labels and weights
        horizon = container.horizon
        train_df = X_train_selected.copy()
        train_df[f"label_h{horizon}"] = y_train.values if hasattr(y_train, "values") else y_train
        train_df[f"sample_weight_h{horizon}"] = w_train.values if hasattr(w_train, "values") else w_train

        val_df = X_val_selected.copy()
        val_df[f"label_h{horizon}"] = y_val.values if hasattr(y_val, "values") else y_val
        val_df[f"sample_weight_h{horizon}"] = w_val.values if hasattr(w_val, "values") else w_val

        test_df = X_test_selected.copy()
        test_df[f"label_h{horizon}"] = y_test.values if hasattr(y_test, "values") else y_test
        test_df[f"sample_weight_h{horizon}"] = w_test.values if hasattr(w_test, "values") else w_test

        return TimeSeriesDataContainer.from_dataframes(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            horizon=horizon,
            feature_columns=list(X_train_selected.columns),
        )

    def _train_single_model(
        self,
        container: TimeSeriesDataContainer,
        model_config: dict[str, Any],
        horizon: int,
    ) -> dict[str, Any]:
        """Train a single model."""
        model_name = model_config["name"]

        trainer_config = ModelFactory.create_trainer(model_config, horizon)

        if model_config.get("optimize_hyperparams", False):
            logger.info(f"  Hyperparameter optimization enabled for {model_name}")
            trainer_config = self._optimize_hyperparams(container, trainer_config, model_config)

        trainer = Trainer(trainer_config)
        training_results = trainer.run(container)

        logger.info(
            f"  {model_name} Val F1: {training_results['evaluation_metrics']['val_f1']:.4f}"
        )

        return {
            "trainer": trainer,
            "results": training_results,
            "config": trainer_config,
        }

    def _optimize_hyperparams(
        self,
        container: TimeSeriesDataContainer,
        trainer_config: Any,
        model_config: dict[str, Any],
    ) -> Any:
        """Run hyperparameter optimization with Optuna."""
        import pandas as pd

        logger.info("    Running Optuna hyperparameter optimization...")

        n_trials = model_config.get("optimization", {}).get("n_trials", 100)

        from src.validation.cv.cv_tuner import TimeSeriesOptunaTuner
        from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

        # Get training data using the new API
        X_train_result, y_train_result, _ = container.get_sklearn_arrays("train", return_df=True)

        # Ensure DataFrame/Series types for tuner
        X_train = (
            X_train_result
            if isinstance(X_train_result, pd.DataFrame)
            else pd.DataFrame(X_train_result)
        )
        y_train = (
            y_train_result
            if isinstance(y_train_result, pd.Series)
            else pd.Series(y_train_result)
        )

        # Create purged CV splitter
        cv_config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=10)
        cv = PurgedKFold(cv_config)

        tuner = TimeSeriesOptunaTuner(
            model_name=trainer_config.model_name,
            cv=cv,
            n_trials=n_trials,
        )

        result = tuner.tune(X=X_train, y=y_train)
        best_params = result.get("best_params", {})

        logger.info(f"    Best params: {best_params}")

        for param, value in best_params.items():
            setattr(trainer_config, param, value)

        return trainer_config

    def _build_ensemble(
        self,
        container: TimeSeriesDataContainer,
        model_results: dict[str, Any],
        horizon: int,
    ) -> dict[str, Any]:
        """Build ensemble from trained models."""
        if self.config is None:
            raise RuntimeError("config is None")

        ensemble_method = self.config["ensemble"]["method"]
        meta_learner = self.config["ensemble"].get("meta_learner", "ridge_meta")

        logger.info(f"  Ensemble method: {ensemble_method}, meta-learner: {meta_learner}")

        base_model_names = self.config["ensemble"].get("base_models", list(model_results.keys()))

        if ensemble_method == "stacking":
            from src.models.config import TrainerConfig

            from .trainer import Trainer

            # Pass ensemble-specific config via model_config dict
            ensemble_config = TrainerConfig(
                model_name="stacking",
                horizon=horizon,
                model_config={
                    "base_models": base_model_names,
                    "meta_learner": meta_learner,
                },
            )

            ensemble_trainer = Trainer(ensemble_config)
            ensemble_results = ensemble_trainer.run(container)

            logger.info(
                f"  Ensemble Val F1: {ensemble_results['evaluation_metrics']['val_f1']:.4f}"
            )

            return {
                "trainer": ensemble_trainer,
                "results": ensemble_results,
                "method": ensemble_method,
            }

        return {}

    def _save_results(self) -> None:
        """Save all results to disk."""
        logger.info(f"\nSaving results to: {self.output_dir}")

        results_path = self.output_dir / "results.json"

        serializable_results: dict[str, dict[str, Any]] = {}
        for key, value in self.results.items():
            serializable_results[key] = {}
            for model_name, model_data in value.items():
                if isinstance(model_data, dict) and "results" in model_data:
                    serializable_results[key][model_name] = {
                        "val_f1": model_data["results"]["evaluation_metrics"]["val_f1"],
                        "val_accuracy": model_data["results"]["evaluation_metrics"]["val_accuracy"],
                    }

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved: {results_path}")

        for model_key, trainer in self.trained_models.items():
            model_path = self.output_dir / f"{model_key}.pkl"
            # Trainer has model attribute, not direct save method
            if hasattr(trainer, "model") and hasattr(trainer.model, "save"):
                trainer.model.save(model_path)
                logger.info(f"Model saved: {model_path}")
            else:
                logger.warning(f"Could not save model for {model_key}")

    def display_results(self):
        """Display results summary (for notebook use)."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RESULTS: {self.experiment_name}")
        print(f"{'='*60}\n")

        for horizon_key, horizon_results in self.results.items():
            print(f"\n{horizon_key.upper()}:")
            for model_name, model_data in horizon_results.items():
                if isinstance(model_data, dict) and "results" in model_data:
                    val_f1 = model_data["results"]["evaluation_metrics"]["val_f1"]
                    val_acc = model_data["results"]["evaluation_metrics"]["val_accuracy"]
                    print(f"  {model_name:20s}: F1={val_f1:.4f}, Acc={val_acc:.4f}")

        print(f"\nResults saved to: {self.output_dir}")


# Alias for backward compatibility
LegacyTrainingOrchestrator = TrainingOrchestrator

__all__ = ["TrainingOrchestrator", "LegacyTrainingOrchestrator"]
