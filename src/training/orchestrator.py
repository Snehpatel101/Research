"""Training orchestrator - master controller for unified training system."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models import Trainer
from src.core.container import TimeSeriesDataContainer
from .config import ExperimentConfig, ModelConfig
from .config_loader import ConfigLoader
from .feature_selector import FeatureSelector
from .model_factory import ModelFactory

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Master controller for unified ML training system.

    Coordinates:
    - Data loading
    - Feature selection
    - Model training (single/multiple/ensemble)
    - Hyperparameter optimization
    - Feature optimization
    - Results aggregation

    Usage:
        config = load_config_from_params(
            symbol='MES',
            horizons=[20],
            models=['xgboost', 'lightgbm'],
            feature_mode='full',
            build_ensemble=True
        )

        orchestrator = TrainingOrchestrator(config)
        results = orchestrator.run()
    """

    def __init__(self, config: dict | ExperimentConfig):
        """
        Initialize orchestrator with configuration.

        Args:
            config: Validated configuration dict or ExperimentConfig
        """
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

        self.results = {}
        self.trained_models = {}
        self.feature_selector = FeatureSelector()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized TrainingOrchestrator: {self.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")

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

    def _run_with_experiment_config(self) -> dict:
        cfg = self.experiment_config

        for horizon in cfg.horizons:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING HORIZON: {horizon}")
            logger.info(f"{'='*60}")

            horizon_results = {}

            for model_cfg in cfg.models:
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
                    from src.features.optimization import optimize_features_for_model

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
        X_train_df, y_train, w_train = container.get_sklearn_arrays("train", return_df=True)

        # Use feature selector to filter
        X_train_filtered = self.feature_selector.select_features(X_train_df, mode=feature_mode)
        feature_list = list(X_train_filtered.columns)

        if len(feature_list) == len(X_train_df.columns):
            logger.info(f"  Feature mode '{feature_mode}': using all {len(feature_list)} features")
            return container

        logger.info(f"  Feature mode '{feature_mode}': {len(feature_list)} features (from {len(X_train_df.columns)})")

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
        from src.core.container import TimeSeriesDataContainer

        split_dfs = {}

        for split in container.available_splits:
            X_df, y, w = container.get_sklearn_arrays(split, return_df=True)

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
        data_dir = cfg.data_dir / timeframe

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Run pipeline with --process-all-timeframes or --output-timeframes {timeframe}"
            )

        logger.info(f"Loading data: {data_dir}, horizon={horizon}")
        container = TimeSeriesDataContainer.from_parquet_dir(data_dir, horizon=horizon)

        logger.info(
            f"Data loaded: Train={container.X_train.shape}, "
            f"Val={container.X_val.shape}, Test={container.X_test.shape}"
        )

        return container

    def _train_model_simple(
        self,
        container: TimeSeriesDataContainer,
        model_cfg: ModelConfig,
        horizon: int,
    ) -> dict:
        """
        Train single model with new ModelConfig interface.

        Args:
            container: Data container
            model_cfg: Model configuration
            horizon: Label horizon

        Returns:
            Dict with trainer, results, and config
        """
        from src.models import TrainerConfig

        trainer_config = TrainerConfig(
            model_name=model_cfg.name,
            horizon=horizon,
            sequence_length=model_cfg.sequence_length,
        )

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
        trainer_config,
        model_cfg: ModelConfig,
    ):
        """
        Run hyperparameter optimization for ModelConfig-based training.

        Args:
            container: Data container
            trainer_config: Base trainer config
            model_cfg: Model configuration with optimization settings

        Returns:
            Updated trainer config with best params
        """
        logger.info("    Running Optuna hyperparameter optimization...")

        from src.validation.cv.cv_tuner import TimeSeriesOptunaTuner

        tuner = TimeSeriesOptunaTuner(
            model_name=trainer_config.model_name,
            horizon=trainer_config.horizon,
            n_splits=5,
        )

        best_params = tuner.optimize(
            X=container.X_train,
            y=container.y_train,
            n_trials=model_cfg.hyperparam_opt_trials,
        )

        logger.info(f"    Best params: {best_params}")

        for param, value in best_params.items():
            setattr(trainer_config, param, value)

        return trainer_config

    def _build_ensemble_simple(
        self,
        container: TimeSeriesDataContainer,
        horizon_results: dict,
        horizon: int,
        cfg: ExperimentConfig,
    ) -> dict:
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
        from src.models import TrainerConfig

        ensemble_method = cfg.ensemble_method or "stacking"
        meta_learner = cfg.meta_learner or "ridge_meta"

        logger.info(f"  Ensemble method: {ensemble_method}, meta-learner: {meta_learner}")

        base_model_names = list(horizon_results.keys())

        if ensemble_method == "stacking":
            ensemble_config = TrainerConfig(
                model_name="stacking",
                horizon=horizon,
                base_models=base_model_names,
                meta_learner=meta_learner,
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

    def _run_with_dict_config(self) -> dict:
        for horizon in self.config["data"]["horizons"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING HORIZON: {horizon}")
            logger.info(f"{'='*60}")

            container = self._load_data(horizon)
            container = self._apply_feature_selection(container)

            horizon_results = {}

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
        data_dir = self.config["data"]["data_dir"]
        logger.info(f"Loading data from: {data_dir}, horizon: {horizon}")

        container = TimeSeriesDataContainer.from_parquet_dir(data_dir, horizon=horizon)

        logger.info(
            f"Data loaded: Train={container.X_train.shape}, Val={container.X_val.shape}, Test={container.X_test.shape}"
        )

        return container

    def _apply_feature_selection(
        self, container: TimeSeriesDataContainer
    ) -> TimeSeriesDataContainer:
        """Apply feature selection based on config mode."""
        feature_mode = self.config["features"]["mode"]
        mtf_strategy = self.config["features"].get("mtf_strategy", "indicators")

        logger.info(f"Applying feature selection: mode={feature_mode}, mtf_strategy={mtf_strategy}")

        X_train_selected = self.feature_selector.select_features(
            container.X_train, mode=feature_mode, mtf_strategy=mtf_strategy
        )
        X_val_selected = self.feature_selector.select_features(
            container.X_val, mode=feature_mode, mtf_strategy=mtf_strategy
        )
        X_test_selected = self.feature_selector.select_features(
            container.X_test, mode=feature_mode, mtf_strategy=mtf_strategy
        )

        logger.info(
            f"Features selected: {len(X_train_selected.columns)} (from {len(container.X_train.columns)})"
        )

        return TimeSeriesDataContainer(
            X_train=X_train_selected,
            y_train=container.y_train,
            X_val=X_val_selected,
            y_val=container.y_val,
            X_test=X_test_selected,
            y_test=container.y_test,
            sample_weights=container.sample_weights,
        )

    def _train_single_model(
        self,
        container: TimeSeriesDataContainer,
        model_config: dict,
        horizon: int,
    ) -> dict:
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

    def _optimize_hyperparams(self, container, trainer_config, model_config):
        """Run hyperparameter optimization with Optuna."""
        logger.info("    Running Optuna hyperparameter optimization...")

        n_trials = model_config.get("optimization", {}).get("n_trials", 100)

        from src.validation.cv.cv_tuner import TimeSeriesOptunaTuner

        tuner = TimeSeriesOptunaTuner(
            model_name=trainer_config.model_name,
            horizon=trainer_config.horizon,
            n_splits=5,
        )

        best_params = tuner.optimize(
            X=container.X_train,
            y=container.y_train,
            n_trials=n_trials,
        )

        logger.info(f"    Best params: {best_params}")

        for param, value in best_params.items():
            setattr(trainer_config, param, value)

        return trainer_config

    def _build_ensemble(self, container, model_results, horizon):
        """Build ensemble from trained models."""
        ensemble_method = self.config["ensemble"]["method"]
        meta_learner = self.config["ensemble"].get("meta_learner", "ridge_meta")

        logger.info(f"  Ensemble method: {ensemble_method}, meta-learner: {meta_learner}")

        base_model_names = self.config["ensemble"].get("base_models", list(model_results.keys()))

        if ensemble_method == "stacking":
            from src.models import Trainer, TrainerConfig

            ensemble_config = TrainerConfig(
                model_name="stacking",
                horizon=horizon,
                base_models=base_model_names,
                meta_learner=meta_learner,
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

    def _save_results(self):
        """Save all results to disk."""
        logger.info(f"\nSaving results to: {self.output_dir}")

        results_path = self.output_dir / "results.json"

        serializable_results = {}
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
            trainer.save(model_path)
            logger.info(f"Model saved: {model_path}")

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


__all__ = ["TrainingOrchestrator"]
