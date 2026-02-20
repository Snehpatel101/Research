"""
Training operations mixin for UnifiedTrainingOrchestrator.

Extracted from unified_orchestrator.py (M2 refactor).
Contains: single-model training, parallel boosting, OOM recovery,
model calibration, OOF generation, walk-forward, regime-aware,
and meta-labeling training modes.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.data.adapters import PreparedData
from src.validation.cv import OOFPrediction

from .services import ModelTrainingRequest, OOFRequest

if TYPE_CHECKING:
    from src.core import PipelineConfig

logger = logging.getLogger(__name__)


class TrainingOpsMixin:
    """Mixin providing training operation methods for the orchestrator."""

    config: PipelineConfig

    def _train_standard(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Standard training: PurgedKFold CV with OOF generation.

        Boosting models run in parallel; neural/transformer models run sequentially.
        """
        from src.core.contracts import get_model_contract

        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")

            boosting_models = []
            sequential_models = []
            for model_name in self.config.models:
                contract = get_model_contract(model_name)
                if contract.model_family == "boosting":
                    boosting_models.append(model_name)
                else:
                    sequential_models.append(model_name)

            if len(boosting_models) >= 2:
                logger.info(
                    f"\nTraining {len(boosting_models)} boosting models in parallel: "
                    f"{boosting_models}"
                )
                self._train_boosting_parallel(df, horizon, boosting_models, additional_dfs)
            else:
                sequential_models = boosting_models + sequential_models

            for model_name in sequential_models:
                self._train_model_sequential(df, model_name, horizon, additional_dfs)

            if len(self._prepared_cache) > 20:
                logger.warning(
                    f"PreparedData cache has {len(self._prepared_cache)} entries — "
                    "consider clearing if memory is constrained"
                )

    def _train_boosting_parallel(
        self,
        df: pd.DataFrame,
        horizon: int,
        boosting_models: list[str],
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Train boosting models in parallel using ParallelTrainingService."""
        from .unified_orchestrator import ModelTrainingResult

        prepared_map: dict[str, PreparedData] = {}
        training_requests: list[ModelTrainingRequest] = []

        for model_name in boosting_models:
            prepared = self._prepare_with_cache(df, model_name, additional_dfs)
            prepared_map[model_name] = prepared
            logger.info(f"  {model_name} data prepared: {prepared.summary()}")

            training_requests.append(ModelTrainingRequest(
                model_name=model_name, horizon=horizon, prepared_data=prepared,
                sequence_length=self.config.sequence_length,
                output_dir=self.output_dir / f"h{horizon}",
                optimize_hyperparams=self.config.optimize_hyperparams,
                n_splits=self.config.n_splits,
                hyperparam_trials=self.config.hyperparam_trials,
                scoring=self.config.optuna_metric,
                use_feature_selection=self.config.optimize_features,
            ))

        parallel_results = self._parallel_service.train_models_parallel(training_requests)

        for model_name, service_result in zip(boosting_models, parallel_results, strict=True):
            prepared = prepared_map[model_name]
            if self.config.auto_calibrate:
                self._calibrate_model(service_result, prepared, model_name)

            self._trained_models[f"{model_name}_h{horizon}"] = service_result.trainer
            result = ModelTrainingResult(
                model_name=service_result.model_name, horizon=service_result.horizon,
                metrics=service_result.metrics, trainer=service_result.trainer,
                training_time_seconds=service_result.training_time_seconds,
                n_features=service_result.n_features, data_rank=service_result.data_rank,
            )
            self._log_feature_importance(model_name, service_result.trainer)

            key = f"{model_name}_h{horizon}"
            self._model_results[key] = result
            if self.config.save_oof:
                oof = self._generate_oof(model_name, prepared, horizon)
                if oof is not None:
                    self._oof_predictions[key] = oof
                    result.oof_prediction = oof

            logger.info(
                f"  {model_name} complete: val_f1={result.metrics.get('val_f1', 0):.4f}, "
                f"time={result.training_time_seconds:.1f}s"
            )

        del prepared_map
        gc.collect()

    def _train_model_sequential(
        self,
        df: pd.DataFrame,
        model_name: str,
        horizon: int,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Train a single model sequentially with data preparation and OOF."""
        logger.info(f"\nTraining {model_name}...")
        prepared = self._prepare_with_cache(df, model_name, additional_dfs)
        logger.info(f"  Data prepared: {prepared.summary()}")

        result = self._train_single_model(model_name, prepared, horizon)
        self._log_feature_importance(model_name, result.trainer)

        key = f"{model_name}_h{horizon}"
        self._model_results[key] = result
        if self.config.save_oof:
            oof = self._generate_oof(model_name, prepared, horizon)
            if oof is not None:
                self._oof_predictions[key] = oof
                result.oof_prediction = oof

        logger.info(
            f"  {model_name} complete: val_f1={result.metrics.get('val_f1', 0):.4f}, "
            f"time={result.training_time_seconds:.1f}s"
        )

    def _train_single_model(self, model_name: str, prepared: PreparedData, horizon: int) -> Any:
        """Train a single model with OOM recovery for neural/transformer models."""
        from src.core.contracts import get_model_contract
        from .unified_orchestrator import ModelTrainingResult

        request = ModelTrainingRequest(
            model_name=model_name, horizon=horizon, prepared_data=prepared,
            sequence_length=self.config.sequence_length,
            output_dir=self.output_dir / f"h{horizon}",
            optimize_hyperparams=self.config.optimize_hyperparams,
            n_splits=self.config.n_splits,
            hyperparam_trials=self.config.hyperparam_trials,
            scoring=self.config.optuna_metric,
            use_feature_selection=self.config.optimize_features,
        )

        training_degraded = False
        contract = get_model_contract(model_name)
        is_neural = contract.model_family in ("neural", "transformer")

        try:
            result = self._model_service.train_model(request)
        except (RuntimeError, MemoryError) as exc:
            if not is_neural or (
                "out of memory" not in str(exc).lower() and not isinstance(exc, MemoryError)
            ):
                raise
            original_batch = getattr(self.config, "batch_size", None) or 64
            reduced_batch = max(original_batch // 2, 1)
            logger.warning(
                f"OOM during {model_name} training — reducing batch size "
                f"from {original_batch} to {reduced_batch} and retrying"
            )
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if hasattr(self.config, "batch_size"):
                self.config.batch_size = reduced_batch
            result = self._model_service.train_model(request)
            training_degraded = True

        if self.config.auto_calibrate:
            self._calibrate_model(result, prepared, model_name)

        self._trained_models[f"{model_name}_h{horizon}"] = result.trainer
        calibrator = getattr(result, "calibrator", None)
        return ModelTrainingResult(
            model_name=result.model_name, horizon=result.horizon,
            metrics=result.metrics, trainer=result.trainer,
            training_time_seconds=result.training_time_seconds,
            n_features=result.n_features, data_rank=result.data_rank,
            calibrator=calibrator, training_degraded=training_degraded,
        )

    def _log_feature_importance(self, model_name: str, trainer: Any) -> None:
        """Log top 10 feature importances after training."""
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

    def _calibrate_model(self, result: Any, prepared: PreparedData, model_name: str) -> None:
        """Calibrate model probabilities using validation set (Phase 4F)."""
        try:
            from src.models.calibration import CalibrationConfig, ProbabilityCalibrator

            logger.info(f"  Calibrating {model_name} probabilities...")
            if not hasattr(result.trainer, "predict_proba"):
                logger.warning(f"    {model_name} doesn't support predict_proba, skipping")
                return
            if prepared.X_val is None or prepared.y_val is None:
                logger.warning("    No validation data available, skipping calibration")
                return
            if len(prepared.y_val) < self.config.calibration_min_samples:
                logger.warning(
                    f"    Insufficient validation samples for calibration "
                    f"({len(prepared.y_val)} < {self.config.calibration_min_samples})"
                )
                return

            val_probas = result.trainer.predict_proba(prepared.X_val)
            if val_probas.ndim == 2 and val_probas.shape[1] > 1:
                val_probas = val_probas[:, 1]

            method = self.config.calibration_method
            if method not in ("isotonic", "sigmoid", "auto"):
                method = "auto"
            calib_config = CalibrationConfig(
                method=method,  # type: ignore[arg-type]
                min_samples_per_class=self.config.calibration_min_samples,
            )
            calibrator = ProbabilityCalibrator(calib_config)
            metrics = calibrator.fit(prepared.y_val, val_probas)
            logger.info(
                f"    Brier improvement: {metrics.brier_improvement:.1%}, "
                f"ECE improvement: {metrics.ece_improvement:.1%}"
            )
            if not hasattr(result, "calibrator"):
                result.calibrator = calibrator
                result.calibration_metrics = metrics
        except ImportError:
            logger.warning("    Calibration module not available, skipping")
        except Exception as e:
            logger.warning(f"    Calibration failed: {e}")

    def _generate_oof(
        self, model_name: str, prepared: PreparedData, horizon: int,
    ) -> OOFPrediction | None:
        """Generate OOF predictions via OOFGenerationService."""
        request = OOFRequest(
            model_name=model_name, horizon=horizon, prepared_data=prepared,
            n_splits=self.config.n_splits, purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
        )
        return self._oof_service.generate_oof(request)

    def _train_walk_forward(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Walk-forward training: expanding/rolling windows for realistic backtesting."""
        from src.core.container import TimeSeriesDataContainer
        from src.models.registry import ModelRegistry
        from .config import ExperimentConfig, ModelConfig
        from .modes import WalkForwardTrainer, WalkForwardTrainerConfig
        from .unified_orchestrator import ModelTrainingResult

        logger.info("Walk-forward training mode")
        horizon = self.config.horizons[0]

        model_configs = [ModelConfig(name=m) for m in self.config.models]
        exp_config = ExperimentConfig(
            symbol=self.config.symbol, horizons=self.config.horizons,
            models=model_configs, data_dir=Path(self.config.data_path).parent,
            output_dir=self.output_dir,
        )
        wf_config = WalkForwardTrainerConfig(
            n_windows=getattr(self.config, "wf_n_windows", self.config.n_splits),
            window_type=getattr(self.config, "wf_window_type", "expanding"),
            min_train_pct=getattr(self.config, "wf_min_train_pct", self.config.train_ratio),
            test_pct=getattr(self.config, "wf_test_pct", self.config.val_ratio),
            gap_bars=self.config.purge_bars, embargo_bars=self.config.embargo_bars,
        )
        trainer = WalkForwardTrainer(exp_config, wf_config)

        prepared = self._data_preparer.prepare(
            df=df, model_name=self.config.models[0], additional_dfs=additional_dfs,
        )

        # Reconstruct FULL dataset (walk-forward does its own splitting)
        arrays_X = [prepared.X_train]
        arrays_y = [prepared.y_train]
        if prepared.X_val is not None:
            arrays_X.append(prepared.X_val)
            arrays_y.append(prepared.y_val)
        if prepared.X_test is not None:
            arrays_X.append(prepared.X_test)
            arrays_y.append(prepared.y_test)

        X_all = np.concatenate(arrays_X)
        y_all = np.concatenate(arrays_y)

        if prepared.data_rank == 2:
            feature_cols = prepared.feature_names
            X_flat = X_all
        else:
            n_flat = int(np.prod(X_all.shape[1:]))
            feature_cols = [f"f{i}" for i in range(n_flat)]
            X_flat = X_all.reshape(X_all.shape[0], -1)

        n_all = len(X_flat)
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= n_all:
            idx = df.index[:n_all]
        else:
            idx = pd.RangeIndex(n_all)

        X_all_df = pd.DataFrame(X_flat, columns=feature_cols, index=idx)
        full_df = X_all_df.copy()
        full_df[f"label_h{horizon}"] = y_all
        full_df[f"sample_weight_h{horizon}"] = np.ones(n_all)

        container = TimeSeriesDataContainer.from_dataframes(
            train_df=full_df, val_df=None, test_df=None,
            horizon=horizon, feature_columns=list(X_all_df.columns),
        )
        # Store original data shape metadata for 4D model reconstruction
        if prepared.data_rank > 2:
            original_shape = X_all.shape[1:]  # (n_timeframes, seq_len, n_features) for 4D
            container.metadata["original_nd_shape"] = original_shape
            container.metadata["data_rank"] = prepared.data_rank
            container.metadata["n_timeframes"] = prepared.n_timeframes
            container.metadata["sequence_length"] = prepared.sequence_length
            logger.info(f"  Stored original shape metadata: {original_shape} (rank={prepared.data_rank})")
        logger.info(
            f"  Walk-forward data: {n_all} samples "
            f"(train={len(prepared.X_train)}, val={prepared.n_val}, test={prepared.n_test})"
        )

        results = trainer.run(container)
        class_names = ["short", "neutral", "long"]

        for model_name, wf_result in results.get("model_results", {}).items():
            key = f"{model_name}_h{horizon}"
            pred_df = wf_result.predictions_df
            pred_col, conf_col = f"{model_name}_pred", f"{model_name}_confidence"
            oof = None

            if pred_col in pred_df.columns:
                valid_mask = ~np.isnan(pred_df[pred_col].values)
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) > 0:
                    preds = pred_df[pred_col].values[valid_mask].astype(int)
                    confidence = pred_df[conf_col].values[valid_mask]
                    y_true_oof = pred_df["y_true"].values[valid_mask].astype(int)

                    prob_cols_wf = [
                        c for c in pred_df.columns if c.startswith(f"{model_name}_prob_class")
                    ]
                    oof_data: dict[str, Any] = {
                        f"{model_name}_pred": preds,
                        f"{model_name}_confidence": confidence,
                        "y_true": y_true_oof,
                    }
                    for i, col_wf in enumerate(prob_cols_wf):
                        oof_col = (
                            f"{model_name}_prob_{class_names[i]}" if i < len(class_names)
                            else f"{model_name}_prob_class{i}"
                        )
                        oof_data[oof_col] = pred_df[col_wf].values[valid_mask]

                    fold_info = []
                    for wr in wf_result.window_results:
                        for wm in wr.window_metrics:
                            fold_info.append({
                                "fold": wm.window, "train_size": wm.train_size,
                                "test_size": wm.test_size, "accuracy": wm.accuracy,
                                "f1": wm.f1, "training_time": wm.training_time,
                            })

                    oof = OOFPrediction(
                        model_name=model_name, predictions=pd.DataFrame(oof_data),
                        fold_info=fold_info, coverage=len(valid_indices) / n_all,
                        original_indices=valid_indices, n_total_samples=n_all,
                    )
                    self._oof_predictions[key] = oof
                    logger.info(
                        f"  {model_name}: OOF coverage={oof.coverage:.1%} "
                        f"({len(valid_indices)}/{n_all} samples)"
                    )
                else:
                    logger.warning(f"  {model_name}: 0 valid predictions in walk-forward results")
            else:
                logger.warning(
                    f"  {model_name}: prediction column '{pred_col}' not found in WF results"
                )

            last_model = None
            if wf_result.model_paths:
                last_path = wf_result.model_paths[-1]
                if last_path.exists():
                    last_model = ModelRegistry.create(model_name)
                    last_model.load(last_path)
                    logger.info(f"  {model_name}: loaded last-window model from {last_path}")

            if last_model is not None:
                self._trained_models[key] = last_model
            else:
                logger.warning(f"  {model_name}: no model available for bundling/deploy")

            self._model_results[key] = ModelTrainingResult(
                model_name=model_name, horizon=horizon,
                metrics={
                    "val_f1": wf_result.aggregated_metrics.get("mean_f1", 0),
                    "val_accuracy": wf_result.aggregated_metrics.get("mean_accuracy", 0),
                },
                trainer=last_model, oof_prediction=oof,
                training_time_seconds=wf_result.total_time,
                n_features=prepared.n_features, data_rank=prepared.data_rank,
            )

        del prepared
        gc.collect()

    def _train_regime_aware(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Regime-aware training: separate models for different market regimes."""
        from .regime_trainer import RegimeAwareTrainer
        from .unified_orchestrator import ModelTrainingResult

        logger.info("Regime-aware training mode")
        logger.info(f"  Detection method: {self.config.regime_detection_method}")
        logger.info(f"  Number of regimes: {self.config.n_regimes}")
        logger.info(f"  Separate models: {self.config.train_separate_regime_models}")

        regime_trainer = RegimeAwareTrainer(self.config)
        self._regime_trainer = regime_trainer

        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")
            for model_name in self.config.models:
                logger.info(f"\nTraining regime-aware: {model_name}...")
                prepared = self._data_preparer.prepare(
                    df=df, model_name=model_name, additional_dfs=additional_dfs,
                )
                logger.info(f"  Data prepared: {prepared.summary()}")

                regime_result = regime_trainer.train(
                    prepared=prepared, horizon=horizon,
                    model_name=model_name, save_models=self.config.save_models,
                )

                for (trained_model_name, regime), regime_model_result in (
                    regime_result.regime_results.items()
                ):
                    if trained_model_name != model_name:
                        continue
                    key = f"{model_name}_h{horizon}_{regime}"
                    self._model_results[key] = ModelTrainingResult(
                        model_name=f"{model_name}_{regime}", horizon=horizon,
                        metrics={
                            "val_f1": regime_model_result.val_f1,
                            "val_accuracy": regime_model_result.val_accuracy,
                            **regime_model_result.metrics,
                        },
                        trainer=regime_model_result.trainer,
                        training_time_seconds=regime_model_result.training_time_seconds,
                        n_features=prepared.n_features, data_rank=prepared.data_rank,
                    )
                    self._trained_models[key] = regime_model_result.trainer
                    logger.info(
                        f"    {model_name}/{regime}: val_f1={regime_model_result.val_f1:.4f}, "
                        f"samples={regime_model_result.n_samples}"
                    )

                aggregated_key = f"{model_name}_h{horizon}_aggregated"
                aggregated_metrics = {
                    k.replace(f"{model_name}_", ""): v
                    for k, v in regime_result.aggregated_metrics.items()
                    if model_name in k or "overall" in k
                }
                self._model_results[aggregated_key] = ModelTrainingResult(
                    model_name=f"{model_name}_regime_aware", horizon=horizon,
                    metrics=aggregated_metrics,
                    training_time_seconds=regime_result.total_time_seconds,
                    n_features=prepared.n_features, data_rank=prepared.data_rank,
                )
                del prepared
                gc.collect()

        logger.info("\nRegime-aware training complete")

    def _train_meta_labeling(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Meta-labeling training (Lopez de Prado 2018): primary model + meta-model."""
        logger.info("=" * 60)
        logger.info("META-LABELING TRAINING (Lopez de Prado 2018)")
        logger.info("=" * 60)
        logger.info(f"Primary model: {self.config.meta_labeling_primary_model}")
        logger.info(f"Meta model: {self.config.meta_labeling_meta_model}")
        logger.info(f"Bet threshold: {self.config.meta_labeling_threshold}")

        start_time = time.time()
        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")
            result = self._train_meta_labeling_for_horizon(df, horizon, additional_dfs)
            key = f"meta_labeling_h{horizon}"
            self._model_results[key] = result
            logger.info(
                f"  Meta-labeling complete: "
                f"primary_acc={result.metrics.get('primary_accuracy', 0):.4f}, "
                f"combined_acc={result.metrics.get('combined_accuracy', 0):.4f}, "
                f"trade_fraction={result.metrics.get('trade_fraction', 0)*100:.1f}%"
            )
            gc.collect()

        logger.info(f"\nMeta-labeling total time: {time.time() - start_time:.1f}s")

    def _train_meta_labeling_for_horizon(
        self, df: pd.DataFrame, horizon: int,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> Any:
        """Train meta-labeling system for a single horizon."""
        from .unified_orchestrator import ModelTrainingResult

        start_time = time.time()
        primary_model_name = self.config.meta_labeling_primary_model
        meta_model_name = self.config.meta_labeling_meta_model

        # Stage 1: Prepare data
        logger.info("\n  STAGE 1: Preparing data...")
        prepared = self._data_preparer.prepare(
            df=df, model_name=primary_model_name, additional_dfs=additional_dfs,
        )
        logger.info(f"    Data: {prepared.n_train} train, {prepared.n_val} val samples")

        X_train = self._flatten_to_2d(prepared.X_train)
        X_val = self._flatten_to_2d(prepared.X_val)
        y_train, y_val = prepared.y_train, prepared.y_val
        feature_names = (
            prepared.feature_names if prepared.data_rank == 2
            else [f"f{i}" for i in range(X_train.shape[1])]
        )

        # Stage 2: Train primary model (direction)
        logger.info("\n  STAGE 2: Training primary model (direction)...")
        primary_result = self._train_single_model(primary_model_name, prepared, horizon)
        primary_trainer = primary_result.trainer
        if primary_trainer is None:
            raise RuntimeError("Primary model training failed")

        primary_train_classes = primary_trainer.model.predict(X_train).class_predictions
        primary_val_classes = primary_trainer.model.predict(X_val).class_predictions
        primary_train_acc = (primary_train_classes == y_train).mean()
        primary_val_acc = (primary_val_classes == y_val).mean()
        logger.info(f"    Primary train/val accuracy: {primary_train_acc:.4f}/{primary_val_acc:.4f}")

        # Stage 3: Create meta-labels (1=correct, 0=wrong)
        logger.info("\n  STAGE 3: Creating meta-labels...")
        meta_labels_train = (primary_train_classes == y_train).astype(int)
        meta_labels_val = (primary_val_classes == y_val).astype(int)
        logger.info(
            f"    Train: {meta_labels_train.mean()*100:.1f}% correct, "
            f"Val: {meta_labels_val.mean()*100:.1f}% correct"
        )

        # Stage 4: Train meta-model (bet sizing)
        logger.info("\n  STAGE 4: Training meta-model (bet sizing)...")
        meta_model = self._create_meta_model(meta_model_name)
        meta_model.fit(X_train, meta_labels_train)

        if hasattr(meta_model, "predict_proba"):
            meta_proba_train = meta_model.predict_proba(X_train)[:, 1]
            meta_proba_val = meta_model.predict_proba(X_val)[:, 1]
        else:
            meta_proba_train = meta_model.predict(X_train).astype(float)
            meta_proba_val = meta_model.predict(X_val).astype(float)

        meta_train_acc = ((meta_proba_train >= 0.5) == meta_labels_train).mean()
        meta_val_acc = ((meta_proba_val >= 0.5) == meta_labels_val).mean()
        logger.info(f"    Meta train/val accuracy: {meta_train_acc:.4f}/{meta_val_acc:.4f}")

        # Stage 5: Evaluate combined system
        logger.info("\n  STAGE 5: Evaluating combined system...")
        threshold = self.config.meta_labeling_threshold
        trades_taken_val = meta_proba_val >= threshold

        if trades_taken_val.sum() > 0:
            combined_val_acc = (
                primary_val_classes[trades_taken_val] == y_val[trades_taken_val]
            ).mean()
            trade_fraction = trades_taken_val.mean()
        else:
            combined_val_acc, trade_fraction = 0.0, 0.0

        improvement = combined_val_acc - primary_val_acc if trade_fraction > 0 else 0.0
        logger.info(
            f"    Threshold={threshold}, trades={trade_fraction*100:.1f}%, "
            f"primary={primary_val_acc:.4f}, combined={combined_val_acc:.4f}, "
            f"improvement={improvement:+.4f}"
        )

        # Stage 6: Store models
        logger.info("\n  STAGE 6: Storing models...")
        model_key = f"meta_labeling_h{horizon}"
        self._trained_models[f"{model_key}_primary"] = primary_trainer
        self._trained_models[f"{model_key}_meta"] = meta_model

        if self.config.save_models:
            import pickle
            models_dir = self.output_dir / "models"
            models_dir.mkdir(exist_ok=True)
            meta_path = models_dir / f"{model_key}_meta.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump({
                    "meta_model": meta_model, "primary_model_name": primary_model_name,
                    "meta_model_name": meta_model_name, "threshold": threshold,
                    "feature_names": feature_names,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"    Saved meta-model to: {meta_path}")

        metrics = {
            "primary_train_accuracy": float(primary_train_acc),
            "primary_val_accuracy": float(primary_val_acc),
            "primary_val_f1": primary_result.metrics.get("val_f1", 0),
            "meta_train_accuracy": float(meta_train_acc),
            "meta_val_accuracy": float(meta_val_acc),
            "combined_accuracy": float(combined_val_acc),
            "primary_accuracy": float(primary_val_acc),
            "trade_fraction": float(trade_fraction),
            "trades_taken": int(trades_taken_val.sum()),
            "total_samples": len(y_val),
            "threshold": threshold,
            "improvement": float(improvement),
            "val_f1": float(combined_val_acc),
            "val_accuracy": float(combined_val_acc),
        }

        return ModelTrainingResult(
            model_name=f"meta_labeling_{primary_model_name}_{meta_model_name}",
            horizon=horizon, metrics=metrics, trainer=primary_trainer,
            training_time_seconds=time.time() - start_time,
            n_features=prepared.n_features, data_rank=prepared.data_rank,
        )

    def _flatten_to_2d(self, X: np.ndarray) -> np.ndarray:
        """Flatten array to 2D if needed."""
        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)

    def _create_meta_model(self, model_name: str) -> Any:
        """Create meta-model for bet sizing (logistic, random_forest, xgboost, lightgbm, catboost)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        rs = self.config.random_state
        nj = self.config.n_jobs

        if model_name == "logistic":
            return LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=rs)
        elif model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100, max_depth=5, class_weight="balanced", random_state=rs, n_jobs=nj,
            )
        elif model_name == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=rs, n_jobs=nj, use_label_encoder=False, eval_metric="logloss",
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to logistic")
                return self._create_meta_model("logistic")
        elif model_name == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=rs, n_jobs=nj, verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available, falling back to logistic")
                return self._create_meta_model("logistic")
        elif model_name == "catboost":
            try:
                import catboost as cb  # type: ignore[import-not-found]
                return cb.CatBoostClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=rs, verbose=False,
                )
            except ImportError:
                logger.warning("CatBoost not available, falling back to logistic")
                return self._create_meta_model("logistic")
        else:
            logger.warning(f"Unknown meta model: {model_name}, using logistic")
            return self._create_meta_model("logistic")
