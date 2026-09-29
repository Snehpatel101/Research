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

from src.core.constants import OHLCV_COLUMNS
from src.data.adapters import PreparedData
from src.models.device import offload_model_to_cpu, release_gpu_memory
from src.validation.cv import OOFPrediction
from src.validation.cv.oof_core import merge_oof_predictions, reindex_oof_to_rows

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

            if len(boosting_models) >= 2 and getattr(self.config, "parallel_training", False):
                logger.info(
                    f"\nTraining {len(boosting_models)} boosting models in parallel: "
                    f"{boosting_models}"
                )
                self._train_boosting_parallel(df, horizon, boosting_models, additional_dfs)
            else:
                sequential_models = boosting_models + sequential_models

            for model_name in sequential_models:
                self._train_model_sequential(df, model_name, horizon, additional_dfs)

            # Auto-evict oldest PreparedData cache entries when cache grows too large.
            # Each entry can be several GB for 3D/4D models, so keeping stale
            # entries risks OOM on memory-constrained machines.
            max_cache_size = 10
            while len(self._prepared_cache) > max_cache_size:
                evicted_key = next(iter(self._prepared_cache))
                del self._prepared_cache[evicted_key]
                gc.collect()
            if len(self._prepared_cache) > max_cache_size // 2:
                logger.debug(
                    f"PreparedData cache at {len(self._prepared_cache)} entries "
                    f"(max {max_cache_size})"
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

            training_requests.append(
                ModelTrainingRequest(
                    model_name=model_name,
                    horizon=horizon,
                    prepared_data=prepared,
                    sequence_length=self.config.sequence_length,
                    output_dir=self.output_dir / f"h{horizon}",
                    optimize_hyperparams=self.config.optimize_hyperparams,
                    n_splits=self.config.n_splits,
                    hyperparam_trials=self.config.hyperparam_trials,
                    scoring=self.config.optuna_metric,
                    use_feature_selection=self._trainer_feature_selection(model_name),
                    max_epochs=self.config.max_epochs,
                    batch_size=getattr(self.config, "batch_size", None),
                    cv_method=self.config.cv_method,
                    embargo_bars=getattr(self.config, "embargo_bars", None),
                    purge_bars=getattr(self.config, "purge_bars", None),
                    n_classes=getattr(self.config, "n_classes", 3),
                    early_stopping_patience=getattr(self.config, "early_stopping_patience", None),
                    optuna_timeout=getattr(self.config, "optuna_timeout", None),
                    use_calibration=self.config.auto_calibrate,
                    calibration_method=self.config.calibration_method,
                )
            )

        parallel_results = self._parallel_service.train_models_parallel(training_requests)

        for model_name, service_result in zip(boosting_models, parallel_results, strict=True):
            prepared = prepared_map[model_name]
            if self.config.auto_calibrate:
                self._calibrate_model(service_result, prepared, model_name)

            self._trained_models[f"{model_name}_h{horizon}"] = service_result.trainer
            result = ModelTrainingResult(
                model_name=service_result.model_name,
                horizon=service_result.horizon,
                metrics=service_result.metrics,
                trainer=service_result.trainer,
                training_time_seconds=service_result.training_time_seconds,
                n_features=service_result.n_features,
                data_rank=service_result.data_rank,
                calibrator=getattr(service_result, "calibrator", None),
            )
            self._log_feature_importance(model_name, service_result.trainer)

            key = f"{model_name}_h{horizon}"
            self._model_results[key] = result
            if self.config.save_oof:
                trained_config = getattr(service_result.trainer.model, "config", None)
                oof = self._generate_oof(model_name, prepared, horizon, model_config=trained_config)
                if oof is not None:
                    self._oof_predictions[key] = oof
                    result.oof_prediction = oof

            logger.info(
                f"  {model_name} complete: val_f1={result.metrics.get('val_f1', 0):.4f}, "
                f"time={result.training_time_seconds:.1f}s"
            )

        del prepared_map
        # Evict boosting PreparedData from cache (no longer needed after parallel training)
        from src.core.contracts import get_model_contract

        for model_name in boosting_models:
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
            self._prepared_cache.pop(cache_key, None)
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

        # Offload neural model to CPU BEFORE OOF generation to free GPU memory.
        # OOF needs to train N fold models on GPU; without this, large models
        # (TFT, Transformer, etc.) cause CUDA OOM during OOF cross-validation.
        offload_model_to_cpu(result.trainer)

        if self.config.save_oof:
            trained_config = getattr(result.trainer.model, "config", None)
            oof = self._generate_oof(model_name, prepared, horizon, model_config=trained_config)
            if oof is not None:
                self._oof_predictions[key] = oof
                result.oof_prediction = oof

        logger.info(
            f"  {model_name} complete: val_f1={result.metrics.get('val_f1', 0):.4f}, "
            f"time={result.training_time_seconds:.1f}s"
        )

        # Free PreparedData and evict from cache to prevent OOM
        # when training sequential models (TCN 3D ~60 GB, PatchTST 4D ~8 GB)
        del prepared
        # Reconstruct cache key from contract properties (same as _prepare_with_cache)
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
            del self._prepared_cache[cache_key]
        # GPU cleanup already done by _offload_trainer_to_cpu before OOF;
        # just collect Python garbage here for prepared data cache eviction.
        gc.collect()

    def _trainer_feature_selection(self, model_name: str) -> bool:
        """Whether Trainer should run its own feature selection for this model.

        The orchestrator already selects each model's features on train-only
        data (_run_feature_selection_on_train_data). Selecting again inside
        Trainer would train the final model on a different feature set than
        its OOF (stacking) models, so it only runs when the orchestrator
        selected nothing (walk-forward mode, which selects per window). Models
        without a per-model set (e.g. a meta-labeling primary outside
        ``models``) then use every prepared column.
        """
        return self.config.optimize_features and not self._per_model_features

    def _train_single_model(self, model_name: str, prepared: PreparedData, horizon: int) -> Any:
        """Train a single model with OOM recovery for neural/transformer models."""
        from src.core.contracts import get_model_contract

        from .unified_orchestrator import ModelTrainingResult

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
            use_feature_selection=self._trainer_feature_selection(model_name),
            max_epochs=self.config.max_epochs,
            batch_size=getattr(self.config, "batch_size", None),
            cv_method=self.config.cv_method,
            embargo_bars=getattr(self.config, "embargo_bars", None),
            purge_bars=getattr(self.config, "purge_bars", None),
            n_classes=getattr(self.config, "n_classes", 3),
            early_stopping_patience=getattr(self.config, "early_stopping_patience", None),
            optuna_timeout=getattr(self.config, "optuna_timeout", None),
            use_calibration=self.config.auto_calibrate,
            calibration_method=self.config.calibration_method,
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
            release_gpu_memory()
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
                use_feature_selection=self._trainer_feature_selection(model_name),
                max_epochs=self.config.max_epochs,
                cv_method=self.config.cv_method,
                batch_size=reduced_batch,
                embargo_bars=getattr(self.config, "embargo_bars", None),
                purge_bars=getattr(self.config, "purge_bars", None),
                n_classes=getattr(self.config, "n_classes", 3),
                early_stopping_patience=getattr(self.config, "early_stopping_patience", None),
                optuna_timeout=getattr(self.config, "optuna_timeout", None),
                use_calibration=self.config.auto_calibrate,
                calibration_method=self.config.calibration_method,
            )
            result = self._model_service.train_model(request)
            training_degraded = True

        if self.config.auto_calibrate:
            self._calibrate_model(result, prepared, model_name)

        self._trained_models[f"{model_name}_h{horizon}"] = result.trainer
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
            training_degraded=training_degraded,
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
            # Trainer.run_prepared already fits a calibrator on the validation
            # split when use_calibration is on — reuse it instead of refitting.
            trainer_calibrator = getattr(result.trainer, "calibrator", None)
            if trainer_calibrator is not None:
                result.calibrator = trainer_calibrator
                return
            predict_proba = self._resolve_predict_proba(result.trainer)
            if predict_proba is None:
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

            # ProbabilityCalibrator.fit requires the FULL (n, n_classes)
            # matrix. The old [:, 1] collapse handed it a 1D array, fit raised
            # ValueError, and the broad except below silently dropped
            # calibration for every multiclass model.
            val_probas = np.asarray(predict_proba(prepared.X_val))
            if val_probas.ndim == 1:
                # 1D output is the positive-class probability — reconstruct
                # the (n, 2) matrix; an (n, 1) reshape would make the
                # calibrator treat this as a 1-class problem.
                val_probas = np.column_stack([1.0 - val_probas, val_probas])

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
            # Always store the fitted calibrator — both result flavors accept
            # attribute assignment (the canonical dataclass has the field; the
            # service dataclass takes a dynamic attribute). The previous
            # `if not hasattr(...)` guard silently dropped the calibrator on
            # any result that already had a calibrator field.
            result.calibrator = calibrator
            result.calibration_metrics = metrics
        except ImportError:
            logger.warning("    Calibration module not available, skipping")
        except Exception as e:
            logger.warning(f"    Calibration failed: {e}")

    @staticmethod
    def _resolve_predict_proba(trainer: Any) -> Any | None:
        """Return a callable X -> (n, n_classes) probabilities, or None.

        Real Trainers expose probabilities through their BaseModel
        (``trainer.model.predict(X).class_probabilities``); simple
        estimator-like trainers expose ``predict_proba`` directly.
        """
        if hasattr(trainer, "predict_proba"):
            return trainer.predict_proba
        model = getattr(trainer, "model", None)
        if model is not None and hasattr(model, "predict"):
            return lambda X: model.predict(X).class_probabilities
        return None

    def _generate_oof(
        self,
        model_name: str,
        prepared: PreparedData,
        horizon: int,
        model_config: dict[str, Any] | None = None,
    ) -> OOFPrediction | None:
        """Generate OOF predictions via OOFGenerationService."""
        # Filter -99 sentinel labels BEFORE OOF generation to prevent
        # fold models from training on invalid labels (Phase 85 audit fix)
        prepared = prepared.filter_invalid_labels()
        request = OOFRequest(
            model_name=model_name,
            horizon=horizon,
            prepared_data=prepared,
            n_splits=self.config.n_splits,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
            model_config=model_config,
            n_classes=getattr(self.config, "n_classes", 3),
        )
        return self._oof_service.generate_oof(request)

    def _train_walk_forward(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Walk-forward training: expanding/rolling windows for realistic backtesting."""
        from src.core.container import TimeSeriesDataContainer

        from .config import _ModeConfig
        from .modes import WalkForwardTrainer, WalkForwardTrainerConfig
        from .unified_orchestrator import ModelTrainingResult

        logger.info("Walk-forward training mode")
        horizon = self.config.horizons[0]

        exp_config = _ModeConfig(
            symbol=self.config.symbol,
            horizons=self.config.horizons,
            models=list(self.config.models),
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

        # Prepare data per-model so each model gets data matching its own
        # contract (rank, sequence length, feature mode, etc.).
        class_names = ["short", "neutral", "long"]

        for model_name in self.config.models:
            # Walk-forward windows select features on their own training data
            # (B08 fix), so they see every feature column.
            float64_cols = df.select_dtypes(include=["float64"]).columns
            df_model = (
                df.astype(dict.fromkeys(float64_cols, np.float32)) if len(float64_cols) else df
            )
            prepared = self._data_preparer.prepare(
                df=df_model,
                model_name=model_name,
                additional_dfs=additional_dfs,
            ).filter_invalid_labels()

            # Walk-forward is the evaluation protocol (honest OOS predictions for
            # the backtest and stacking). Windows re-select features and re-fit
            # scalers, so the deployable model is trained exactly like standard
            # mode: train-split per-model features, same prepared-data path.
            deploy_result = self._train_single_model(
                model_name,
                self._prepare_with_cache(df, model_name, additional_dfs),
                horizon,
            )
            self._clear_prepared_cache()

            # Row of the source DataFrame for every walk-forward sample
            # (train, val, test concatenated in that order below)
            index_parts = [prepared.train_indices, prepared.val_indices]
            if prepared.X_test is not None:
                index_parts.append(prepared.test_indices)
            row_positions = (
                np.concatenate(index_parts) if all(p is not None for p in index_parts) else None
            )

            # Save metadata before freeing prepared data
            _n_features = prepared.n_features
            _data_rank = prepared.data_rank
            _n_timeframes = getattr(prepared, "n_timeframes", None)
            _sequence_length = getattr(prepared, "sequence_length", None)
            # Capture original ND shape now, before prepared is freed below
            original_shape = prepared.X_train.shape[1:] if _data_rank > 2 else None

            # Reconstruct FULL dataset as flat float32 (walk-forward does its own splitting).
            # Flatten each split individually and free 3D arrays eagerly to avoid
            # holding 3D + 2D + DataFrame copies simultaneously (~240GB peak → ~60GB).
            flat_parts = []
            label_parts = []

            for arr_x, arr_y in [
                (prepared.X_train, prepared.y_train),
                (prepared.X_val, prepared.y_val) if prepared.X_val is not None else (None, None),
                (prepared.X_test, prepared.y_test) if prepared.X_test is not None else (None, None),
            ]:
                if arr_x is None:
                    continue
                if _data_rank == 2:
                    flat_parts.append(arr_x.astype(np.float32, copy=False))
                else:
                    # Flatten 3D/4D → 2D and convert to float32 in one step
                    flat_parts.append(
                        arr_x.reshape(arr_x.shape[0], -1).astype(np.float32, copy=False)
                    )
                label_parts.append(arr_y)

            # Resolve feature column names and free prepared BEFORE concatenation
            # to avoid holding both prepared's arrays and flat_parts simultaneously
            if _data_rank == 2:
                feature_cols = prepared.feature_names
            else:
                n_flat = flat_parts[0].shape[1]
                feature_cols = [f"f{i}" for i in range(n_flat)]

            del prepared
            gc.collect()

            X_flat = np.concatenate(flat_parts)
            y_all = np.concatenate(label_parts)
            del flat_parts, label_parts

            n_all = len(X_flat)
            # Always use RangeIndex — after filter_invalid_labels(), n_all samples
            # are non-contiguous so df.index[:n_all] would be misaligned
            idx = pd.RangeIndex(n_all)

            # Build DataFrame for container — add labels directly, no .copy()
            X_all_df = pd.DataFrame(X_flat, columns=feature_cols, index=idx)
            feature_col_list = list(X_all_df.columns)
            X_all_df[f"label_h{horizon}"] = y_all
            X_all_df[f"sample_weight_h{horizon}"] = np.ones(n_all)

            # Log after building DataFrame
            logger.info(
                f"  Walk-forward data for {model_name}: {n_all} samples "
                f"(rank={_data_rank}, features={_n_features})"
            )

            # Free flat arrays before container creation
            del X_flat, y_all
            gc.collect()

            container = TimeSeriesDataContainer.from_dataframes(
                train_df=X_all_df,
                val_df=None,
                test_df=None,
                horizon=horizon,
                feature_columns=feature_col_list,
            )
            del X_all_df  # Container owns the data now
            gc.collect()

            # Store original data shape metadata for 4D model reconstruction
            if _data_rank > 2 and original_shape is not None:
                container.metadata["original_nd_shape"] = original_shape
                container.metadata["data_rank"] = _data_rank
                if _n_timeframes is not None:
                    container.metadata["n_timeframes"] = _n_timeframes
                if _sequence_length is not None:
                    container.metadata["sequence_length"] = _sequence_length
                logger.info(
                    f"  Stored original shape metadata: {original_shape} " f"(rank={_data_rank})"
                )

            # Create a single-model config for the walk-forward trainer
            single_model_config = _ModeConfig(
                symbol=exp_config.symbol,
                horizons=exp_config.horizons,
                models=[model_name],
                data_dir=exp_config.data_dir,
                output_dir=exp_config.output_dir,
            )
            single_trainer = WalkForwardTrainer(single_model_config, wf_config)
            single_trainer._pipeline_config = self.config

            results = single_trainer.run(container)
            del container, single_trainer  # Free after extracting results
            gc.collect()

            for result_model_name, wf_result in results.get("model_results", {}).items():
                key = f"{result_model_name}_h{horizon}"
                pred_df = wf_result.predictions_df
                pred_col = f"{result_model_name}_pred"
                conf_col = f"{result_model_name}_confidence"
                oof = None

                if pred_col in pred_df.columns:
                    valid_mask = ~np.isnan(pred_df[pred_col].values)
                    valid_indices = np.where(valid_mask)[0]

                    if len(valid_indices) > 0:
                        # OOFPrediction schema contract (matches the tabular and
                        # sequence producers): the predictions DataFrame is
                        # FULL-LENGTH (n_total_samples rows, NaN where no
                        # prediction) and original_indices marks the valid rows.
                        # Consumers index it positionally with original_indices,
                        # so a compact frame would raise IndexError downstream.
                        preds = pred_df[pred_col].values.astype(float)
                        confidence = pred_df[conf_col].values.astype(float)
                        y_true_oof = pred_df["y_true"].values.astype(float)

                        prob_cols_wf = [
                            c
                            for c in pred_df.columns
                            if c.startswith(f"{result_model_name}_prob_class")
                        ]
                        oof_data: dict[str, Any] = {
                            f"{result_model_name}_pred": preds,
                            f"{result_model_name}_confidence": confidence,
                            "y_true": y_true_oof,
                        }
                        for i, col_wf in enumerate(prob_cols_wf):
                            oof_col = (
                                f"{result_model_name}_prob_{class_names[i]}"
                                if i < len(class_names)
                                else f"{result_model_name}_prob_class{i}"
                            )
                            oof_data[oof_col] = pred_df[col_wf].values

                        fold_info = []
                        for wr in wf_result.window_results:
                            for wm in wr.window_metrics:
                                fold_info.append(
                                    {
                                        "fold": wm.window,
                                        "train_size": wm.train_size,
                                        "test_size": wm.test_size,
                                        "accuracy": wm.accuracy,
                                        "f1": wm.f1,
                                        "training_time": wm.training_time,
                                    }
                                )

                        oof = reindex_oof_to_rows(
                            OOFPrediction(
                                model_name=result_model_name,
                                predictions=pd.DataFrame(oof_data),
                                fold_info=fold_info,
                                coverage=len(valid_indices) / n_all,
                                original_indices=valid_indices,
                                n_total_samples=n_all,
                            ),
                            row_positions,
                        )
                        self._oof_predictions[key] = oof
                        logger.info(
                            f"  {result_model_name}: OOF coverage={oof.coverage:.1%} "
                            f"({len(valid_indices)}/{n_all} samples)"
                        )
                    else:
                        logger.warning(
                            f"  {result_model_name}: 0 valid predictions in "
                            f"walk-forward results"
                        )
                else:
                    logger.warning(
                        f"  {result_model_name}: prediction column '{pred_col}' "
                        f"not found in WF results"
                    )

                self._model_results[key] = ModelTrainingResult(
                    model_name=result_model_name,
                    horizon=horizon,
                    metrics={
                        "val_f1": wf_result.aggregated_metrics.get("mean_f1", 0),
                        "val_accuracy": wf_result.aggregated_metrics.get("mean_accuracy", 0),
                    },
                    trainer=deploy_result.trainer,
                    calibrator=getattr(deploy_result, "calibrator", None),
                    oof_prediction=oof,
                    training_time_seconds=wf_result.total_time
                    + deploy_result.training_time_seconds,
                    n_features=_n_features,
                    data_rank=_data_rank,
                )

    def _train_regime_aware(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Regime-aware training: separate models for different market regimes."""
        from .regime_trainer import RegimeAwareTrainer

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
                # Filter DataFrame to per-model feature subset before preparation
                df_model = df
                if self._per_model_features and model_name in self._per_model_features:
                    model_features = set(self._per_model_features[model_name])
                    all_features = set(self._all_feature_names)
                    drop_cols = [
                        c for c in df.columns if c in all_features and c not in model_features
                    ]
                    if drop_cols:
                        df_model = df.drop(columns=drop_cols)
                        logger.debug(f"Filtered to {len(model_features)} features for {model_name}")

                # Downcast float64 → float32 to halve memory during preparation
                float64_cols = df_model.select_dtypes(include=["float64"]).columns
                if len(float64_cols) > 0:
                    df_model = df_model.astype(dict.fromkeys(float64_cols, np.float32))

                prepared = self._data_preparer.prepare(
                    df=df_model,
                    model_name=model_name,
                    additional_dfs=additional_dfs,
                )
                logger.info(f"  Data prepared: {prepared.summary()}")

                regime_result = regime_trainer.train(
                    prepared=prepared,
                    horizon=horizon,
                    model_name=model_name,
                    save_models=self.config.save_models,
                    # Regime detection needs OHLCV, which per-model feature
                    # selection may have dropped from df_model
                    raw_ohlcv=df[[c for c in OHLCV_COLUMNS if c in df.columns]],
                )

                self._record_regime_model(model_name, horizon, prepared, regime_result)
                del prepared
                # Move neural models to CPU and reset torch state
                for (_rname, _regime), rr in regime_result.regime_results.items():
                    offload_model_to_cpu(rr.trainer)
                release_gpu_memory()

        logger.info("\nRegime-aware training complete")

    def _record_regime_model(
        self,
        model_name: str,
        horizon: int,
        prepared: PreparedData,
        regime_result: Any,
    ) -> None:
        """Record one regime-routed model: per-regime trainers + regime-routed OOF.

        The OOF for regime r comes from CV on regime r's training samples
        (mapped to source rows), so each bar's OOF prediction is made by a
        model of its own regime that never saw it — the same routing the
        deployed RegimeBundle applies. Stacking and the backtest consume it
        like any other model's OOF.
        """
        from dataclasses import asdict, replace

        from .unified_orchestrator import ModelTrainingResult

        members = {
            regime: rr
            for (name, regime), rr in regime_result.regime_results.items()
            if name == model_name and rr.trainer is not None
        }
        if not members:
            logger.warning(f"  {model_name}: no regime had enough samples; nothing trained")
            return

        train_regimes = regime_result.train_regimes.to_numpy()
        regime_oofs = []
        for regime in members:
            mask = train_regimes == regime
            subset = replace(
                prepared,
                X_train=prepared.X_train[mask],
                y_train=prepared.y_train[mask],
                train_weights=(
                    prepared.train_weights[mask] if prepared.train_weights is not None else None
                ),
                train_indices=(
                    prepared.train_indices[mask] if prepared.train_indices is not None else None
                ),
            )
            oof = self._generate_oof(model_name, subset, horizon)
            if oof is not None:
                regime_oofs.append(oof)
        oof = merge_oof_predictions(regime_oofs) if regime_oofs else None

        n_total = sum(rr.n_samples for rr in members.values())
        metrics = {
            "val_f1": sum(rr.val_f1 * rr.n_samples for rr in members.values()) / n_total,
            "val_accuracy": sum(rr.val_accuracy * rr.n_samples for rr in members.values())
            / n_total,
            **{f"{regime}_val_f1": rr.val_f1 for regime, rr in members.items()},
        }
        key = f"{model_name}_h{horizon}"
        if oof is not None:
            self._oof_predictions[key] = oof
        for regime, rr in members.items():
            self._trained_models[f"{key}_{regime}"] = rr.trainer
            logger.info(
                f"    {model_name}/{regime}: val_f1={rr.val_f1:.4f}, samples={rr.n_samples}"
            )
        self._model_results[key] = ModelTrainingResult(
            model_name=model_name,
            horizon=horizon,
            metrics=metrics,
            oof_prediction=oof,
            training_time_seconds=regime_result.total_time_seconds,
            n_features=prepared.n_features,
            data_rank=prepared.data_rank,
            mode_artifacts={
                "kind": "regime",
                "regime_trainers": {regime: rr.trainer for regime, rr in members.items()},
                "detector_config": asdict(regime_result.detector.config),
                # Fallback for bars whose regime has no model: the best-covered regime
                "default_regime": max(members, key=lambda r: members[r].n_samples),
            },
        )

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
                f"primary_acc={result.metrics.get('primary_val_accuracy', 0):.4f}, "
                f"combined_acc={result.metrics.get('combined_accuracy', 0):.4f}, "
                f"trade_fraction={result.metrics.get('trade_fraction', 0)*100:.1f}%"
            )
            gc.collect()

        logger.info(f"\nMeta-labeling total time: {time.time() - start_time:.1f}s")

    def _train_meta_labeling_for_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> Any:
        """Train a meta-labeling system (primary direction + meta bet filter).

        1. Primary model trains on its prepared data (any input rank).
        2. Primary OOF predictions come from the standard OOF path, so the
           meta-labels "primary was right" are never in-sample.
        3. The meta-model learns P(primary correct) from the primary's own
           model input; its cross-validated probabilities filter the primary
           OOF into the system OOF used by the backtest.
        """
        from src.validation.cv import PurgedKFold, PurgedKFoldConfig

        from .unified_orchestrator import ModelTrainingResult

        start_time = time.time()
        primary_model_name = self.config.meta_labeling_primary_model
        meta_model_name = self.config.meta_labeling_meta_model
        threshold = self.config.meta_labeling_threshold

        # Stage 1: Prepare primary data (per-model feature subset when selected)
        logger.info("\n  STAGE 1: Preparing data...")
        df_model = df
        if self._per_model_features and primary_model_name in self._per_model_features:
            model_features = set(self._per_model_features[primary_model_name])
            all_features = set(self._all_feature_names)
            drop_cols = [c for c in df.columns if c in all_features and c not in model_features]
            df_model = df.drop(columns=drop_cols)
        float64_cols = df_model.select_dtypes(include=["float64"]).columns
        if len(float64_cols) > 0:
            df_model = df_model.astype(dict.fromkeys(float64_cols, np.float32))
        prepared = self._data_preparer.prepare(
            df=df_model, model_name=primary_model_name, additional_dfs=additional_dfs
        ).filter_invalid_labels()
        logger.info(f"    Data: {prepared.n_train} train, {prepared.n_val} val samples")

        # Stage 2: Primary model + its OOF predictions (at source rows)
        logger.info("\n  STAGE 2: Training primary model (direction)...")
        primary_result = self._train_single_model(primary_model_name, prepared, horizon)
        primary_trainer = primary_result.trainer
        primary_oof = self._generate_oof(primary_model_name, prepared, horizon)
        if primary_oof is None:
            raise RuntimeError(f"Could not generate OOF predictions for {primary_model_name}")

        rows = prepared.train_indices
        oof_classes = primary_oof.get_class_predictions()[rows]
        covered = ~np.isnan(oof_classes)
        primary_train_classes = oof_classes[covered].astype(np.int64)
        y_train = prepared.y_train[covered]

        # Meta-model input = exactly what the primary model consumes
        X_meta_train = self._primary_model_input(primary_trainer, prepared, prepared.X_train)
        X_meta_train = X_meta_train[covered]
        X_meta_val = self._primary_model_input(primary_trainer, prepared, prepared.X_val)
        primary_val_classes = primary_trainer.model.predict(X_meta_val).class_predictions
        y_val = prepared.y_val
        X_meta_train = X_meta_train.reshape(len(X_meta_train), -1)
        X_meta_val = X_meta_val.reshape(len(X_meta_val), -1)

        # Stage 3: Meta-labels (1 = primary correct), all out-of-sample
        meta_labels_train = (primary_train_classes == y_train).astype(int)
        meta_labels_val = (primary_val_classes == y_val).astype(int)
        logger.info(
            f"\n  STAGE 3: Meta-labels: train {meta_labels_train.mean():.1%} correct, "
            f"val {meta_labels_val.mean():.1%} correct"
        )

        # Stage 4: Meta-model, plus cross-validated P(correct) on train rows
        logger.info("\n  STAGE 4: Training meta-model (bet filter)...")
        cv = PurgedKFold(
            PurgedKFoldConfig(
                n_splits=self.config.n_splits,
                purge_bars=self.config.purge_bars,
                embargo_bars=min(self.config.embargo_bars, int(len(y_train) * 0.1)),
            )
        )
        meta_proba_oof = np.full(len(y_train), np.nan)
        index_frame = pd.DataFrame(index=range(len(y_train)))
        for tr_idx, va_idx in cv.split(index_frame, pd.Series(meta_labels_train)):
            if len(np.unique(meta_labels_train[tr_idx])) < 2:
                continue  # a fold with one class cannot fit a classifier
            fold_meta = self._create_meta_model(meta_model_name)
            fold_meta.fit(X_meta_train[tr_idx], meta_labels_train[tr_idx])
            meta_proba_oof[va_idx] = fold_meta.predict_proba(X_meta_train[va_idx])[:, 1]

        meta_model = self._create_meta_model(meta_model_name)
        meta_model.fit(X_meta_train, meta_labels_train)
        meta_proba_val = meta_model.predict_proba(X_meta_val)[:, 1]
        meta_val_acc = float(((meta_proba_val >= 0.5) == meta_labels_val).mean())

        # Stage 5: Evaluate the combined system on validation
        trades_val = meta_proba_val >= threshold
        trade_fraction = float(trades_val.mean())
        primary_val_acc = float((primary_val_classes == y_val).mean())
        combined_val_acc = (
            float((primary_val_classes[trades_val] == y_val[trades_val]).mean())
            if trades_val.any()
            else 0.0
        )
        logger.info(
            f"\n  STAGE 5: threshold={threshold}, trades={trade_fraction:.1%}, "
            f"primary={primary_val_acc:.4f}, combined={combined_val_acc:.4f}"
        )

        # System OOF: primary OOF, neutral where the meta filter rejects
        system_frame = primary_oof.predictions.copy()
        pred_col = f"{primary_model_name}_pred"
        rejected = np.zeros(len(system_frame), dtype=bool)
        filter_rows = rows[covered]
        rejected[filter_rows] = ~(meta_proba_oof >= threshold)
        system_frame.loc[rejected, pred_col] = 0.0
        system_oof = OOFPrediction(
            model_name=primary_model_name,
            predictions=system_frame,
            fold_info=primary_oof.fold_info,
            coverage=primary_oof.coverage,
            original_indices=primary_oof.original_indices,
            sequence_length=primary_oof.sequence_length,
            n_total_samples=primary_oof.n_total_samples,
        )

        metrics = {
            "primary_val_accuracy": primary_val_acc,
            "primary_val_f1": primary_result.metrics.get("val_f1", 0),
            "meta_val_accuracy": meta_val_acc,
            "combined_accuracy": combined_val_acc,
            "trade_fraction": trade_fraction,
            "trades_taken": int(trades_val.sum()),
            "total_samples": len(y_val),
            "threshold": threshold,
            "improvement": combined_val_acc - primary_val_acc if trades_val.any() else 0.0,
            "val_f1": combined_val_acc,
            "val_accuracy": combined_val_acc,
        }
        model_key = f"meta_labeling_h{horizon}"
        self._trained_models[f"{model_key}_primary"] = primary_trainer
        self._trained_models[f"{model_key}_meta"] = meta_model

        return ModelTrainingResult(
            model_name=f"meta_labeling_{primary_model_name}_{meta_model_name}",
            horizon=horizon,
            metrics=metrics,
            trainer=primary_trainer,
            oof_prediction=system_oof,
            training_time_seconds=time.time() - start_time,
            n_features=prepared.n_features,
            data_rank=prepared.data_rank,
            calibrator=primary_result.calibrator,
            mode_artifacts={
                "kind": "meta_labeling",
                "primary_model": primary_model_name,
                "meta_model": meta_model,
                "meta_model_name": meta_model_name,
                "threshold": threshold,
            },
        )

    @staticmethod
    def _primary_model_input(trainer: Any, prepared: PreparedData, X: np.ndarray) -> np.ndarray:
        """Restrict prepared features to the columns the trained model consumes."""
        columns = list(getattr(trainer, "feature_columns", None) or [])
        if prepared.data_rank != 2 or not columns or columns == list(prepared.feature_names):
            return X
        positions = [prepared.feature_names.index(c) for c in columns]
        return X[:, positions]

    def _create_meta_model(self, model_name: str) -> Any:
        """Create meta-model for bet sizing (logistic, random_forest, xgboost, lightgbm, catboost)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        rs = self.config.random_state
        nj = self.config.n_jobs

        if model_name == "logistic":
            return LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=rs
            )
        elif model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight="balanced",
                random_state=rs,
                n_jobs=nj,
            )
        elif model_name == "xgboost":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=rs,
                    n_jobs=nj,
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
                    random_state=rs,
                    n_jobs=nj,
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
                    random_state=rs,
                    verbose=False,
                )
            except ImportError:
                logger.warning("CatBoost not available, falling back to logistic")
                return self._create_meta_model("logistic")
        else:
            logger.warning(f"Unknown meta model: {model_name}, using logistic")
            return self._create_meta_model("logistic")
