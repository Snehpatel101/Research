"""
LightGBM Model - Leaf-wise gradient boosting for 3-class prediction.

Supports GPU training (device='cuda'), early stopping, sample weights,
and feature importance extraction. Uses leaf-wise tree growth for
faster training on large datasets.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..registry import register

logger = logging.getLogger(__name__)

# Module-level cache for CUDA availability check
_LIGHTGBM_CUDA_AVAILABLE: bool | None = None


def _check_cuda_available() -> bool:
    """Check if LightGBM GPU is available (cached).

    Verifies both PyTorch CUDA availability and that LightGBM was built
    with GPU support. Standard pip LightGBM doesn't include GPU support.
    """
    global _LIGHTGBM_CUDA_AVAILABLE
    if _LIGHTGBM_CUDA_AVAILABLE is not None:
        return _LIGHTGBM_CUDA_AVAILABLE

    if not LIGHTGBM_AVAILABLE:
        _LIGHTGBM_CUDA_AVAILABLE = False
        return False

    try:
        # Check PyTorch CUDA first (fast check)
        try:
            import torch
            if not torch.cuda.is_available():
                logger.debug("PyTorch CUDA not available, skipping LightGBM GPU check")
                _LIGHTGBM_CUDA_AVAILABLE = False
                return False
        except ImportError:
            # PyTorch not installed, proceed with LightGBM check
            pass

        # Try to create a small dataset with GPU device
        # This will fail if LightGBM wasn't built with GPU support
        params = {"device": "cuda", "verbose": -1}
        data = lgb.Dataset(np.array([[1.0]]), label=np.array([0]))
        data.construct()

        # Attempt to create a booster with GPU params to verify GPU support
        # LightGBM will raise an error if GPU support isn't compiled in
        params_full = {
            "device": "cuda",
            "objective": "binary",
            "verbose": -1,
            "num_iterations": 0,
        }
        try:
            bst = lgb.Booster(params=params_full, train_set=data)
            del bst
        except lgb.basic.LightGBMError as e:
            # LightGBM raises specific error if GPU not supported
            if "GPU" in str(e) or "cuda" in str(e).lower():
                logger.debug(f"LightGBM GPU not supported in this build: {e}")
                _LIGHTGBM_CUDA_AVAILABLE = False
                return False
            raise

        _LIGHTGBM_CUDA_AVAILABLE = True
        logger.debug("LightGBM CUDA check passed")
    except Exception as e:
        logger.debug(f"LightGBM GPU not available: {e}")
        _LIGHTGBM_CUDA_AVAILABLE = False

    return _LIGHTGBM_CUDA_AVAILABLE


@register(
    name="lightgbm",
    family="boosting",
    description="LightGBM gradient boosting with GPU support",
    aliases=["lgbm"],
)
class LightGBMModel(BaseModel):
    """LightGBM gradient boosting classifier with GPU support."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        super().__init__(config)
        self._model: lgb.Booster | None = None
        self._feature_names: list[str] | None = None
        self._n_classes: int = 3
        self._use_gpu: bool = self._config.get("use_gpu", False)

        if self._use_gpu:
            if _check_cuda_available():
                logger.info("LightGBM GPU training enabled")
            else:
                logger.warning("CUDA not available for LightGBM, falling back to CPU")
                self._use_gpu = False

    @property
    def model_family(self) -> str:
        return "boosting"

    @property
    def requires_scaling(self) -> bool:
        return False

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict[str, Any]:
        return {
            "n_estimators": 500,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "learning_rate": 0.05,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 50,
            "boosting_type": "gbdt",
            "use_gpu": False,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingMetrics:
        """Train LightGBM model with early stopping."""
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        # Merge config
        train_config = self._config.copy()
        if config:
            train_config.update(config)

        # Convert labels: -1,0,1 -> 0,1,2
        y_train_lgb = self._convert_labels_to_lgb(y_train)
        y_val_lgb = self._convert_labels_to_lgb(y_val)

        # Apply class weights if enabled (balances short/neutral/long predictions)
        final_weights = sample_weights
        if train_config.get("use_class_weights", True):
            # Compute balanced class weights
            unique_classes, class_counts = np.unique(y_train_lgb, return_counts=True)
            n_samples = len(y_train_lgb)
            n_classes = len(unique_classes)
            class_weight_values = n_samples / (n_classes * class_counts)

            # Create mapping from class to weight (handles missing classes)
            class_weight_dict = dict(zip(unique_classes, class_weight_values, strict=False))

            # Map weights to samples using the dictionary
            sample_class_weights = np.array([class_weight_dict[int(c)] for c in y_train_lgb])

            # Combine with existing sample weights
            if sample_weights is not None:
                final_weights = sample_weights * sample_class_weights
            else:
                final_weights = sample_class_weights

            logger.debug(f"Class weights applied: {class_weight_dict}")

        # Create Dataset objects
        # Note: free_raw_data=True (default) frees raw data after construction
        # to reduce memory usage. We only need the internal LightGBM format.
        dtrain = lgb.Dataset(
            X_train,
            label=y_train_lgb,
            weight=final_weights,
            free_raw_data=True,  # Free raw data to reduce memory by ~50%
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val_lgb,
            reference=dtrain,
            free_raw_data=True,  # Free raw data to reduce memory
        )

        # Build parameters
        params = self._build_params(train_config)
        n_estimators = train_config.get("n_estimators", 500)
        early_stopping = train_config.get("early_stopping_rounds", 50)

        logger.info(
            f"Training LightGBM: n_estimators={n_estimators}, "
            f"max_depth={params.get('max_depth', -1)}, "
            f"num_leaves={params.get('num_leaves', 31)}, "
            f"gpu={'on' if self._use_gpu else 'off'}"
        )

        evals_result: dict[str, dict[str, list[float]]] = {}

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=50 if train_config.get("verbosity", -1) >= 0 else 0),
            lgb.record_evaluation(evals_result),
        ]

        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # Free LightGBM Dataset objects to release memory
        del dtrain, dval

        training_time = time.time() - start_time

        # Extract metrics
        metric_name = "multi_logloss"
        train_losses = evals_result.get("train", {}).get(metric_name, [])
        val_losses = evals_result.get("val", {}).get(metric_name, [])

        best_iteration = self._model.best_iteration
        epochs_trained = best_iteration if best_iteration > 0 else len(train_losses)

        # Compute accuracy and F1
        train_metrics = self._compute_metrics(X_train, y_train)
        val_metrics = self._compute_metrics(X_val, y_val)

        self._is_fitted = True

        logger.info(
            f"Training complete: epochs={epochs_trained}, val_f1={val_metrics['f1']:.4f}, "
            f"time={training_time:.1f}s"
        )

        return TrainingMetrics(
            train_loss=train_losses[-1] if train_losses else 0.0,
            val_loss=val_losses[-1] if val_losses else 0.0,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            train_f1=train_metrics["f1"],
            val_f1=val_metrics["f1"],
            epochs_trained=epochs_trained,
            training_time_seconds=training_time,
            early_stopped=epochs_trained < n_estimators,
            best_epoch=best_iteration if best_iteration > 0 else None,
            history={f"train_{metric_name}": train_losses, f"val_{metric_name}": val_losses},
            metadata={
                "best_iteration": best_iteration,
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "use_gpu": self._use_gpu,
            },
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        probabilities = self._model.predict(X, num_iteration=self._model.best_iteration)
        class_predictions_lgb = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_lgb(class_predictions_lgb)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"model": "lightgbm"},
        )

    def save(self, path: Path) -> None:
        """Save model (text) and metadata (pickle) to directory."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path / "model.txt"))

        metadata = {
            "config": self._config,
            "feature_names": self._feature_names,
            "n_classes": self._n_classes,
            "use_gpu": self._use_gpu,
            "best_iteration": self._model.best_iteration,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved LightGBM model to {path}")

    def load(self, path: Path) -> None:
        """Load model from directory."""
        path = Path(path)
        model_path = path / "model.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = lgb.Booster(model_file=str(model_path))

        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            self._config = metadata.get("config", self._config)
            self._feature_names = metadata.get("feature_names")
            self._n_classes = metadata.get("n_classes", 3)
            self._use_gpu = metadata.get("use_gpu", False)

        self._is_fitted = True
        logger.info(f"Loaded LightGBM model from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importances by gain."""
        if not self._is_fitted:
            return None

        importance = self._model.feature_importance(importance_type="gain")
        feature_names = self._feature_names or [
            f"f{i}" for i in range(len(importance))
        ]

        return dict(zip(feature_names, importance.tolist(), strict=False))

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def _build_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Build LightGBM parameter dict.

        Enforces constraint: num_leaves <= 2^max_depth
        """
        max_depth = config.get("max_depth", 6)
        num_leaves = config.get("num_leaves", 31)

        # Enforce LightGBM constraint: num_leaves <= 2^max_depth
        if max_depth > 0:
            max_valid_leaves = 2 ** max_depth
            if num_leaves > max_valid_leaves:
                logger.warning(
                    f"num_leaves ({num_leaves}) exceeds max for max_depth={max_depth} "
                    f"(max={max_valid_leaves}). Capping to {max_valid_leaves}."
                )
                num_leaves = max_valid_leaves

        params = {
            "objective": "multiclass",
            "num_class": self._n_classes,
            "metric": "multi_logloss",
            "boosting_type": config.get("boosting_type", "gbdt"),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": config.get("min_child_samples", 20),
            "subsample": config.get("subsample", 0.8),
            "colsample_bytree": config.get("colsample_bytree", 0.8),
            "learning_rate": config.get("learning_rate", 0.05),
            "reg_alpha": config.get("reg_alpha", 0.1),
            "reg_lambda": config.get("reg_lambda", 1.0),
            "seed": config.get("random_state", 42),
            "verbosity": config.get("verbosity", -1),
            "force_col_wise": True,
        }

        if self._use_gpu:
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"
            params["num_threads"] = config.get("n_jobs", -1)
            if params["num_threads"] == -1:
                params["num_threads"] = 0  # LightGBM uses 0 for all cores

        return params

    def _convert_labels_to_lgb(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from -1,0,1 to 0,1,2."""
        return map_labels_to_classes(labels)

    def _convert_labels_from_lgb(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from 0,1,2 to -1,0,1."""
        return map_classes_to_labels(labels)

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        from sklearn.metrics import accuracy_score, f1_score

        probabilities = self._model.predict(X, num_iteration=self._model.best_iteration)
        y_pred_lgb = np.argmax(probabilities, axis=1)
        y_pred = self._convert_labels_from_lgb(y_pred_lgb)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = ["LightGBMModel"]
