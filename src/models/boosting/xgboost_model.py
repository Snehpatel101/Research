"""
XGBoost Model - GPU-accelerated gradient boosting for 3-class prediction.

Supports GPU training (tree_method='hist', device='cuda'), early stopping,
sample weights, and feature importance extraction.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..registry import register

logger = logging.getLogger(__name__)

# Module-level cache for CUDA availability check
_CUDA_AVAILABLE: bool | None = None


def _check_cuda_available() -> bool:
    """Check if XGBoost CUDA is available (cached).

    Uses a lightweight check that creates a Booster without training,
    avoiding the overhead of running actual training on every instantiation.
    """
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE

    try:
        # Check PyTorch CUDA first (fast check)
        try:
            import torch

            if not torch.cuda.is_available():
                logger.debug("PyTorch CUDA not available, skipping XGBoost GPU check")
                _CUDA_AVAILABLE = False
                return False
        except ImportError:
            # PyTorch not installed, proceed with XGBoost check
            pass

        # Verify XGBoost can use CUDA by creating a Booster (no training)
        test_data = xgb.DMatrix(np.array([[1.0]]), label=np.array([0]))
        params = {"device": "cuda", "tree_method": "hist"}
        # Create booster without training - much lighter than xgb.train()
        bst = xgb.Booster(params, [test_data])
        del bst
        _CUDA_AVAILABLE = True
        logger.debug("XGBoost CUDA check passed")
    except Exception as e:
        logger.debug(f"XGBoost CUDA not available: {e}")
        _CUDA_AVAILABLE = False

    return _CUDA_AVAILABLE


@register(
    name="xgboost",
    family="boosting",
    description="XGBoost gradient boosting with GPU support",
    aliases=["xgb"],
)
class XGBoostModel(BaseModel):
    """XGBoost gradient boosting classifier with GPU support."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: xgb.Booster | None = None
        self._feature_names: list[str] | None = None
        self._n_classes: int = 3
        self._use_gpu: bool = self._config.get("use_gpu", False)

        if self._use_gpu:
            if _check_cuda_available():
                logger.info("XGBoost GPU training enabled")
            else:
                logger.warning("CUDA not available, falling back to CPU")
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
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "learning_rate": 0.05,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 50,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "use_gpu": False,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 1,
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
        """Train XGBoost model with early stopping."""
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        # Merge config
        train_config = self._config.copy()
        if config:
            train_config.update(config)

        # Convert labels: -1,0,1 -> 0,1,2
        y_train_xgb = self._convert_labels_to_xgb(y_train)
        y_val_xgb = self._convert_labels_to_xgb(y_val)

        # Apply class weights if enabled (balances short/neutral/long predictions)
        final_weights = sample_weights
        if train_config.get("use_class_weights", True):
            # Compute balanced class weights
            unique_classes, class_counts = np.unique(y_train_xgb, return_counts=True)
            n_samples = len(y_train_xgb)
            n_classes = len(unique_classes)
            class_weight_values = n_samples / (n_classes * class_counts)

            # Create mapping from class to weight (handles missing classes)
            class_weight_dict = dict(zip(unique_classes, class_weight_values, strict=False))

            # Map weights to samples using the dictionary
            sample_class_weights = np.array([class_weight_dict[int(c)] for c in y_train_xgb])

            # Combine with existing sample weights
            if sample_weights is not None:
                final_weights = sample_weights * sample_class_weights
            else:
                final_weights = sample_class_weights

            logger.debug(f"Class weights applied: {class_weight_dict}")

        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train_xgb, weight=final_weights)
        dval = xgb.DMatrix(X_val, label=y_val_xgb)

        # Build parameters and train
        params = self._build_params(train_config)
        n_estimators = train_config.get("n_estimators", 500)
        early_stopping = train_config.get("early_stopping_rounds", 50)

        evals = [(dtrain, "train"), (dval, "val")]
        evals_result: dict[str, dict[str, list[float]]] = {}

        logger.info(
            f"Training XGBoost: n_estimators={n_estimators}, "
            f"max_depth={params['max_depth']}, gpu={'on' if self._use_gpu else 'off'}"
        )

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            evals_result=evals_result,
            verbose_eval=50 if train_config.get("verbosity", 1) > 0 else False,
        )

        training_time = time.time() - start_time

        # Extract metrics
        metric_name = train_config.get("eval_metric", "mlogloss")
        train_losses = evals_result.get("train", {}).get(metric_name, [])
        val_losses = evals_result.get("val", {}).get(metric_name, [])

        best_iteration = self._model.best_iteration
        epochs_trained = best_iteration + 1

        # Compute accuracy and F1
        train_metrics = self._compute_metrics(dtrain, y_train)
        val_metrics = self._compute_metrics(dval, y_val)

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
            best_epoch=best_iteration,
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

        dmatrix = xgb.DMatrix(X)
        probabilities = self._model.predict(dmatrix)
        class_predictions_xgb = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_xgb(class_predictions_xgb)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"model": "xgboost"},
        )

    def save(self, path: Path) -> None:
        """Save model (JSON) and metadata (pickle) to directory."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path / "model.json"))

        metadata = {
            "config": self._config,
            "feature_names": self._feature_names,
            "n_classes": self._n_classes,
            "use_gpu": self._use_gpu,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved XGBoost model to {path}")

    def load(self, path: Path) -> None:
        """Load model from directory."""
        path = Path(path)
        model_path = path / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = xgb.Booster()
        self._model.load_model(str(model_path))

        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            self._config = metadata.get("config", self._config)
            self._feature_names = metadata.get("feature_names")
            self._n_classes = metadata.get("n_classes", 3)
            self._use_gpu = metadata.get("use_gpu", False)

        self._is_fitted = True
        logger.info(f"Loaded XGBoost model from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importances by gain."""
        if not self._is_fitted:
            return None

        importance = self._model.get_score(importance_type="gain")

        if self._feature_names:
            result = {}
            for key, value in importance.items():
                if key.startswith("f"):
                    idx = int(key[1:])
                    if idx < len(self._feature_names):
                        result[self._feature_names[idx]] = value
                else:
                    result[key] = value
            return result

        return importance

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def _build_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """Build XGBoost parameter dict."""
        params = {
            "objective": "multi:softprob",
            "num_class": self._n_classes,
            "eval_metric": config.get("eval_metric", "mlogloss"),
            "max_depth": config.get("max_depth", 6),
            "min_child_weight": config.get("min_child_weight", 10),
            "subsample": config.get("subsample", 0.8),
            "colsample_bytree": config.get("colsample_bytree", 0.8),
            "eta": config.get("learning_rate", 0.05),
            "gamma": config.get("gamma", 0.1),
            "alpha": config.get("reg_alpha", 0.1),
            "lambda": config.get("reg_lambda", 1.0),
            "tree_method": "hist",
            "seed": config.get("random_state", 42),
            "verbosity": min(config.get("verbosity", 1), 1),
        }

        if self._use_gpu:
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"
            params["nthread"] = config.get("n_jobs", -1)

        return params

    def _convert_labels_to_xgb(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from -1,0,1 to 0,1,2."""
        return map_labels_to_classes(labels)

    def _convert_labels_from_xgb(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from 0,1,2 to -1,0,1."""
        return map_classes_to_labels(labels)

    def _compute_metrics(self, dmatrix: xgb.DMatrix, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        from sklearn.metrics import accuracy_score, f1_score

        probabilities = self._model.predict(dmatrix)
        y_pred_xgb = np.argmax(probabilities, axis=1)
        y_pred = self._convert_labels_from_xgb(y_pred_xgb)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = ["XGBoostModel"]
