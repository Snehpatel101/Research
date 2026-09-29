"""
Soft-voting meta-learner for stacking ensembles.

Averages the base models' class probabilities instead of learning a
combination. It is the no-fit baseline of the stacking family: it cannot
overfit the OOF predictions and works for any mix of base models (2D, 3D, 4D)
because it only sees their aligned probabilities.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.core.utils.safe_pickle import safe_pickle_dump, safe_pickle_load

from ..base import BaseModel, PredictionResult, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..registry import register

logger = logging.getLogger(__name__)

# OOFAligner.stacking_features appends mean_confidence, agreement, entropy
N_DERIVED_STACKING_FEATURES = 3


@register(
    name="voting_meta",
    family="meta_learner",
    description="Soft-voting meta-learner (mean of base-model probabilities)",
    aliases=["soft_voting_meta", "vote_meta"],
)
class VotingMetaLearner(BaseModel):
    """
    Soft-voting meta-learner: prediction = mean of base-model probabilities.

    Input: stacking features from OOFAligner — ``n_models * n_classes``
    probability columns followed by 3 derived columns (ignored here).

    Example:
        meta = VotingMetaLearner(config={"n_classes": 3})
        meta.fit(stacking_train, y_train, stacking_val, y_val)
        output = meta.predict(stacking_features)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._n_prob_columns: int | None = None

    @property
    def model_family(self) -> str:
        return "meta_learner"

    @property
    def requires_scaling(self) -> bool:
        return False

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict[str, Any]:
        return {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingMetrics:
        """Record the stacking layout; soft voting has no parameters to learn."""
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        n_prob_columns = X_train.shape[1] - N_DERIVED_STACKING_FEATURES
        if n_prob_columns <= 0 or n_prob_columns % self._n_classes != 0:
            raise ValueError(
                f"VotingMetaLearner expects n_models * {self._n_classes} probability "
                f"columns + {N_DERIVED_STACKING_FEATURES} derived columns, "
                f"got {X_train.shape[1]} columns"
            )
        self._n_prob_columns = n_prob_columns
        self._is_fitted = True

        train_probs = self._vote(X_train)
        val_probs = self._vote(X_val)
        y_train_sk = map_labels_to_classes(y_train, self._n_classes)
        y_val_sk = map_labels_to_classes(y_val, self._n_classes)
        labels = list(range(self._n_classes))
        train_pred = map_classes_to_labels(train_probs.argmax(axis=1), self._n_classes)
        val_pred = map_classes_to_labels(val_probs.argmax(axis=1), self._n_classes)

        return TrainingMetrics(
            train_loss=float(log_loss(y_train_sk, train_probs, labels=labels)),
            val_loss=float(log_loss(y_val_sk, val_probs, labels=labels)),
            train_accuracy=float(accuracy_score(y_train, train_pred)),
            val_accuracy=float(accuracy_score(y_val, val_pred)),
            train_f1=float(f1_score(y_train, train_pred, average="macro", zero_division=0)),
            val_f1=float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
            epochs_trained=0,
            training_time_seconds=time.time() - start_time,
            early_stopped=False,
            best_epoch=None,
            history={},
            metadata={
                "meta_learner": "voting",
                "n_models": n_prob_columns // self._n_classes,
            },
        )

    def _vote(self, X: np.ndarray) -> np.ndarray:
        """Mean class probability across base models, renormalized."""
        if self._n_prob_columns is None:
            raise RuntimeError("VotingMetaLearner is not fitted")
        probs = np.asarray(X[:, : self._n_prob_columns], dtype=np.float64)
        per_model = probs.reshape(len(X), -1, self._n_classes)
        mean = np.clip(per_model.mean(axis=1), 1e-12, None)
        result: np.ndarray = mean / mean.sum(axis=1, keepdims=True)
        return result

    def predict(self, X: np.ndarray) -> PredictionResult:
        self._validate_fitted()
        self._validate_input_shape(X, "X")
        probabilities = self._vote(X)
        return PredictionResult(
            class_predictions=map_classes_to_labels(probabilities.argmax(axis=1), self._n_classes),
            class_probabilities=probabilities,
            confidence=probabilities.max(axis=1),
            metadata={"meta_learner": "voting"},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X).class_probabilities

    def save(self, path: Path) -> None:
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        safe_pickle_dump(
            {
                "config": self._config,
                "n_classes": self._n_classes,
                "n_prob_columns": self._n_prob_columns,
            },
            path / "metadata.pkl",
        )

    def load(self, path: Path) -> None:
        metadata = safe_pickle_load(Path(path) / "metadata.pkl")
        self._config = metadata.get("config", self._config)
        self._n_classes = metadata.get("n_classes", 3)
        self._n_prob_columns = metadata["n_prob_columns"]
        self._is_fitted = True

    def get_feature_importance(self) -> dict[str, float] | None:
        return None


__all__ = ["VotingMetaLearner"]
