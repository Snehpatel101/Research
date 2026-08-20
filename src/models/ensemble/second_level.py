"""
Second-level stacking for large ensembles (12+ models).

Phase 16D: Meta-meta-learner implementation for hierarchical stacking.

For a 12-model ensemble, groups base models into clusters by prediction
diversity, creates intermediate meta-learners per cluster, then combines
with a final meta-learner.

Architecture:
    Base Models (12) -> Cluster Meta-Learners (3-4) -> Final Meta-Learner (1)

This hierarchical approach:
1. Reduces overfitting risk with many base models
2. Groups similar models together to reduce redundancy
3. Allows specialized meta-learners per cluster (e.g., boosting for tabular, MLP for neural)
4. Provides interpretable intermediate representations

Example:
    >>> from src.models.ensemble.second_level import (
    ...     SecondLevelStacker,
    ...     SecondLevelStackingConfig,
    ... )
    >>>
    >>> config = SecondLevelStackingConfig(
    ...     n_clusters=3,
    ...     cluster_meta_type="ridge",
    ...     final_meta_type="xgboost",
    ...     use_diversity_clustering=True,
    ... )
    >>>
    >>> stacker = SecondLevelStacker(config)
    >>> stacker.fit(oof_predictions, labels, model_names=base_model_names)
    >>> predictions = stacker.predict(new_predictions)

Reference:
    - Wolpert (1992) "Stacked Generalization"
    - Caruana et al. (2004) "Ensemble Selection from Libraries of Models"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score

from src.core.utils.safe_pickle import safe_pickle_load

logger = logging.getLogger(__name__)


@dataclass
class SecondLevelStackingConfig:
    """
    Configuration for second-level stacking.

    Attributes:
        n_clusters: Number of model clusters (default 3).
        cluster_meta_type: Meta-learner type for each cluster.
        final_meta_type: Meta-learner type for final combination.
        use_diversity_clustering: If True, cluster by prediction diversity.
        min_cluster_size: Minimum models per cluster.
        max_cluster_size: Maximum models per cluster.
        use_probabilities: If True, use class probabilities as features.
        val_ratio: Fraction of data to use for validation.
    """

    n_clusters: int = 3
    n_classes: int = 3  # 2 for binary mode; strides meta-feature slicing
    cluster_meta_type: str = "ridge"  # ridge, xgboost, mlp, or calibrated_ridge
    final_meta_type: str = "xgboost"  # ridge, xgboost, mlp, or calibrated_ridge
    use_diversity_clustering: bool = True
    min_cluster_size: int = 2
    max_cluster_size: int = 6
    use_probabilities: bool = True
    val_ratio: float = 0.2


@dataclass
class ClusterInfo:
    """Information about a model cluster."""

    cluster_id: int
    model_indices: list[int]
    model_names: list[str]
    meta_learner: Any = None  # The trained cluster meta-learner
    feature_columns: list[int] = field(default_factory=list)  # OOF feature indices


@dataclass
class SecondLevelResult:
    """Result from second-level stacking training."""

    n_clusters: int
    cluster_info: list[ClusterInfo]
    cluster_meta_type: str
    final_meta_type: str
    train_f1: float
    val_f1: float
    training_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": [len(c.model_indices) for c in self.cluster_info],
            "cluster_models": [c.model_names for c in self.cluster_info],
            "cluster_meta_type": self.cluster_meta_type,
            "final_meta_type": self.final_meta_type,
            "train_f1": self.train_f1,
            "val_f1": self.val_f1,
            "training_time_seconds": self.training_time_seconds,
            "metadata": self.metadata,
        }


class SecondLevelStacker:
    """
    Two-level stacking for large ensembles.

    Level 1: Cluster base models by prediction diversity, train cluster meta-learners.
    Level 2: Combine cluster predictions with final meta-learner.

    This hierarchical approach handles 12+ model ensembles more effectively by:
    1. Reducing the meta-learner input dimensionality
    2. Grouping similar models to reduce redundancy
    3. Allowing specialized handling per model family
    """

    def __init__(self, config: SecondLevelStackingConfig | None = None) -> None:
        """
        Initialize SecondLevelStacker.

        Args:
            config: Configuration for second-level stacking.
        """
        self.config = config or SecondLevelStackingConfig()

        # State
        self._cluster_info: list[ClusterInfo] = []
        self._final_meta_learner: Any = None
        self._n_classes: int = self.config.n_classes
        self._is_fitted: bool = False
        self._model_names: list[str] = []
        self._n_features_per_model: int = 0
        self._result: SecondLevelResult | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if stacker has been fitted."""
        return self._is_fitted

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self._cluster_info)

    def fit(
        self,
        oof_predictions: np.ndarray,
        labels: np.ndarray,
        model_names: list[str] | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> SecondLevelResult:
        """
        Fit two-level stacking.

        Args:
            oof_predictions: Shape (n_samples, n_features) where n_features is
                n_models * n_classes for probability features, or n_models for class predictions.
            labels: Shape (n_samples,) - training labels.
            model_names: Optional list of base model names.
            sample_weights: Optional sample weights.

        Returns:
            SecondLevelResult with training metrics and cluster info.
        """
        import time

        start_time = time.time()

        n_samples, n_features = oof_predictions.shape

        # Determine number of models and features per model
        if self.config.use_probabilities:
            self._n_features_per_model = self._n_classes
            n_models = n_features // self._n_classes
        else:
            self._n_features_per_model = 1
            n_models = n_features

        # Set model names
        if model_names is not None:
            if len(model_names) != n_models:
                raise ValueError(f"Expected {n_models} model names, got {len(model_names)}")
            self._model_names = model_names
        else:
            self._model_names = [f"model_{i}" for i in range(n_models)]

        logger.info(
            f"Second-level stacking: {n_models} models, "
            f"{n_samples} samples, {self.config.n_clusters} clusters"
        )

        # Step 1: Cluster models by prediction diversity
        self._cluster_models(oof_predictions, n_models)

        # Step 2: Split data for validation (time-series aware)
        n_train = int(n_samples * (1 - self.config.val_ratio))
        X_train = oof_predictions[:n_train]
        y_train = labels[:n_train]
        X_val = oof_predictions[n_train:]
        y_val = labels[n_train:]

        w_train = None
        if sample_weights is not None:
            w_train = sample_weights[:n_train]

        # Step 3: Train cluster meta-learners and get cluster-level OOF
        cluster_train_predictions = self._train_cluster_meta_learners(X_train, y_train, w_train)
        cluster_val_predictions = self._get_cluster_predictions(X_val)

        # Step 4: Train final meta-learner
        self._train_final_meta_learner(
            cluster_train_predictions,
            y_train,
            cluster_val_predictions,
            y_val,
            w_train,
        )

        # Step 5: Compute metrics
        train_preds = self.predict(X_train)
        val_preds = self.predict(X_val)

        train_f1 = float(f1_score(y_train, train_preds, average="macro", zero_division=0))
        val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))

        training_time = time.time() - start_time
        self._is_fitted = True

        logger.info(
            f"Second-level stacking complete: train_f1={train_f1:.4f}, "
            f"val_f1={val_f1:.4f}, time={training_time:.1f}s"
        )

        self._result = SecondLevelResult(
            n_clusters=self.n_clusters,
            cluster_info=self._cluster_info.copy(),
            cluster_meta_type=self.config.cluster_meta_type,
            final_meta_type=self.config.final_meta_type,
            train_f1=train_f1,
            val_f1=val_f1,
            training_time_seconds=training_time,
            metadata={
                "n_models": n_models,
                "n_samples_train": n_train,
                "n_samples_val": n_samples - n_train,
                "use_diversity_clustering": self.config.use_diversity_clustering,
                "use_probabilities": self.config.use_probabilities,
            },
        )

        return self._result

    def predict(self, oof_predictions: np.ndarray) -> np.ndarray:
        """
        Predict using two-level stacking.

        Args:
            oof_predictions: Shape (n_samples, n_features) - OOF predictions
                from base models.

        Returns:
            Class predictions of shape (n_samples,).
        """
        if not self._is_fitted:
            raise ValueError("SecondLevelStacker not fitted. Call fit() first.")

        # Get cluster-level predictions
        cluster_predictions = self._get_cluster_predictions(oof_predictions)

        # Get final predictions
        output = self._final_meta_learner.predict(cluster_predictions)
        return output.class_predictions

    def predict_proba(self, oof_predictions: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using two-level stacking.

        Args:
            oof_predictions: Shape (n_samples, n_features) - OOF predictions
                from base models.

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        if not self._is_fitted:
            raise ValueError("SecondLevelStacker not fitted. Call fit() first.")

        cluster_predictions = self._get_cluster_predictions(oof_predictions)
        output = self._final_meta_learner.predict(cluster_predictions)
        return output.class_probabilities

    def _cluster_models(
        self,
        oof_predictions: np.ndarray,
        n_models: int,
    ) -> None:
        """
        Cluster models based on prediction diversity.

        Uses agglomerative clustering on the correlation matrix of model predictions.
        Models with similar predictions are grouped together.
        """
        # Compute correlation matrix between models
        if self.config.use_diversity_clustering:
            correlation_matrix = self._compute_model_correlation_matrix(oof_predictions, n_models)
            # Convert correlation to distance: higher correlation = lower distance
            distance_matrix = 1 - np.abs(correlation_matrix)
        else:
            # Use random clustering (for comparison/testing)
            distance_matrix = None

        # Determine optimal number of clusters
        n_clusters = min(self.config.n_clusters, n_models // 2)
        n_clusters = max(n_clusters, 1)

        if distance_matrix is not None:
            # Agglomerative clustering with precomputed distances
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
        else:
            # Random assignment
            cluster_labels = np.arange(n_models) % n_clusters

        # Build cluster info
        self._cluster_info = []
        for cluster_id in range(n_clusters):
            model_indices = np.where(cluster_labels == cluster_id)[0].tolist()

            # Skip empty clusters
            if not model_indices:
                continue

            # Ensure minimum cluster size
            if len(model_indices) < self.config.min_cluster_size:
                # Steal from largest cluster or merge
                continue

            # Compute feature column indices for this cluster
            feature_cols = []
            for idx in model_indices:
                start = idx * self._n_features_per_model
                end = start + self._n_features_per_model
                feature_cols.extend(range(start, end))

            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                model_indices=model_indices,
                model_names=[self._model_names[i] for i in model_indices],
                feature_columns=feature_cols,
            )
            self._cluster_info.append(cluster_info)

        # Log clustering results
        for cluster in self._cluster_info:
            logger.info(
                f"Cluster {cluster.cluster_id}: {len(cluster.model_indices)} models "
                f"({', '.join(cluster.model_names[:3])}{'...' if len(cluster.model_names) > 3 else ''})"
            )

    def _compute_model_correlation_matrix(
        self,
        oof_predictions: np.ndarray,
        n_models: int,
    ) -> np.ndarray:
        """
        Compute pairwise correlation matrix between model predictions.

        For probability features, uses the max probability (argmax class probability).
        """
        # Extract per-model predictions (use max probability for each model)
        model_predictions = []
        for i in range(n_models):
            if self.config.use_probabilities:
                start = i * self._n_classes
                end = start + self._n_classes
                probs = oof_predictions[:, start:end]
                # Use max probability as representative prediction
                model_pred = np.max(probs, axis=1)
            else:
                model_pred = oof_predictions[:, i]
            model_predictions.append(model_pred)

        model_predictions = np.column_stack(model_predictions)

        # Compute correlation matrix
        correlation_matrix = np.corrcoef(model_predictions.T)

        # Handle NaN (constant predictions)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        return correlation_matrix

    def _train_cluster_meta_learners(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weights: np.ndarray | None,
    ) -> np.ndarray:
        """
        Train a meta-learner for each cluster and return cluster-level OOF predictions.
        """
        from src.models.ensemble import MetaLearnerFactory

        factory = MetaLearnerFactory()

        # Map short names to factory names
        meta_type_map = {
            "ridge": "ridge_meta",
            "xgboost": "xgboost_meta",
            "mlp": "mlp_meta",
            "calibrated_ridge": "calibrated_meta",
        }
        cluster_meta_name = meta_type_map.get(
            self.config.cluster_meta_type,
            self.config.cluster_meta_type + "_meta",
        )

        cluster_predictions = []

        for cluster in self._cluster_info:
            # Extract features for this cluster
            X_cluster = X_train[:, cluster.feature_columns]

            # Create and train meta-learner
            meta_learner = factory.create(cluster_meta_name)

            # Use simple train/val split within cluster training
            n_samples = len(y_train)
            split = int(n_samples * 0.8)

            X_train_cluster = X_cluster[:split]
            y_train_cluster = y_train[:split]
            X_val_cluster = X_cluster[split:]
            y_val_cluster = y_train[split:]

            w_train = None
            if sample_weights is not None:
                w_train = sample_weights[:split]

            meta_learner.fit(
                X_train=X_train_cluster,
                y_train=y_train_cluster,
                X_val=X_val_cluster,
                y_val=y_val_cluster,
                sample_weights=w_train,
            )

            cluster.meta_learner = meta_learner

            # Get cluster predictions for full training set
            output = meta_learner.predict(X_cluster)
            cluster_predictions.append(output.class_probabilities)

        # Stack cluster predictions
        return np.hstack(cluster_predictions)

    def _get_cluster_predictions(
        self,
        oof_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Get predictions from each cluster meta-learner.
        """
        cluster_predictions = []

        for cluster in self._cluster_info:
            X_cluster = oof_predictions[:, cluster.feature_columns]
            output = cluster.meta_learner.predict(X_cluster)
            cluster_predictions.append(output.class_probabilities)

        return np.hstack(cluster_predictions)

    def _train_final_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None,
    ) -> None:
        """
        Train the final meta-learner on cluster-level predictions.
        """
        from src.models.ensemble import MetaLearnerFactory

        factory = MetaLearnerFactory()

        meta_type_map = {
            "ridge": "ridge_meta",
            "xgboost": "xgboost_meta",
            "mlp": "mlp_meta",
            "calibrated_ridge": "calibrated_meta",
        }
        final_meta_name = meta_type_map.get(
            self.config.final_meta_type,
            self.config.final_meta_type + "_meta",
        )

        self._final_meta_learner = factory.create(final_meta_name)

        self._final_meta_learner.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weights=sample_weights,
        )

    def save(self, path: Path | str) -> None:
        """
        Save second-level stacker to disk.

        Args:
            path: Directory to save to.
        """
        if not self._is_fitted:
            raise ValueError("SecondLevelStacker not fitted. Call fit() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save cluster meta-learners
        for i, cluster in enumerate(self._cluster_info):
            cluster_path = path / f"cluster_{i}"
            cluster.meta_learner.save(cluster_path)

        # Save final meta-learner
        final_path = path / "final_meta_learner"
        self._final_meta_learner.save(final_path)

        # Save metadata
        metadata = {
            "config": {
                "n_clusters": self.config.n_clusters,
                "cluster_meta_type": self.config.cluster_meta_type,
                "final_meta_type": self.config.final_meta_type,
                "use_diversity_clustering": self.config.use_diversity_clustering,
                "use_probabilities": self.config.use_probabilities,
                "val_ratio": self.config.val_ratio,
            },
            "cluster_info": [
                {
                    "cluster_id": c.cluster_id,
                    "model_indices": c.model_indices,
                    "model_names": c.model_names,
                    "feature_columns": c.feature_columns,
                }
                for c in self._cluster_info
            ],
            "n_classes": self._n_classes,
            "n_features_per_model": self._n_features_per_model,
            "model_names": self._model_names,
        }
        joblib.dump(metadata, path / "second_level_metadata.joblib")

        logger.info(f"Saved SecondLevelStacker to {path}")

    def load(self, path: Path | str) -> None:
        """
        Load second-level stacker from disk.

        Args:
            path: Directory to load from.
        """
        path = Path(path)

        if not (path / "second_level_metadata.joblib").exists():
            raise FileNotFoundError(f"Metadata not found at {path}")

        # Load metadata
        metadata = safe_pickle_load(path / "second_level_metadata.joblib")

        # Restore config
        config_dict = metadata["config"]
        self.config = SecondLevelStackingConfig(**config_dict)

        # Restore state
        self._n_classes = metadata["n_classes"]
        self._n_features_per_model = metadata["n_features_per_model"]
        self._model_names = metadata["model_names"]

        # Load cluster meta-learners
        from src.models.ensemble import MetaLearnerFactory

        factory = MetaLearnerFactory()

        meta_type_map = {
            "ridge": "ridge_meta",
            "xgboost": "xgboost_meta",
            "mlp": "mlp_meta",
            "calibrated_ridge": "calibrated_meta",
        }
        cluster_meta_name = meta_type_map.get(
            self.config.cluster_meta_type,
            self.config.cluster_meta_type + "_meta",
        )

        self._cluster_info = []
        for i, cluster_data in enumerate(metadata["cluster_info"]):
            cluster = ClusterInfo(
                cluster_id=cluster_data["cluster_id"],
                model_indices=cluster_data["model_indices"],
                model_names=cluster_data["model_names"],
                feature_columns=cluster_data["feature_columns"],
            )

            meta_learner = factory.create(cluster_meta_name)
            meta_learner.load(path / f"cluster_{i}")
            cluster.meta_learner = meta_learner

            self._cluster_info.append(cluster)

        # Load final meta-learner
        final_meta_name = meta_type_map.get(
            self.config.final_meta_type,
            self.config.final_meta_type + "_meta",
        )
        self._final_meta_learner = factory.create(final_meta_name)
        self._final_meta_learner.load(path / "final_meta_learner")

        self._is_fitted = True
        logger.info(f"Loaded SecondLevelStacker from {path}")

    def get_cluster_summary(self) -> dict[str, Any]:
        """
        Get summary of model clustering.

        Returns:
            Dictionary with cluster information.
        """
        if not self._cluster_info:
            return {"n_clusters": 0, "clusters": []}

        return {
            "n_clusters": len(self._cluster_info),
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "n_models": len(c.model_indices),
                    "model_names": c.model_names,
                }
                for c in self._cluster_info
            ],
        }


def build_second_level_stacking(
    oof_predictions: np.ndarray,
    labels: np.ndarray,
    model_names: list[str] | None = None,
    n_clusters: int = 3,
    cluster_meta_type: str = "ridge",
    final_meta_type: str = "xgboost",
) -> SecondLevelStacker:
    """
    Convenience function to build a second-level stacker.

    Args:
        oof_predictions: OOF predictions from base models.
        labels: Training labels.
        model_names: Optional base model names.
        n_clusters: Number of model clusters.
        cluster_meta_type: Meta-learner type for clusters.
        final_meta_type: Meta-learner type for final combination.

    Returns:
        Fitted SecondLevelStacker.
    """
    config = SecondLevelStackingConfig(
        n_clusters=n_clusters,
        cluster_meta_type=cluster_meta_type,
        final_meta_type=final_meta_type,
    )

    stacker = SecondLevelStacker(config)
    stacker.fit(oof_predictions, labels, model_names)

    return stacker


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SecondLevelStackingConfig",
    "SecondLevelStacker",
    "SecondLevelResult",
    "ClusterInfo",
    "build_second_level_stacking",
]
