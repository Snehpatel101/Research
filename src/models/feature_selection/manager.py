"""
Feature Selection Manager for Model Training.

Provides a unified interface for running feature selection and applying
the results to training and inference data. Integrates with the existing
WalkForwardFeatureSelector while adding persistence and model-family awareness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.cross_validation.feature_selector import (
    FeatureSelectionResult,
    WalkForwardFeatureSelector,
)
from src.cross_validation.purged_kfold import PurgedKFold

from .config import FeatureSelectionConfig, ModelFamilyDefaults
from .result import PersistedFeatureSelection

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FeatureSelectionManager:
    """
    Manages feature selection for model training pipeline.

    Provides:
    - Model-family-aware feature selection configuration
    - Walk-forward feature selection execution
    - Feature subset application to training/inference data
    - Persistence with model artifacts

    Example:
        >>> manager = FeatureSelectionManager.from_model_family("boosting")
        >>> result = manager.select_features(X_train_df, y_train, sample_weights)
        >>> X_train_selected = manager.apply_selection(X_train_df)
        >>> manager.save(model_path / "feature_selection.json")
    """

    def __init__(
        self,
        config: FeatureSelectionConfig | None = None,
        n_features: int = 50,
        method: str = "mda",
        model_family: str | None = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize FeatureSelectionManager.

        Args:
            config: FeatureSelectionConfig instance (preferred)
            n_features: Number of features to select (if config not provided)
            method: Feature importance method (if config not provided)
            model_family: Model family for auto-configuration
            random_state: Random seed for reproducibility
        """
        if config is not None:
            self.config = config
        elif model_family is not None:
            self.config = FeatureSelectionConfig.from_model_family(
                model_family,
                override={"n_features": n_features, "method": method, "random_state": random_state},
            )
        else:
            self.config = FeatureSelectionConfig(
                n_features=n_features,
                method=method,
                random_state=random_state,
            )

        self._result: PersistedFeatureSelection | None = None
        self._selector: WalkForwardFeatureSelector | None = None
        self._all_features: list[str] | None = None

    @classmethod
    def from_model_family(
        cls,
        model_family: str,
        n_features: int | None = None,
        method: str | None = None,
        random_state: int = 42,
    ) -> FeatureSelectionManager:
        """
        Create manager with model-family-specific defaults.

        Args:
            model_family: Model family ('boosting', 'classical', 'neural', etc.)
            n_features: Override default n_features (optional)
            method: Override default method (optional)
            random_state: Random seed

        Returns:
            FeatureSelectionManager instance
        """
        override: dict[str, Any] = {"random_state": random_state}
        if n_features is not None:
            override["n_features"] = n_features
        if method is not None:
            override["method"] = method

        config = FeatureSelectionConfig.from_model_family(model_family, override)
        return cls(config=config)

    @classmethod
    def disabled(cls) -> FeatureSelectionManager:
        """Create a disabled manager (passthrough)."""
        return cls(config=FeatureSelectionConfig.disabled())

    @property
    def is_enabled(self) -> bool:
        """Check if feature selection is enabled."""
        return self.config.enabled

    @property
    def is_fitted(self) -> bool:
        """Check if feature selection has been run."""
        return self._result is not None

    @property
    def selected_features(self) -> list[str]:
        """Get list of selected feature names."""
        if self._result is None:
            return []
        return self._result.selected_features

    @property
    def n_features_selected(self) -> int:
        """Get number of selected features."""
        if self._result is None:
            return 0
        return self._result.n_features_selected

    @property
    def result(self) -> PersistedFeatureSelection | None:
        """Get the persisted feature selection result."""
        return self._result

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weights: pd.Series | np.ndarray | None = None,
        n_splits: int = 5,
        purge_bars: int = 60,
        embargo_bars: int = 1440,
        label_end_times: pd.Series | None = None,
    ) -> PersistedFeatureSelection:
        """
        Run walk-forward feature selection.

        Performs feature selection using time-series aware cross-validation
        to prevent lookahead bias. Features that appear consistently across
        folds are selected as stable features.

        Args:
            X: Feature DataFrame with named columns
            y: Labels (Series or array)
            sample_weights: Optional sample weights
            n_splits: Number of CV folds for stability analysis
            purge_bars: Number of bars to purge between train/test
            embargo_bars: Number of bars to embargo after test
            label_end_times: Optional label end times for overlapping label purging

        Returns:
            PersistedFeatureSelection with selected features

        Raises:
            ValueError: If X is not a DataFrame or feature selection is disabled
        """
        if not self.config.enabled:
            logger.info("Feature selection disabled, using all features")
            self._all_features = list(X.columns)
            self._result = PersistedFeatureSelection.passthrough(self._all_features)
            return self._result

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame with named columns for feature selection"
            )

        self._all_features = list(X.columns)
        n_features_original = len(self._all_features)

        # Determine number of features to select
        n_to_select = self.config.n_features
        if n_to_select <= 0 or n_to_select >= n_features_original:
            logger.info(
                f"n_features={n_to_select} >= total features ({n_features_original}), "
                "using all features"
            )
            self._result = PersistedFeatureSelection.passthrough(self._all_features)
            return self._result

        logger.info(
            f"Running walk-forward feature selection: "
            f"n_features={n_to_select}, method={self.config.method}, "
            f"n_splits={n_splits}"
        )

        # Create CV splits for walk-forward selection
        cv = PurgedKFold(
            n_splits=n_splits,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Convert y to Series if needed
        y_series = pd.Series(y) if isinstance(y, np.ndarray) else y
        w_series = (
            pd.Series(sample_weights) if isinstance(sample_weights, np.ndarray) else sample_weights
        )

        # Generate CV splits
        cv_splits = list(cv.split(X, y_series, label_end_times=label_end_times))

        # Initialize selector
        self._selector = WalkForwardFeatureSelector(
            n_features_to_select=n_to_select,
            selection_method=self.config.method,
            n_estimators=self.config.n_estimators,
            min_feature_frequency=self.config.min_feature_frequency,
            use_clustered_importance=self.config.use_clustered_importance,
            max_clusters=self.config.max_clusters,
            random_state=self.config.random_state,
        )

        # Run walk-forward feature selection
        selection_result: FeatureSelectionResult = self._selector.select_features_walkforward(
            X=X,
            y=y_series,
            cv_splits=cv_splits,
            sample_weights=w_series,
        )

        # Build stable feature list
        stable_features = selection_result.stable_features

        # If no stable features found, fall back to selecting top features from first fold
        if not stable_features:
            logger.warning(
                f"No stable features found (min_frequency={self.config.min_feature_frequency}). "
                f"Falling back to top {n_to_select} features from first fold."
            )
            if selection_result.per_fold_selections:
                first_fold_features = list(selection_result.per_fold_selections[0])
                stable_features = first_fold_features[:n_to_select]
            else:
                # Ultimate fallback: use first n_to_select features
                stable_features = self._all_features[:n_to_select]

        # Build feature indices
        feature_indices = {f: self._all_features.index(f) for f in stable_features}

        # Get stability scores
        stability_scores = selection_result.get_stability_scores()

        # Get importance scores from last fold (most recent data)
        importance_scores: dict[str, float] = {}
        if selection_result.importance_history:
            last_importance = selection_result.importance_history[-1].get("importance", {})
            importance_scores = {f: last_importance.get(f, 0.0) for f in stable_features}

        # Create persisted result
        self._result = PersistedFeatureSelection(
            selected_features=stable_features,
            feature_indices=feature_indices,
            selection_method=self.config.method,
            n_features_original=n_features_original,
            n_features_selected=len(stable_features),
            stability_scores={f: stability_scores.get(f, 0.0) for f in stable_features},
            importance_scores=importance_scores,
            metadata={
                "n_splits": n_splits,
                "min_feature_frequency": self.config.min_feature_frequency,
                "n_features_requested": n_to_select,
                "model_family": self.config.model_family,
            },
        )

        logger.info(
            f"Feature selection complete: {self._result.n_features_selected} features selected "
            f"(reduction: {self._result.reduction_ratio:.1%})"
        )

        return self._result

    def select_features_single_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        sample_weights: pd.Series | np.ndarray | None = None,
    ) -> PersistedFeatureSelection:
        """
        Run feature selection on a single training fold.

        Faster alternative to walk-forward selection when CV-based
        stability analysis is not needed.

        Args:
            X_train: Training feature DataFrame
            y_train: Training labels
            sample_weights: Optional sample weights

        Returns:
            PersistedFeatureSelection with selected features
        """
        if not self.config.enabled:
            self._all_features = list(X_train.columns)
            self._result = PersistedFeatureSelection.passthrough(self._all_features)
            return self._result

        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame with named columns")

        self._all_features = list(X_train.columns)
        n_features_original = len(self._all_features)

        n_to_select = self.config.n_features
        if n_to_select <= 0 or n_to_select >= n_features_original:
            self._result = PersistedFeatureSelection.passthrough(self._all_features)
            return self._result

        logger.info(
            f"Running single-fold feature selection: "
            f"n_features={n_to_select}, method={self.config.method}"
        )

        # Initialize selector
        self._selector = WalkForwardFeatureSelector(
            n_features_to_select=n_to_select,
            selection_method=self.config.method,
            n_estimators=self.config.n_estimators,
            min_feature_frequency=1.0,  # Not used for single fold
            use_clustered_importance=self.config.use_clustered_importance,
            max_clusters=self.config.max_clusters,
            random_state=self.config.random_state,
        )

        # Convert y to Series if needed
        y_series = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
        w_series = (
            pd.Series(sample_weights) if isinstance(sample_weights, np.ndarray) else sample_weights
        )

        # Compute importance
        importance = self._selector._compute_importance(X_train, y_series, w_series)

        # Select top features
        selected_features = importance.nlargest(n_to_select).index.tolist()

        # Build result
        feature_indices = {f: self._all_features.index(f) for f in selected_features}

        self._result = PersistedFeatureSelection(
            selected_features=selected_features,
            feature_indices=feature_indices,
            selection_method=self.config.method,
            n_features_original=n_features_original,
            n_features_selected=len(selected_features),
            stability_scores={f: 1.0 for f in selected_features},
            importance_scores=importance.to_dict(),
            metadata={
                "single_fold": True,
                "model_family": self.config.model_family,
            },
        )

        logger.info(
            f"Single-fold feature selection complete: "
            f"{self._result.n_features_selected} features selected"
        )

        return self._result

    def apply_selection(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Apply feature selection to data.

        Args:
            X: Input data (DataFrame or ndarray)
            feature_names: Feature names if X is ndarray (required if ndarray)

        Returns:
            numpy array with selected features only

        Raises:
            RuntimeError: If feature selection has not been run
            ValueError: If feature names don't match
        """
        if self._result is None:
            raise RuntimeError("Feature selection has not been run. Call select_features() first.")

        # Get feature names
        if isinstance(X, pd.DataFrame):
            all_features = list(X.columns)
            X_values = X.values
        else:
            if feature_names is None:
                raise ValueError("feature_names must be provided when X is a numpy array")
            all_features = feature_names
            X_values = X

        # Get column indices for selected features
        indices = self._result.get_column_indices(all_features)

        if not indices:
            raise ValueError(
                "No selected features found in input data. "
                f"Selected: {self._result.selected_features[:5]}..., "
                f"Available: {all_features[:5]}..."
            )

        return X_values[:, indices]

    def apply_selection_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection and return a DataFrame.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with selected features only
        """
        if self._result is None:
            raise RuntimeError("Feature selection has not been run. Call select_features() first.")

        return X[self._result.selected_features]

    def save(self, path: Path) -> None:
        """
        Save feature selection result to file.

        Args:
            path: Path to save JSON file
        """
        if self._result is None:
            raise RuntimeError("No feature selection result to save. Call select_features() first.")
        self._result.save(path)
        logger.info(f"Saved feature selection to {path}")

    def load(self, path: Path) -> PersistedFeatureSelection:
        """
        Load feature selection result from file.

        Args:
            path: Path to load from

        Returns:
            PersistedFeatureSelection instance
        """
        self._result = PersistedFeatureSelection.load(path)
        logger.info(
            f"Loaded feature selection from {path}: " f"{self._result.n_features_selected} features"
        )
        return self._result

    @classmethod
    def load_from_path(cls, path: Path) -> FeatureSelectionManager:
        """
        Create manager by loading from saved file.

        Args:
            path: Path to saved feature selection JSON

        Returns:
            FeatureSelectionManager with loaded result
        """
        manager = cls()
        manager.load(path)
        return manager

    def get_feature_report(self) -> dict[str, Any]:
        """
        Generate a report of feature selection results.

        Returns:
            Dict with selection statistics and top features
        """
        if self._result is None:
            return {"status": "not_run"}

        # Sort features by stability score
        sorted_features = sorted(
            self._result.stability_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "status": "complete",
            "n_features_original": self._result.n_features_original,
            "n_features_selected": self._result.n_features_selected,
            "reduction_ratio": self._result.reduction_ratio,
            "selection_method": self._result.selection_method,
            "top_10_features": [{"name": f, "stability": s} for f, s in sorted_features[:10]],
            "metadata": self._result.metadata,
        }


__all__ = ["FeatureSelectionManager"]
