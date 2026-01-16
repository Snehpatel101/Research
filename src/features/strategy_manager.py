"""
Feature Strategy Manager - Bridges MODEL_FEATURE_STRATEGIES to training configuration.

This module provides the integration layer between feature strategies defined per model
and the actual training workflow, resolving baseline features against available data
and validating feature diversity for ensembles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from .strategies import MODEL_FEATURE_STRATEGIES, ModelFeatureStrategy, get_strategy_for_model
from src.phase1.utils.constants import METADATA_COLUMNS
from src.phase1.utils.feature_sets import _is_label_column

logger = logging.getLogger(__name__)


@dataclass
class ResolvedFeatureSet:
    """
    Resolved feature set for a specific model.

    Contains the final list of features after filtering strategy baseline
    against available data columns.
    """

    model_name: str
    feature_columns: list[str]
    n_features: int = field(init=False)
    baseline_requested: int = 0
    baseline_available: int = 0
    optimized: bool = False
    optimization_method: str = ""
    included_families: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compute n_features from feature_columns length."""
        self.n_features = len(self.feature_columns)

    def __repr__(self) -> str:
        opt_str = f", optimized={self.optimization_method}" if self.optimized else ""
        return (
            f"ResolvedFeatureSet({self.model_name}, "
            f"n_features={self.n_features}, "
            f"baseline={self.baseline_available}/{self.baseline_requested}{opt_str})"
        )


class FeatureStrategyManager:
    """
    Manages feature strategy resolution for model training.

    Bridges MODEL_FEATURE_STRATEGIES to the training configuration flow by:
    - Detecting available features from data
    - Resolving strategy baselines against available features
    - Validating feature counts against strategy bounds
    - Supporting ensemble feature diversity validation
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        available_features: list[str] | None = None,
    ) -> None:
        """
        Initialize the feature strategy manager.

        Args:
            df: DataFrame to detect features from (optional if available_features provided)
            available_features: Pre-computed list of available feature columns

        Raises:
            ValueError: If neither df nor available_features is provided
        """
        if df is None and available_features is None:
            raise ValueError("Must provide either df or available_features")

        if available_features is not None:
            self._available_features = list(available_features)
        else:
            self._available_features = self._detect_features(df)

        self._available_set = set(self._available_features)
        self._cache: dict[str, ResolvedFeatureSet] = {}

        logger.debug(
            "FeatureStrategyManager initialized with %d available features",
            len(self._available_features),
        )

    def _detect_features(self, df: pd.DataFrame) -> list[str]:
        """
        Detect feature columns from DataFrame.

        Excludes metadata columns and label columns.

        Args:
            df: DataFrame to detect features from

        Returns:
            List of feature column names
        """
        features = []
        for col in df.columns:
            if col in METADATA_COLUMNS:
                continue
            if _is_label_column(col):
                continue
            features.append(col)
        return features

    def get_strategy(self, model_name: str) -> ModelFeatureStrategy:
        """
        Get the feature strategy for a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelFeatureStrategy for the model

        Raises:
            ValueError: If model has no defined strategy
        """
        return get_strategy_for_model(model_name)

    def get_features_for_model(
        self,
        model_name: str,
        custom_baseline: list[str] | None = None,
        strict: bool = True,
    ) -> ResolvedFeatureSet:
        """
        Get resolved features for a model.

        Resolves the model's strategy baseline against available features,
        with optional custom baseline override.

        Args:
            model_name: Name of the model
            custom_baseline: Optional custom baseline to use instead of strategy default
            strict: If True, raise ValueError when available features < min_features

        Returns:
            ResolvedFeatureSet with filtered features

        Raises:
            ValueError: If strict=True and available features below strategy minimum
        """
        cache_key = f"{model_name}:{custom_baseline is not None}:{strict}"
        if cache_key in self._cache and custom_baseline is None:
            return self._cache[cache_key]

        strategy = self.get_strategy(model_name)

        baseline = custom_baseline if custom_baseline is not None else strategy.baseline_features
        baseline_requested = len(baseline)

        available_baseline = [f for f in baseline if f in self._available_set]
        baseline_available = len(available_baseline)

        if baseline_available < baseline_requested:
            missing = set(baseline) - self._available_set
            logger.info(
                "Model '%s': %d/%d baseline features available (missing: %s)",
                model_name,
                baseline_available,
                baseline_requested,
                list(missing)[:5],
            )

        if strict and baseline_available < strategy.min_features:
            raise ValueError(
                f"Model '{model_name}' requires at least {strategy.min_features} features, "
                f"but only {baseline_available} baseline features are available. "
                f"Missing features: {list(set(baseline) - self._available_set)[:10]}"
            )

        resolved = ResolvedFeatureSet(
            model_name=model_name,
            feature_columns=available_baseline,
            baseline_requested=baseline_requested,
            baseline_available=baseline_available,
            optimized=False,
            optimization_method="",
            included_families=list(strategy.preferred_families),
        )

        if custom_baseline is None:
            self._cache[cache_key] = resolved

        return resolved

    def get_features_for_ensemble(
        self,
        base_models: list[str],
        strict: bool = True,
    ) -> dict[str, ResolvedFeatureSet]:
        """
        Get resolved features for all base models in an ensemble.

        Args:
            base_models: List of base model names
            strict: If True, raise ValueError when available features < min_features

        Returns:
            Dictionary mapping model name to ResolvedFeatureSet

        Raises:
            ValueError: If strict=True and any model has insufficient features
        """
        return {model: self.get_features_for_model(model, strict=strict) for model in base_models}

    def validate_feature_diversity(
        self,
        base_models: list[str],
        min_unique_ratio: float = 0.3,
        strict: bool = False,
    ) -> tuple[bool, list[str]]:
        """
        Validate feature diversity across ensemble base models.

        Checks that base models have sufficient feature diversity to benefit
        from ensemble combination. Models with highly overlapping features
        may not provide ensemble diversity benefits.

        Args:
            base_models: List of base model names
            min_unique_ratio: Minimum ratio of unique features per model
                            (features not shared with all other models)
            strict: If True, raise ValueError when available features < min_features
                   (defaults to False for diversity validation)

        Returns:
            Tuple of (is_valid, warning_messages)
            - is_valid: True if diversity criteria met
            - warning_messages: List of warnings if diversity is low
        """
        if len(base_models) < 2:
            return True, []

        feature_sets = self.get_features_for_ensemble(base_models, strict=strict)
        all_features = [set(fs.feature_columns) for fs in feature_sets.values()]

        shared_features = set.intersection(*all_features)
        total_unique = set.union(*all_features)

        warnings = []
        is_valid = True

        if not total_unique:
            return False, ["No features available for any model"]

        for model, resolved in feature_sets.items():
            model_features = set(resolved.feature_columns)
            unique_to_model = model_features - shared_features

            if model_features:
                unique_ratio = len(unique_to_model) / len(model_features)
                if unique_ratio < min_unique_ratio:
                    is_valid = False
                    warnings.append(
                        f"Model '{model}' has low feature diversity: "
                        f"{unique_ratio:.1%} unique features (min: {min_unique_ratio:.1%})"
                    )

        overlap_ratio = len(shared_features) / len(total_unique) if total_unique else 0
        if overlap_ratio > 0.8:
            warnings.append(
                f"High feature overlap across ensemble: {overlap_ratio:.1%} shared features. "
                "Consider using models with more diverse feature strategies."
            )

        if not warnings:
            logger.debug(
                "Feature diversity validation passed for %d models: "
                "%d shared features, %d total unique features",
                len(base_models),
                len(shared_features),
                len(total_unique),
            )

        return is_valid, warnings


def get_features_for_model(
    model_name: str,
    df: pd.DataFrame,
    strict: bool = True,
) -> list[str]:
    """
    Convenience function to get features for a model from a DataFrame.

    Args:
        model_name: Name of the model
        df: DataFrame with feature columns
        strict: If True, raise ValueError when available features < min_features

    Returns:
        List of feature column names for the model

    Raises:
        ValueError: If strict=True and available features below strategy minimum
    """
    manager = FeatureStrategyManager(df=df)
    resolved = manager.get_features_for_model(model_name, strict=strict)
    return resolved.feature_columns


__all__ = [
    "ResolvedFeatureSet",
    "FeatureStrategyManager",
    "get_features_for_model",
]
