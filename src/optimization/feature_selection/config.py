"""
Configuration for per-model feature selection.

Provides model-family-specific defaults and configuration for feature selection.
Different model families have different optimal feature selection strategies:
- Boosting models: Benefit from MDI/MDA-based selection (50 features)
- Classical models: Benefit from more aggressive feature reduction (40 features)
- Neural/Sequence models: Walk-forward MDA with moderate features (80 features)
- Transformer models: Walk-forward MDA with curated features (60 features)

This module consolidates configuration from:
- src/models/feature_selection/config.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelFamilyDefaults:
    """
    Default feature selection settings per model family.

    Each model family has different optimal feature selection configurations:
    - Boosting: Benefits from MDA selection, moderate feature count (50)
    - Classical: Benefits from more aggressive feature reduction (40)
    - Neural/Sequence: Walk-forward MDA with moderate features (80)
    - Transformer: Walk-forward MDA with curated features (60)
    - Ensemble: Inherits from base model family (disabled)
    """

    # Boosting models (XGBoost, LightGBM, CatBoost)
    BOOSTING = {
        "enabled": True,
        "n_features": 50,
        "method": "mda",
        "min_feature_frequency": 0.6,
        "n_estimators": 50,
    }

    # Classical models (Random Forest, Logistic, SVM)
    CLASSICAL = {
        "enabled": True,
        "n_features": 40,
        "method": "mda",
        "min_feature_frequency": 0.7,
        "n_estimators": 50,
    }

    # Neural/sequence models (LSTM, GRU, TCN, InceptionTime, ResNet, N-BEATS)
    # RNN/CNN models benefit from moderate feature reduction via walk-forward MDA
    NEURAL = {
        "enabled": True,
        "n_features": 80,
        "method": "mda",
        "min_feature_frequency": 0.5,
        "n_estimators": 50,
    }

    # Transformer models (PatchTST, iTransformer, TFT)
    # Transformers benefit from more curated feature sets
    TRANSFORMER = {
        "enabled": True,
        "n_features": 60,
        "method": "mda",
        "min_feature_frequency": 0.5,
        "n_estimators": 50,
    }

    # Ensemble models - typically inherit from base model family
    ENSEMBLE = {
        "enabled": False,  # Applied to base models instead
        "n_features": 0,
        "method": "mda",
        "min_feature_frequency": 0.5,
        "n_estimators": 50,
    }

    @classmethod
    def get_defaults(cls, model_family: str) -> dict[str, Any]:
        """
        Get default feature selection config for a model family.

        Args:
            model_family: One of 'boosting', 'classical', 'neural', 'transformer', 'ensemble'

        Returns:
            Dict with default configuration values

        Raises:
            ValueError: If model_family is not recognized
        """
        family_lower = model_family.lower().strip()

        mapping = {
            "boosting": cls.BOOSTING,
            "classical": cls.CLASSICAL,
            "neural": cls.NEURAL,
            "sequence": cls.NEURAL,  # Alias for neural
            "transformer": cls.TRANSFORMER,
            "patchtst": cls.TRANSFORMER,
            "itransformer": cls.TRANSFORMER,
            "tft": cls.TRANSFORMER,
            "ensemble": cls.ENSEMBLE,
        }

        if family_lower not in mapping:
            raise ValueError(
                f"Unknown model family '{model_family}'. " f"Valid families: {list(mapping.keys())}"
            )

        return mapping[family_lower].copy()

    @classmethod
    def is_enabled_by_default(cls, model_family: str) -> bool:
        """Check if feature selection is enabled by default for a model family."""
        try:
            defaults = cls.get_defaults(model_family)
            return bool(defaults.get("enabled", False))
        except ValueError:
            return False


@dataclass
class FeatureSelectionConfig:
    """
    Configuration for feature selection in model training.

    Attributes:
        enabled: Whether to perform feature selection
        n_features: Number of top features to select (0 = use all)
        method: Feature importance method ('mda', 'mdi', 'hybrid')
        min_feature_frequency: Minimum fold frequency for stable features
        n_estimators: Number of trees for importance calculation
        use_clustered_importance: Use clustered MDA for correlated features
        max_clusters: Maximum number of feature clusters
        random_state: Random seed for reproducibility
        model_family: Model family for auto-configuration (optional)
    """

    enabled: bool = True
    n_features: int = 50
    method: str = "mda"
    min_feature_frequency: float = 0.6
    n_estimators: int = 50
    mda_n_repeats: int = 5
    use_clustered_importance: bool = False
    max_clusters: int = 20
    mtf_max_per_timeframe: int = 8
    regime_conditional: bool = False
    random_state: int = 42
    model_family: str | None = None

    # Store the selected feature names after selection is run
    _selected_features: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and apply model family defaults."""
        # Apply model family defaults if specified
        if self.model_family and self.enabled:
            self._apply_family_defaults()

        # Validation
        if self.enabled:
            if self.n_features < 0:
                raise ValueError(f"n_features must be >= 0, got {self.n_features}")
            if self.method not in ("mda", "mdi", "hybrid"):
                raise ValueError(f"method must be 'mda', 'mdi', or 'hybrid', got '{self.method}'")
            if not 0 < self.min_feature_frequency <= 1:
                raise ValueError(
                    f"min_feature_frequency must be in (0, 1], got {self.min_feature_frequency}"
                )
            if self.n_estimators <= 0:
                raise ValueError(f"n_estimators must be > 0, got {self.n_estimators}")
            if self.mda_n_repeats <= 0:
                raise ValueError(f"mda_n_repeats must be > 0, got {self.mda_n_repeats}")

    def _apply_family_defaults(self) -> None:
        """Apply model family defaults where not explicitly set."""
        if not self.model_family:
            return

        defaults = ModelFamilyDefaults.get_defaults(self.model_family)

        # Only override if using default values (indicates not explicitly set)
        # We check against the class defaults
        if self.n_features == 50:  # Default value
            self.n_features = defaults.get("n_features", self.n_features)
        if self.method == "mda":  # Default value
            self.method = defaults.get("method", self.method)
        if self.min_feature_frequency == 0.6:  # Default value
            self.min_feature_frequency = defaults.get(
                "min_feature_frequency", self.min_feature_frequency
            )

        # Check if family disables feature selection by default
        if not defaults.get("enabled", True):
            self.enabled = False

    @classmethod
    def from_model_family(
        cls,
        model_family: str,
        override: dict[str, Any] | None = None,
    ) -> FeatureSelectionConfig:
        """
        Create config from model family defaults with optional overrides.

        Args:
            model_family: Model family name
            override: Optional dict of values to override defaults

        Returns:
            FeatureSelectionConfig instance
        """
        defaults = ModelFamilyDefaults.get_defaults(model_family)
        defaults["model_family"] = model_family

        if override:
            defaults.update(override)

        return cls(**defaults)

    @classmethod
    def disabled(cls) -> FeatureSelectionConfig:
        """Create a disabled feature selection config."""
        return cls(enabled=False, n_features=0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "n_features": self.n_features,
            "method": self.method,
            "min_feature_frequency": self.min_feature_frequency,
            "n_estimators": self.n_estimators,
            "mda_n_repeats": self.mda_n_repeats,
            "use_clustered_importance": self.use_clustered_importance,
            "max_clusters": self.max_clusters,
            "mtf_max_per_timeframe": self.mtf_max_per_timeframe,
            "regime_conditional": self.regime_conditional,
            "random_state": self.random_state,
            "model_family": self.model_family,
            "selected_features": self._selected_features,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSelectionConfig:
        """Create from dictionary."""
        selected = data.pop("selected_features", [])
        config = cls(**data)
        config._selected_features = selected
        return config


@dataclass
class FeatureSelectorConfig:
    """
    Configuration for walk-forward feature selection.

    This is the simpler configuration class used by WalkForwardFeatureSelector.

    Attributes:
        n_features_to_select: Number of top features to select per fold
        selection_method: Method for computing importance (mda, mdi, hybrid)
        n_estimators: Number of trees in importance estimator
        min_feature_frequency: Minimum fraction of folds feature must appear in
        use_clustered_importance: Whether to use clustered MDA for correlated features
        max_clusters: Maximum number of feature clusters (if using clustered)
    """

    n_features_to_select: int = 50
    selection_method: str = "mda"  # mda, mdi, or hybrid
    n_estimators: int = 50
    mda_n_repeats: int = 5
    min_feature_frequency: float = 0.6
    use_clustered_importance: bool = False
    max_clusters: int = 20

    def __post_init__(self) -> None:
        if self.n_features_to_select <= 0:
            raise ValueError(f"n_features_to_select must be > 0, got {self.n_features_to_select}")
        if self.selection_method not in ("mda", "mdi", "hybrid"):
            raise ValueError(
                f"selection_method must be mda/mdi/hybrid, got {self.selection_method}"
            )
        if self.mda_n_repeats <= 0:
            raise ValueError(f"mda_n_repeats must be > 0, got {self.mda_n_repeats}")
        if not 0 < self.min_feature_frequency <= 1:
            raise ValueError(
                f"min_feature_frequency must be in (0, 1], got {self.min_feature_frequency}"
            )


__all__ = [
    "FeatureSelectionConfig",
    "FeatureSelectorConfig",
    "ModelFamilyDefaults",
]
