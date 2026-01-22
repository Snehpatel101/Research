"""
Per-model configuration for heterogeneous ensembles.

Phase 1 SNwH Implementation.

This module defines PerModelConfig which allows each base model
in an ensemble to have its own timeframe, feature mode, and MTF settings.
It also defines EnsemblePlan for planning heterogeneous ensemble training.

Example:
    # Simple usage with contract defaults
    config = PerModelConfig(name="xgboost")
    print(config.resolved_timeframe)  # "15min" (from contract)

    # Explicit overrides
    config = PerModelConfig(
        name="lstm",
        timeframe="10min",  # Override contract default
        sequence_length=120,
    )

    # Create ensemble plan
    plan = EnsemblePlan.from_model_names(
        base_models=["xgboost", "lstm", "patchtst"],
        meta_learner="ridge_meta",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.contracts import ModelContract


@dataclass
class PerModelConfig:
    """
    Per-model configuration for heterogeneous ensembles.

    Each base model in an ensemble can have different:
    - Primary timeframe
    - Feature mode (engineered/raw/hybrid)
    - MTF mode (none/indicators/multi_stream)
    - Sequence length
    - Model-specific hyperparameters

    Attributes:
        name: Model name (e.g., "xgboost", "lstm")
        timeframe: Primary timeframe override (None = use contract default)
        feature_mode: Feature mode override (None = use contract default)
        optimize_features: Whether to run feature optimization
        feature_opt_trials: Number of trials for feature optimization
        mtf_mode: MTF mode override (None = use contract default)
        mtf_timeframes: MTF timeframes override (None = use contract default)
        sequence_length: Sequence length override (None = use contract default)
        hyperparameters: Model-specific hyperparameter overrides
    """

    # Model identity
    name: str

    # Timeframe (defaults from ModelContract)
    timeframe: str | None = None  # None = use contract default

    # Feature configuration
    feature_mode: str | None = None  # None = use contract default
    optimize_features: bool = False
    feature_opt_trials: int = 30

    # MTF configuration
    mtf_mode: str | None = None  # None = use contract default
    mtf_timeframes: list[str] | None = None

    # Sequence configuration
    sequence_length: int | None = None  # None = use contract default

    # Model-specific hyperparameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Ensure name is lowercase
        self.name = self.name.lower().strip()

    @property
    def contract(self) -> ModelContract:
        """Get the ModelContract for this model."""
        from src.core.contracts import get_model_contract

        return get_model_contract(self.name)

    @property
    def resolved_timeframe(self) -> str:
        """Get the resolved timeframe (explicit or contract default)."""
        return self.timeframe or self.contract.primary_timeframe

    @property
    def resolved_feature_mode(self) -> str:
        """Get the resolved feature mode (explicit or contract default)."""
        return self.feature_mode or self.contract.feature_mode.value

    @property
    def resolved_mtf_mode(self) -> str:
        """Get the resolved MTF mode (explicit or contract default)."""
        return self.mtf_mode or self.contract.mtf_mode.value

    @property
    def resolved_mtf_timeframes(self) -> list[str]:
        """Get the resolved MTF timeframes (explicit or contract default)."""
        if self.mtf_timeframes is not None:
            return self.mtf_timeframes
        return list(self.contract.mtf_timeframes)

    @property
    def resolved_sequence_length(self) -> int:
        """Get the resolved sequence length (explicit or contract default)."""
        return self.sequence_length or self.contract.sequence_length

    @property
    def input_rank(self) -> int:
        """Get the input rank from contract."""
        return self.contract.input_rank.value

    @property
    def adapter_id(self) -> str:
        """Get the adapter ID from contract."""
        return self.contract.adapter_id

    @property
    def model_family(self) -> str:
        """Get the model family from contract."""
        return self.contract.model_family

    def to_trainer_config_kwargs(self) -> dict[str, Any]:
        """
        Get kwargs to pass to TrainerConfig.

        Returns:
            Dict of kwargs for TrainerConfig
        """
        return {
            "model_name": self.name,
            "primary_timeframe": self.resolved_timeframe,
            "feature_mode": self.resolved_feature_mode,
            "mtf_mode": self.resolved_mtf_mode,
            "mtf_timeframes": self.resolved_mtf_timeframes,
            "sequence_length": self.resolved_sequence_length,
            "input_rank": self.input_rank,
            "adapter_id": self.adapter_id,
            "model_config": self.hyperparameters,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "timeframe": self.timeframe,
            "feature_mode": self.feature_mode,
            "optimize_features": self.optimize_features,
            "feature_opt_trials": self.feature_opt_trials,
            "mtf_mode": self.mtf_mode,
            "mtf_timeframes": self.mtf_timeframes,
            "sequence_length": self.sequence_length,
            "hyperparameters": self.hyperparameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerModelConfig:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            timeframe=data.get("timeframe"),
            feature_mode=data.get("feature_mode"),
            optimize_features=data.get("optimize_features", False),
            feature_opt_trials=data.get("feature_opt_trials", 30),
            mtf_mode=data.get("mtf_mode"),
            mtf_timeframes=data.get("mtf_timeframes"),
            sequence_length=data.get("sequence_length"),
            hyperparameters=data.get("hyperparameters", {}),
        )

    @classmethod
    def from_string(cls, model_name: str) -> PerModelConfig:
        """Create from just a model name (all defaults from contract)."""
        return cls(name=model_name)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PerModelConfig(name='{self.name}', "
            f"timeframe='{self.resolved_timeframe}', "
            f"feature_mode='{self.resolved_feature_mode}', "
            f"input_rank={self.input_rank})"
        )


@dataclass
class EnsemblePlan:
    """
    Plan for training a heterogeneous ensemble.

    This captures all base model configurations and meta-learner settings,
    enabling the trainer to route data correctly for each model.

    Attributes:
        base_models: List of PerModelConfig for base models
        meta_learner: Name of the meta-learner model
        meta_learner_config: Configuration for the meta-learner
        stacking_method: "soft" (probabilities) or "hard" (predictions)
        passthrough_features: Whether to include original features in meta-learner
        n_folds: Number of CV folds for OOF generation
        purge_bars: Number of bars to purge between folds
        embargo_bars: Number of bars to embargo after test fold

    Example:
        # Create plan from model names
        plan = EnsemblePlan.from_model_names(
            base_models=["xgboost", "lstm", "patchtst"],
            meta_learner="ridge_meta",
        )

        # Check if heterogeneous
        print(plan.is_heterogeneous)  # True (mixed families)

        # Get required timeframes
        print(plan.required_timeframes)  # {"15min", "5min", "1min"}
    """

    # Base models
    base_models: list[PerModelConfig]

    # Meta-learner
    meta_learner: str = "ridge_meta"
    meta_learner_config: dict[str, Any] = field(default_factory=dict)

    # Ensemble settings
    stacking_method: str = "soft"  # soft (probabilities) or hard (predictions)
    passthrough_features: bool = False  # Include original features in meta-learner

    # OOF settings
    n_folds: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440

    def __post_init__(self) -> None:
        """Validate ensemble configuration."""
        if len(self.base_models) < 2:
            raise ValueError("Ensemble requires at least 2 base models")

        valid_methods = {"soft", "hard"}
        if self.stacking_method not in valid_methods:
            raise ValueError(
                f"stacking_method must be one of {valid_methods}, " f"got '{self.stacking_method}'"
            )

    @property
    def is_heterogeneous(self) -> bool:
        """Check if ensemble has models from different families."""
        families = {m.model_family for m in self.base_models}
        return len(families) > 1

    @property
    def required_timeframes(self) -> set[str]:
        """Get all unique timeframes required by base models."""
        tfs: set[str] = set()
        for model in self.base_models:
            tfs.add(model.resolved_timeframe)
            tfs.update(model.resolved_mtf_timeframes)
        return tfs

    @property
    def has_tabular_models(self) -> bool:
        """Check if any base model requires tabular (2D) input."""
        return any(m.input_rank == 2 for m in self.base_models)

    @property
    def has_sequence_models(self) -> bool:
        """Check if any base model requires sequence (3D) input."""
        return any(m.input_rank == 3 for m in self.base_models)

    @property
    def has_multi_tf_models(self) -> bool:
        """Check if any base model requires multi-TF (4D) input."""
        return any(m.input_rank == 4 for m in self.base_models)

    @property
    def base_model_names(self) -> list[str]:
        """Get list of base model names."""
        return [m.name for m in self.base_models]

    def get_models_by_adapter(self) -> dict[str, list[PerModelConfig]]:
        """
        Group base models by adapter type.

        Returns:
            Dict mapping adapter_id to list of PerModelConfig
        """
        groups: dict[str, list[PerModelConfig]] = {
            "tabular": [],
            "sequence": [],
            "multi_stream": [],
        }
        for model in self.base_models:
            groups[model.adapter_id].append(model)
        return {k: v for k, v in groups.items() if v}

    def get_models_by_family(self) -> dict[str, list[PerModelConfig]]:
        """
        Group base models by model family.

        Returns:
            Dict mapping family name to list of PerModelConfig
        """
        groups: dict[str, list[PerModelConfig]] = {}
        for model in self.base_models:
            family = model.model_family
            if family not in groups:
                groups[family] = []
            groups[family].append(model)
        return groups

    def validate_compatibility(self) -> tuple[bool, list[str]]:
        """
        Validate that ensemble configuration is valid.

        Returns:
            (is_valid, list_of_issues)
        """
        issues: list[str] = []

        # Check base models exist in registry
        from src.models.registry import ModelRegistry

        for model in self.base_models:
            if not ModelRegistry.is_registered(model.name):
                issues.append(f"Base model '{model.name}' not registered")

        # Check meta-learner exists
        if not ModelRegistry.is_registered(self.meta_learner):
            issues.append(f"Meta-learner '{self.meta_learner}' not registered")

        # Check meta-learner is actually a meta-learner family
        try:
            from src.core.contracts import get_model_contract

            meta_contract = get_model_contract(self.meta_learner)
            if meta_contract.model_family != "meta_learner":
                issues.append(
                    f"Meta-learner '{self.meta_learner}' is not a meta_learner family model "
                    f"(got '{meta_contract.model_family}')"
                )
        except ValueError:
            pass  # Already caught above

        # For voting/blending, warn about heterogeneous inputs
        if self.stacking_method == "hard" and self.is_heterogeneous:
            issues.append(
                "Hard voting with heterogeneous models may have alignment issues. "
                "Consider using soft voting (stacking_method='soft')."
            )

        return len(issues) == 0, issues

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "base_models": [m.to_dict() for m in self.base_models],
            "meta_learner": self.meta_learner,
            "meta_learner_config": self.meta_learner_config,
            "stacking_method": self.stacking_method,
            "passthrough_features": self.passthrough_features,
            "n_folds": self.n_folds,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnsemblePlan:
        """Deserialize from dictionary."""
        return cls(
            base_models=[PerModelConfig.from_dict(m) for m in data["base_models"]],
            meta_learner=data.get("meta_learner", "ridge_meta"),
            meta_learner_config=data.get("meta_learner_config", {}),
            stacking_method=data.get("stacking_method", "soft"),
            passthrough_features=data.get("passthrough_features", False),
            n_folds=data.get("n_folds", 5),
            purge_bars=data.get("purge_bars", 60),
            embargo_bars=data.get("embargo_bars", 1440),
        )

    @classmethod
    def from_model_names(
        cls,
        base_models: list[str],
        meta_learner: str = "ridge_meta",
        **kwargs: Any,
    ) -> EnsemblePlan:
        """
        Create EnsemblePlan from model names (all defaults from contracts).

        Args:
            base_models: List of base model names
            meta_learner: Meta-learner name
            **kwargs: Additional EnsemblePlan parameters

        Returns:
            EnsemblePlan with contract-based defaults
        """
        return cls(
            base_models=[PerModelConfig.from_string(name) for name in base_models],
            meta_learner=meta_learner,
            **kwargs,
        )

    def __repr__(self) -> str:
        """String representation."""
        model_names = ", ".join(self.base_model_names)
        return (
            f"EnsemblePlan(base_models=[{model_names}], "
            f"meta_learner='{self.meta_learner}', "
            f"is_heterogeneous={self.is_heterogeneous})"
        )


__all__ = [
    "PerModelConfig",
    "EnsemblePlan",
]
