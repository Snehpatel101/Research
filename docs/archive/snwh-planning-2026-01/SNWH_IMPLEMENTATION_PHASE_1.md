# SNwH Implementation: Phase 1 - Configuration Layer Extensions

## Overview

Phase 1 extends the configuration system to support per-model configuration. This enables:
1. Each model to declare its preferred timeframe, feature mode, and MTF strategy
2. Heterogeneous ensembles with different base model configurations
3. Automatic adapter routing based on model requirements

---

## 1.1 Extend TrainerConfig

### File: `src/models/config/trainer_config.py`

**Lines 27-102: Add new fields after existing fields**

```python
# ADD these new fields to TrainerConfig dataclass (after line 102)

@dataclass
class TrainerConfig:
    """Configuration for model training (hyperparameters + training settings)."""

    # ... existing fields (lines 31-102) ...

    # NEW: Per-model configuration fields (Phase 1 SNwH)
    primary_timeframe: str = field(
        default_factory=lambda: _get_global_or_default("timeframes.default_primary", "5min")
    )
    mtf_mode: str = field(default="indicators")  # none, indicators, multi_stream
    mtf_timeframes: list[str] = field(default_factory=list)  # Additional TFs for multi_stream
    feature_mode: str = field(default="engineered")  # engineered, raw, hybrid
    adapter_id: str | None = field(default=None)  # tabular, sequence, multi_stream (auto-resolved if None)

    # Contract fields (resolved at runtime from ModelContract)
    input_rank: int = field(default=2)  # 2, 3, or 4
    min_features: int = field(default=4)
    max_features: int = field(default=200)

    def __post_init__(self) -> None:
        """Validate and convert configuration values."""
        # ... existing validation (lines 104-118) ...

        # NEW: Validate mtf_mode
        valid_mtf_modes = {"none", "indicators", "multi_stream"}
        if self.mtf_mode not in valid_mtf_modes:
            raise ValueError(
                f"mtf_mode must be one of {valid_mtf_modes}, got '{self.mtf_mode}'"
            )

        # NEW: Validate feature_mode
        valid_feature_modes = {"engineered", "raw", "hybrid"}
        if self.feature_mode not in valid_feature_modes:
            raise ValueError(
                f"feature_mode must be one of {valid_feature_modes}, got '{self.feature_mode}'"
            )

        # NEW: Validate input_rank
        if self.input_rank not in {2, 3, 4}:
            raise ValueError(f"input_rank must be 2, 3, or 4, got {self.input_rank}")

        # NEW: Auto-resolve adapter_id if not set
        if self.adapter_id is None:
            self.adapter_id = self._resolve_adapter_id()

    def _resolve_adapter_id(self) -> str:
        """Resolve adapter ID from input_rank."""
        if self.input_rank == 2:
            return "tabular"
        elif self.input_rank == 3:
            return "sequence"
        elif self.input_rank == 4:
            return "multi_stream"
        else:
            return "tabular"

    @classmethod
    def from_model_contract(
        cls,
        model_name: str,
        horizon: int,
        **overrides,
    ) -> "TrainerConfig":
        """
        Create TrainerConfig from a ModelContract.

        This is the preferred way to create TrainerConfig for SNwH.

        Args:
            model_name: Name of the model
            horizon: Training horizon
            **overrides: Additional overrides

        Returns:
            TrainerConfig with contract-based defaults
        """
        from src.contracts import get_model_contract

        contract = get_model_contract(model_name)

        return cls(
            model_name=model_name,
            horizon=horizon,
            primary_timeframe=contract.primary_timeframe,
            mtf_mode=contract.mtf_mode.value,
            mtf_timeframes=list(contract.mtf_timeframes),
            feature_mode=contract.feature_mode.value,
            adapter_id=contract.adapter_id,
            input_rank=contract.input_rank.value,
            min_features=contract.min_features,
            max_features=contract.max_features,
            sequence_length=contract.sequence_length,
            **overrides,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            # ... existing fields ...
            "model_name": self.model_name,
            "horizon": self.horizon,
            # ... other existing fields ...

            # NEW fields
            "primary_timeframe": self.primary_timeframe,
            "mtf_mode": self.mtf_mode,
            "mtf_timeframes": self.mtf_timeframes,
            "feature_mode": self.feature_mode,
            "adapter_id": self.adapter_id,
            "input_rank": self.input_rank,
            "min_features": self.min_features,
            "max_features": self.max_features,
        }
        return d
```

---

## 1.2 Add PerModelConfig for Ensembles

### New File: `src/models/config/per_model_config.py`

```python
"""
Per-model configuration for heterogeneous ensembles.

This module defines PerModelConfig which allows each base model
in an ensemble to have its own timeframe, feature mode, and MTF settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.contracts import get_model_contract, ModelContract


@dataclass
class PerModelConfig:
    """
    Per-model configuration for heterogeneous ensembles.

    Each base model in an ensemble can have different:
    - Primary timeframe
    - Feature mode (engineered/raw/hybrid)
    - MTF mode (none/indicators/multi_stream)
    - Feature selection settings

    Example:
        config = PerModelConfig(
            name="xgboost",
            timeframe="15min",
            optimize_features=True,
            feature_opt_trials=30,
        )
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

    def __post_init__(self):
        """Validate and resolve defaults from contract."""
        # Ensure name is lowercase
        self.name = self.name.lower().strip()

    @property
    def contract(self) -> ModelContract:
        """Get the ModelContract for this model."""
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


@dataclass
class EnsemblePlan:
    """
    Plan for training a heterogeneous ensemble.

    This captures all base model configurations and meta-learner settings,
    enabling the trainer to route data correctly.
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

    def __post_init__(self):
        """Validate ensemble configuration."""
        if len(self.base_models) < 2:
            raise ValueError("Ensemble requires at least 2 base models")

    @property
    def is_heterogeneous(self) -> bool:
        """Check if ensemble has models from different families."""
        families = {m.contract.model_family for m in self.base_models}
        return len(families) > 1

    @property
    def required_timeframes(self) -> set[str]:
        """Get all unique timeframes required by base models."""
        tfs = set()
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

    def validate_compatibility(self) -> tuple[bool, list[str]]:
        """
        Validate that ensemble configuration is valid.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check base models exist
        from src.models.registry import ModelRegistry
        for model in self.base_models:
            if not ModelRegistry.is_registered(model.name):
                issues.append(f"Base model '{model.name}' not registered")

        # Check meta-learner exists
        if not ModelRegistry.is_registered(self.meta_learner):
            issues.append(f"Meta-learner '{self.meta_learner}' not registered")

        # For non-stacking ensembles, check homogeneity
        # (voting/blending require same input shape)
        if not self.is_heterogeneous:
            input_ranks = {m.input_rank for m in self.base_models}
            if len(input_ranks) > 1:
                issues.append(
                    f"Non-heterogeneous ensemble requires same input rank, "
                    f"got ranks: {input_ranks}"
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
    ) -> EnsemblePlan:
        """
        Create EnsemblePlan from model names (all defaults from contracts).

        Args:
            base_models: List of base model names
            meta_learner: Meta-learner name

        Returns:
            EnsemblePlan with contract-based defaults
        """
        return cls(
            base_models=[PerModelConfig.from_string(name) for name in base_models],
            meta_learner=meta_learner,
        )


__all__ = [
    "PerModelConfig",
    "EnsemblePlan",
]
```

---

## 1.3 Update ModelDataRequirements

### File: `src/models/config/data_requirements.py`

**Lines 47-83: Add new fields to ModelDataRequirements**

```python
@dataclass(frozen=True)
class ModelDataRequirements:
    """
    Data preparation requirements for a specific model type.

    These requirements inform Phase 1 about what data format
    each model type in Phase 2 will expect.
    """
    # Existing fields...
    model_name: str
    family: ModelFamily
    feature_set: str
    requires_scaling: bool = False
    scaler_type: ScalerType = ScalerType.NONE
    requires_sequences: bool = False
    sequence_length: int = 60
    max_features: int | None = None
    supports_categorical: bool = False
    supports_missing: bool = False
    feature_selection_method: str = "mda"
    feature_selection_n_features: int = 0
    description: str = ""

    # NEW fields (Phase 1 SNwH)
    input_rank: int = 2  # 2D, 3D, or 4D
    feature_mode: str = "engineered"  # engineered, raw, hybrid
    mtf_mode: str = "none"  # none, indicators, multi_stream
    primary_timeframe: str = "5min"  # Default primary TF
    mtf_timeframes: tuple[str, ...] = ()  # Additional TFs for multi_stream
    min_features: int = 4  # Minimum features required

    @property
    def adapter_id(self) -> str:
        """Get adapter ID based on input_rank."""
        if self.input_rank == 2:
            return "tabular"
        elif self.input_rank == 3:
            return "sequence"
        elif self.input_rank == 4:
            return "multi_stream"
        return "tabular"


# UPDATE MODEL_DATA_REQUIREMENTS entries with new fields
# Example for xgboost:
MODEL_DATA_REQUIREMENTS["xgboost"] = ModelDataRequirements(
    model_name="xgboost",
    family=ModelFamily.BOOSTING,
    feature_set="boosting_optimal",
    requires_scaling=False,
    scaler_type=ScalerType.NONE,
    requires_sequences=False,
    max_features=100,
    supports_categorical=True,
    supports_missing=True,
    description="XGBoost gradient boosting",
    # NEW fields
    input_rank=2,
    feature_mode="engineered",
    mtf_mode="indicators",
    primary_timeframe="15min",
    min_features=40,
)
```

---

## 1.4 Extend UnifiedConfig

### File: `src/config/unified.py`

**Add new section for per-model configuration**

```python
# Add after line 439 (after OOMRecoverySection)

@dataclass
class ModelConfigSection:
    """Per-model configuration section."""

    # Default configurations by family
    defaults: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "boosting": {
            "primary_timeframe": "15min",
            "mtf_mode": "indicators",
            "feature_mode": "engineered",
        },
        "neural": {
            "primary_timeframe": "5min",
            "mtf_mode": "indicators",
            "feature_mode": "engineered",
        },
        "transformer": {
            "primary_timeframe": "1min",
            "mtf_mode": "multi_stream",
            "feature_mode": "raw",
        },
        "classical": {
            "primary_timeframe": "15min",
            "mtf_mode": "none",
            "feature_mode": "engineered",
        },
    })

    # Per-model overrides
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_model_config(self, model_name: str, model_family: str) -> dict[str, Any]:
        """
        Get merged configuration for a model.

        Priority: override > family default
        """
        config = dict(self.defaults.get(model_family, {}))
        if model_name in self.overrides:
            config.update(self.overrides[model_name])
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfigSection":
        return cls(
            defaults=data.get("defaults", cls.__dataclass_fields__["defaults"].default_factory()),
            overrides=data.get("overrides", {}),
        )


# Add to UnifiedConfig dataclass (after oom_recovery field)
@dataclass
class UnifiedConfig:
    # ... existing fields ...

    # NEW: Per-model configuration
    model_config: ModelConfigSection = field(default_factory=ModelConfigSection)

    def get_trainer_config_for_model(
        self,
        model_name: str,
        horizon: int | None = None,
        **overrides,
    ) -> Any:
        """
        Get TrainerConfig for a specific model using contract-based defaults.

        This is the SNwH-compatible way to create TrainerConfig.

        Args:
            model_name: Name of the model
            horizon: Training horizon (defaults to max active horizon)
            **overrides: Additional overrides

        Returns:
            TrainerConfig with contract-based defaults
        """
        from src.models.config.trainer_config import TrainerConfig
        from src.contracts import get_model_contract

        horizon = horizon or self.horizons.max_horizon
        contract = get_model_contract(model_name)

        # Get family-level and model-level overrides
        model_overrides = self.model_config.get_model_config(
            model_name, contract.model_family
        )

        return TrainerConfig.from_model_contract(
            model_name=model_name,
            horizon=horizon,
            # Base settings from UnifiedConfig
            batch_size=self.training.batch_size,
            max_epochs=self.training.max_epochs,
            early_stopping_patience=self.training.early_stopping_patience,
            random_seed=self.random_seed,
            output_dir=self.output_dir,
            device=self.training.device,
            mixed_precision=self.training.mixed_precision,
            # Overrides from model_config section
            **model_overrides,
            # User-provided overrides
            **overrides,
        )
```

---

## 1.5 Update Package Exports

### File: `src/models/config/__init__.py`

```python
# Add new exports
from .per_model_config import PerModelConfig, EnsemblePlan

__all__ = [
    # ... existing exports ...
    "PerModelConfig",
    "EnsemblePlan",
]
```

---

## Summary: Phase 1 Changes

| File | Type | Changes |
|------|------|---------|
| `src/models/config/trainer_config.py` | MODIFY | Add 8 new fields, `from_model_contract()` classmethod |
| `src/models/config/per_model_config.py` | NEW | PerModelConfig, EnsemblePlan dataclasses |
| `src/models/config/data_requirements.py` | MODIFY | Add 6 new fields to ModelDataRequirements |
| `src/config/unified.py` | MODIFY | Add ModelConfigSection, `get_trainer_config_for_model()` |
| `src/models/config/__init__.py` | MODIFY | Export new classes |

## Dependencies

- Phase 0 (contracts) must be completed first
- `src/contracts/model_contract.py` provides `get_model_contract()`

## Migration Notes

1. Existing code using `TrainerConfig(model_name=..., horizon=...)` continues to work
2. New code should prefer `TrainerConfig.from_model_contract(...)` for SNwH compatibility
3. Ensemble code should use `EnsemblePlan` for heterogeneous ensembles

## Backward Compatibility

- All new TrainerConfig fields have defaults
- Existing trainer.py code continues to work without modification
- New functionality is opt-in via new methods

## Next Steps

After Phase 1 is implemented, proceed to Phase 2 (Adapter Architecture) which will:
1. Create AdapterRegistry for automatic data routing
2. Implement TabularAdapter, SequenceAdapter, MultiStreamAdapter
3. Wire adapters into trainer.py
