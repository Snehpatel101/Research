"""
Feature Specification - Complete specification for 5-dimension optimization.

Captures everything needed to reproduce a model's training:
- Triple barrier parameters (labeling)
- Selected features
- Feature parameters (e.g., RSI period)
- Feature timeframes
- Model hyperparameters

This enables Optuna to optimize across all 5 dimensions while maintaining
full reproducibility and inference parity.

Phase 3 of the SNwH (Unified Multi-Timeframe Model Factory) implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FeatureSpec:
    """
    Complete specification for model training - all 5 dimensions.

    This dataclass captures everything needed to reproduce a model's training,
    enabling Optuna to optimize across all dimensions while ensuring that
    inference uses the exact same configuration.

    Dimensions:
        1. Triple Barrier Parameters: profit_threshold, loss_threshold, max_holding_bars
        2. Feature Selection: which features are included
        3. Feature Parameters: configuration for each feature (e.g., RSI period)
        4. Feature Timeframes: which timeframe each feature uses
        5. Model Hyperparameters: model-specific tuning parameters

    Example:
        >>> spec = FeatureSpec(
        ...     profit_threshold=0.015,
        ...     loss_threshold=0.010,
        ...     max_holding_bars=120,
        ...     selected_features=["rsi_14", "atr_20", "macd_signal"],
        ...     feature_params={"rsi": {"period": 14}, "atr": {"window": 20}},
        ...     feature_timeframes={"rsi_14": "15min", "atr_20": "5min"},
        ...     hyperparameters={"learning_rate": 0.01, "max_depth": 6},
        ...     model_name="xgboost",
        ... )
        >>> spec.save("artifacts/spec.json")
        >>> loaded = FeatureSpec.load("artifacts/spec.json")
        >>> assert spec.schema_hash == loaded.schema_hash
    """

    # Dimension 1: Triple Barrier Parameters
    profit_threshold: float  # e.g., 0.015 (1.5%)
    loss_threshold: float  # e.g., 0.010 (1.0%)
    max_holding_bars: int  # e.g., 120 bars

    # Dimension 2: Feature Selection
    selected_features: list[str] = field(default_factory=list)
    # e.g., ["rsi_14", "atr_20", "macd_signal"]

    # Dimension 3: Feature Parameters
    feature_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # e.g., {"rsi": {"period": 14}, "atr": {"window": 20}}

    # Dimension 4: Feature Timeframes
    feature_timeframes: dict[str, str] = field(default_factory=dict)
    # e.g., {"rsi_14": "15min", "atr_20": "5min"}

    # Dimension 5: Model Hyperparameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    # e.g., {"learning_rate": 0.01, "max_depth": 6}

    # Metadata
    model_name: str = ""
    optuna_trial_id: int = -1
    optuna_study_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Validation metrics from the trial (optional)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    # e.g., {"sharpe_ratio": 1.5, "accuracy": 0.62, "profit_factor": 1.8}

    @property
    def schema_hash(self) -> str:
        """
        Generate hash of the spec for versioning/comparison.

        The hash is deterministic and only includes the 5 dimensions
        plus model name - not metadata like timestamps or trial IDs.

        Returns:
            16-character hex string hash
        """
        spec_dict = {
            "profit_threshold": self.profit_threshold,
            "loss_threshold": self.loss_threshold,
            "max_holding_bars": self.max_holding_bars,
            "selected_features": sorted(self.selected_features),
            "feature_params": self._sort_nested_dict(self.feature_params),
            "feature_timeframes": self._sort_nested_dict(self.feature_timeframes),
            "hyperparameters": self._sort_nested_dict(self.hyperparameters),
            "model_name": self.model_name,
        }
        json_str = json.dumps(spec_dict, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    @staticmethod
    def _sort_nested_dict(d: dict[str, Any]) -> dict[str, Any]:
        """Recursively sort dictionary keys for deterministic hashing."""
        result = {}
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                result[key] = FeatureSpec._sort_nested_dict(value)
            elif isinstance(value, list):
                result[key] = sorted(value) if all(isinstance(x, str) for x in value) else value
            else:
                result[key] = value
        return result

    @property
    def n_features(self) -> int:
        """Return the number of selected features."""
        return len(self.selected_features)

    @property
    def has_mtf_features(self) -> bool:
        """Check if spec uses multiple timeframes."""
        timeframes = set(self.feature_timeframes.values())
        return len(timeframes) > 1

    @property
    def unique_timeframes(self) -> list[str]:
        """Get list of unique timeframes used."""
        return sorted(set(self.feature_timeframes.values()))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the spec
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpec:
        """
        Create from dictionary.

        Args:
            data: Dictionary with FeatureSpec fields

        Returns:
            FeatureSpec instance
        """
        # Filter to only known fields to handle forward compatibility
        known_fields = {
            "profit_threshold",
            "loss_threshold",
            "max_holding_bars",
            "selected_features",
            "feature_params",
            "feature_timeframes",
            "hyperparameters",
            "model_name",
            "optuna_trial_id",
            "optuna_study_name",
            "created_at",
            "validation_metrics",
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def save(self, path: str | Path) -> None:
        """
        Save to JSON file.

        Args:
            path: Path to save the JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> FeatureSpec:
        """
        Load from JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            FeatureSpec instance
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the spec for completeness and consistency.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check triple barrier parameters
        if self.profit_threshold <= 0:
            issues.append("profit_threshold must be positive")
        if self.loss_threshold <= 0:
            issues.append("loss_threshold must be positive")
        if self.max_holding_bars <= 0:
            issues.append("max_holding_bars must be positive")

        # Check feature consistency
        if not self.selected_features:
            issues.append("selected_features is empty")

        # Check that all features have timeframes if any do
        if self.feature_timeframes:
            missing_timeframes = set(self.selected_features) - set(self.feature_timeframes.keys())
            if missing_timeframes:
                issues.append(f"Features missing timeframes: {missing_timeframes}")

        # Check model name
        if not self.model_name:
            issues.append("model_name is required")

        return len(issues) == 0, issues

    def with_trial_info(
        self,
        trial_id: int,
        study_name: str = "",
        metrics: dict[str, float] | None = None,
    ) -> FeatureSpec:
        """
        Create a copy with Optuna trial information added.

        Args:
            trial_id: Optuna trial ID
            study_name: Optuna study name
            metrics: Validation metrics from the trial

        Returns:
            New FeatureSpec with trial info
        """
        data = self.to_dict()
        data["optuna_trial_id"] = trial_id
        if study_name:
            data["optuna_study_name"] = study_name
        if metrics:
            data["validation_metrics"] = metrics
        return FeatureSpec.from_dict(data)

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"FeatureSpec(model={self.model_name}, "
            f"features={self.n_features}, "
            f"barriers=({self.profit_threshold:.3f}/{self.loss_threshold:.3f}/{self.max_holding_bars}), "
            f"hash={self.schema_hash})"
        )


__all__ = [
    "FeatureSpec",
]
