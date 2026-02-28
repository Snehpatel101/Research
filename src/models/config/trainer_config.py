"""TrainerConfig dataclass for model training configuration."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from .environment import resolve_device

logger = logging.getLogger(__name__)


def _get_global_or_default(attr_path: str, fallback: Any) -> Any:
    try:
        from src.config.global_config import get_global_config

        config = get_global_config()
        parts = attr_path.split(".")
        value = config
        for part in parts:
            value = getattr(value, part)
        return value
    except Exception as e:
        logger.warning(
            f"Failed to get config attribute '{attr_path}': {e}. Using fallback: {fallback}"
        )
        return fallback


@dataclass
class TrainerConfig:
    """Configuration for model training (hyperparameters + training settings)."""

    model_name: str
    horizon: int = 20
    feature_set: str = "boosting_optimal"
    pipeline_run_id: str | None = None
    sequence_length: int = field(
        default_factory=lambda: _get_global_or_default("training.sequence_length", 60)
    )
    batch_size: int = field(
        default_factory=lambda: _get_global_or_default("training.batch_size", 512)
    )
    max_epochs: int = field(
        default_factory=lambda: _get_global_or_default("training.max_epochs", 100)
    )
    early_stopping_patience: int = field(
        default_factory=lambda: _get_global_or_default("training.early_stopping_patience", 15)
    )
    random_seed: int = field(default_factory=lambda: _get_global_or_default("random_seed", 42))
    experiment_name: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
    model_config: dict[str, Any] = field(default_factory=dict)
    device: str = field(default_factory=lambda: _get_global_or_default("training.device", "auto"))
    mixed_precision: bool = field(
        default_factory=lambda: _get_global_or_default("training.mixed_precision", True)
    )
    num_workers: int = field(
        default_factory=lambda: _get_global_or_default("training.num_workers", 4)
    )
    pin_memory: bool = field(
        default_factory=lambda: _get_global_or_default("training.pin_memory", True)
    )
    use_calibration: bool = field(
        default_factory=lambda: _get_global_or_default("calibration.enabled", True)
    )
    calibration_method: str = field(
        default_factory=lambda: _get_global_or_default("calibration.method", "auto")
    )
    evaluate_test_set: bool = True
    use_feature_selection: bool = field(
        default_factory=lambda: _get_global_or_default("features.selection.enabled", True)
    )
    feature_selection_n_features: int = 50
    feature_selection_method: str = field(
        default_factory=lambda: _get_global_or_default("features.selection.method", "mda")
    )
    feature_selection_cv_splits: int = field(
        default_factory=lambda: _get_global_or_default("features.selection.cv_splits", 5)
    )
    feature_selection_min_frequency: float = 0.6
    deterministic_mode: bool = False
    nan_check_raise_error: bool = True
    checkpoint_interval: int = 10
    keep_n_checkpoints: int = 3
    checkpoint_dir: str | None = None
    tracking_enabled: bool = field(
        default_factory=lambda: _get_global_or_default("tracking.enabled", True)
    )
    tracking_backend: str = field(
        default_factory=lambda: _get_global_or_default("tracking.backend", "local")
    )
    tracking_uri: str | None = None
    tracking_tags: dict[str, str] = field(default_factory=dict)
    oom_recovery_enabled: bool = field(
        default_factory=lambda: _get_global_or_default("oom_recovery.enabled", True)
    )
    oom_max_retries: int = field(
        default_factory=lambda: _get_global_or_default("oom_recovery.max_retries", 3)
    )
    oom_batch_reduction_factor: float = field(
        default_factory=lambda: _get_global_or_default("oom_recovery.batch_reduction_factor", 0.5)
    )
    oom_min_batch_size: int = field(
        default_factory=lambda: _get_global_or_default("oom_recovery.min_batch_size", 8)
    )

    # =========================================================================
    # Phase 1 SNwH: Per-model configuration fields
    # =========================================================================
    primary_timeframe: str = field(
        default_factory=lambda: _get_global_or_default("timeframes.default_primary", "5min")
    )
    mtf_mode: str = field(default="indicators")  # none, indicators, multi_stream
    mtf_timeframes: list[str] = field(default_factory=list)  # Additional TFs for multi_stream
    feature_mode: str = field(default="engineered")  # engineered, raw, hybrid
    adapter_id: str | None = field(default=None)  # tabular, sequence, multi_stream (auto-resolved)

    # Contract fields (resolved at runtime from ModelContract)
    input_rank: int = field(default=2)  # 2, 3, or 4
    min_features: int = field(default=4)
    max_features: int = field(default=200)

    # =========================================================================
    # Phase 7: Pre-training validation configuration
    # =========================================================================
    check_leakage: bool = True  # Run leakage detection before training
    check_lookahead: bool = True  # Run lookahead audit before training
    validation_correlation_threshold: float = 0.5  # Threshold for leakage detection
    project_root: Path | None = None  # Project root for lineage validation

    def __post_init__(self) -> None:
        """Validate and convert configuration values."""
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be non-negative, "
                f"got {self.early_stopping_patience}"
            )
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Phase 1 SNwH: Validate new fields
        valid_mtf_modes = {"none", "indicators", "multi_stream"}
        if self.mtf_mode not in valid_mtf_modes:
            raise ValueError(f"mtf_mode must be one of {valid_mtf_modes}, got '{self.mtf_mode}'")

        valid_feature_modes = {"engineered", "raw", "hybrid", "oof_probs"}
        if self.feature_mode not in valid_feature_modes:
            raise ValueError(
                f"feature_mode must be one of {valid_feature_modes}, got '{self.feature_mode}'"
            )

        if self.input_rank not in {2, 3, 4}:
            raise ValueError(f"input_rank must be 2, 3, or 4, got {self.input_rank}")

        # Auto-resolve adapter_id if not set
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "feature_set": self.feature_set,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "random_seed": self.random_seed,
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "model_config": self.model_config,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "use_calibration": self.use_calibration,
            "calibration_method": self.calibration_method,
            "evaluate_test_set": self.evaluate_test_set,
            "use_feature_selection": self.use_feature_selection,
            "feature_selection_n_features": self.feature_selection_n_features,
            "feature_selection_method": self.feature_selection_method,
            "feature_selection_cv_splits": self.feature_selection_cv_splits,
            "deterministic_mode": self.deterministic_mode,
            "nan_check_raise_error": self.nan_check_raise_error,
            "checkpoint_interval": self.checkpoint_interval,
            "keep_n_checkpoints": self.keep_n_checkpoints,
            "checkpoint_dir": self.checkpoint_dir,
            "tracking_enabled": self.tracking_enabled,
            "tracking_backend": self.tracking_backend,
            "tracking_uri": self.tracking_uri,
            "tracking_tags": self.tracking_tags,
            "oom_recovery_enabled": self.oom_recovery_enabled,
            "oom_max_retries": self.oom_max_retries,
            "oom_batch_reduction_factor": self.oom_batch_reduction_factor,
            "oom_min_batch_size": self.oom_min_batch_size,
            # Phase 1 SNwH fields
            "primary_timeframe": self.primary_timeframe,
            "mtf_mode": self.mtf_mode,
            "mtf_timeframes": self.mtf_timeframes,
            "feature_mode": self.feature_mode,
            "adapter_id": self.adapter_id,
            "input_rank": self.input_rank,
            "min_features": self.min_features,
            "max_features": self.max_features,
            # Phase 7: validation fields
            "check_leakage": self.check_leakage,
            "check_lookahead": self.check_lookahead,
            "validation_correlation_threshold": self.validation_correlation_threshold,
            "project_root": str(self.project_root) if self.project_root else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerConfig":
        """Create TrainerConfig from dictionary."""
        return cls(**data)

    @classmethod
    def from_model_contract(
        cls,
        model_name: str,
        horizon: int,
        **overrides: Any,
    ) -> "TrainerConfig":
        """
        Create TrainerConfig from a ModelContract.

        This is the preferred way to create TrainerConfig for SNwH-compatible
        configuration. It automatically populates timeframe, MTF mode, and
        other settings from the model's contract.

        Args:
            model_name: Name of the model (e.g., "xgboost", "lstm", "patchtst")
            horizon: Training horizon (e.g., 20)
            **overrides: Additional overrides for any TrainerConfig field

        Returns:
            TrainerConfig with contract-based defaults

        Example:
            # Create config for LSTM with contract defaults
            config = TrainerConfig.from_model_contract("lstm", horizon=20)

            # Override specific settings
            config = TrainerConfig.from_model_contract(
                "xgboost",
                horizon=20,
                batch_size=512,
                max_epochs=200,
            )
        """
        from src.core.contracts import get_model_contract

        contract = get_model_contract(model_name)

        # Build kwargs from contract (normalize model_name to lowercase)
        contract_kwargs: dict[str, Any] = {
            "model_name": model_name.lower().strip(),
            "horizon": horizon,
            "primary_timeframe": contract.primary_timeframe,
            "mtf_mode": contract.mtf_mode.value,
            "mtf_timeframes": list(contract.mtf_timeframes),
            "feature_mode": contract.feature_mode.value,
            "adapter_id": contract.adapter_id,
            "input_rank": contract.input_rank.value,
            "min_features": contract.min_features,
            "max_features": contract.max_features,
            "sequence_length": contract.sequence_length,
        }

        # Apply overrides (user values take precedence)
        contract_kwargs.update(overrides)

        return cls(**contract_kwargs)

    def get_resolved_device(self) -> str:
        """Get the resolved device (auto -> cuda/cpu)."""
        return resolve_device(self.device)

    # =========================================================================
    # Phase 5 SNwH: Feature Strategy Integration
    # =========================================================================

    def get_feature_strategy(self) -> Any:
        """
        Get the feature strategy for this model.

        Returns:
            ModelFeatureStrategy for the configured model
        """
        from src.data.features.strategies import get_strategy_for_model

        return get_strategy_for_model(self.model_name)

    def get_baseline_features(self) -> list[str]:
        """
        Get baseline feature list from strategy.

        Returns:
            List of baseline feature names
        """
        strategy = self.get_feature_strategy()
        return list(strategy.baseline_features)

    def resolve_features(
        self,
        available_features: list[str],
        strict: bool = True,
    ) -> list[str]:
        """
        Resolve features based on strategy and availability.

        Args:
            available_features: Features available in the data
            strict: Raise error if insufficient features

        Returns:
            List of resolved feature names
        """
        from src.data.features.strategy_manager import FeatureStrategyManager

        manager = FeatureStrategyManager(available_features=available_features)
        resolved = manager.get_features_for_model(self.model_name, strict=strict)

        # Store resolution result for logging
        self._resolved_features = resolved

        return resolved.feature_columns
