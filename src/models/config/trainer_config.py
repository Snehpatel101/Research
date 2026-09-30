"""TrainerConfig dataclass for model training configuration."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from .environment import resolve_device

logger = logging.getLogger(__name__)

# Settings of the Trainer-level feature selection, removed in Phase 117 (MLFactory
# selects every model's features on train-only data before its Trainer runs).
# Saved configs that still carry them load with a warning.
REMOVED_FIELDS: frozenset[str] = frozenset(
    {
        "use_feature_selection",
        "feature_selection_n_features",
        "feature_selection_method",
        "feature_selection_cv_splits",
        "feature_selection_min_frequency",
        "feature_selection_purge_bars",
        "feature_selection_embargo_bars",
    }
)


def _get_global_or_default(attr_path: str, fallback: Any) -> Any:
    """Thin wrapper over the canonical config lookup (src/config/utils.py)."""
    from src.config.utils import get_config_value

    return get_config_value(attr_path, fallback)


@dataclass
class TrainerConfig:
    """Configuration for model training (hyperparameters + training settings)."""

    model_name: str
    horizon: int = 20
    # Named feature set filter; "" = pick by model alias, None = use every input column
    feature_set: str | None = "boosting_optimal"
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
    # None = auto-detect per device in the DataLoader (4 workers + pinned memory
    # on CUDA, 0 workers on CPU where forked workers duplicate the parent)
    num_workers: int | None = field(
        default_factory=lambda: _get_global_or_default("training.num_workers", None)
    )
    pin_memory: bool | None = field(
        default_factory=lambda: _get_global_or_default("training.pin_memory", None)
    )
    use_calibration: bool = field(
        default_factory=lambda: _get_global_or_default("calibration.enabled", True)
    )
    calibration_method: str = field(
        default_factory=lambda: _get_global_or_default("calibration.method", "auto")
    )
    evaluate_test_set: bool = True
    deterministic_mode: bool = False
    nan_check_raise_error: bool = True
    checkpoint_interval: int = 50
    keep_n_checkpoints: int = 3
    checkpoint_dir: str | None = None
    # Experiment tracking (ExperimentConfig.tracking via PipelineConfig):
    # "none" | "local" | "mlflow"; each Trainer run is one tracker run, a child
    # of tracking_parent_run_id when set (MLFactory's run). experiment_name
    # names the tracker experiment.
    tracking_backend: str = "none"
    tracking_uri: str | None = None
    tracking_parent_run_id: str | None = None
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
        """
        Convert to dictionary for serialization.

        Field-driven (iterates the dataclass fields) so it can never drift
        out of sync when fields are added — the previous hand-maintained
        key list silently dropped pipeline_run_id.
        """
        from dataclasses import fields as dataclass_fields

        result: dict[str, Any] = {}
        for f in dataclass_fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)
            result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerConfig":
        """Create TrainerConfig from dictionary.

        Keys of removed settings (``REMOVED_FIELDS``) are dropped with a
        warning, so configs saved by older versions still load.
        """
        removed = sorted(REMOVED_FIELDS.intersection(data))
        if removed:
            logger.warning(
                f"TrainerConfig: ignoring removed settings {removed} (Trainer-level "
                "feature selection was removed; MLFactory selects features per model "
                "on train-only data)"
            )
        return cls(**{k: v for k, v in data.items() if k not in REMOVED_FIELDS})

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
