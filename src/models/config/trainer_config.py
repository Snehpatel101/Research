"""TrainerConfig dataclass for model training configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.global_config import GlobalConfig

from .environment import resolve_device


def _get_global_or_default(attr_path: str, fallback: Any) -> Any:
    try:
        from src.config.global_config import get_global_config

        config = get_global_config()
        parts = attr_path.split(".")
        value = config
        for part in parts:
            value = getattr(value, part)
        return value
    except Exception:
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
        default_factory=lambda: _get_global_or_default("training.batch_size", 256)
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerConfig":
        """Create TrainerConfig from dictionary."""
        return cls(**data)

    def get_resolved_device(self) -> str:
        """Get the resolved device (auto -> cuda/cpu)."""
        return resolve_device(self.device)
