"""TrainerConfig dataclass for model training configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .environment import resolve_device


@dataclass
class TrainerConfig:
    """Configuration for model training (hyperparameters + training settings)."""
    model_name: str
    horizon: int = 20
    feature_set: str = "boosting_optimal"
    sequence_length: int = 60
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 15
    random_seed: int = 42
    experiment_name: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
    model_config: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    # Calibration settings
    use_calibration: bool = True
    calibration_method: str = "auto"  # "auto", "isotonic", "sigmoid"
    # Test set evaluation (default True, but marked as one-shot)
    evaluate_test_set: bool = True
    # Feature selection settings
    use_feature_selection: bool = True  # Enable per-model feature selection
    feature_selection_n_features: int = 50  # Number of features to select (0 = auto)
    feature_selection_method: str = "mda"  # "mda", "mdi", "hybrid"
    feature_selection_cv_splits: int = 5  # CV splits for stability analysis

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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerConfig":
        """Create TrainerConfig from dictionary."""
        return cls(**data)

    def get_resolved_device(self) -> str:
        """Get the resolved device (auto -> cuda/cpu)."""
        return resolve_device(self.device)
