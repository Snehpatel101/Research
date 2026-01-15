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
    # NOTE (CFG-002b): TrainerConfig.feature_set controls feature SELECTION during training,
    # which is different from PipelineConfig.feature_set that controls feature GENERATION.
    #
    # - PipelineConfig.feature_set="full" (CFG-002a) generates ALL ~180 features in Phase 1.
    #   This ensures the complete feature superset is available for any model type.
    #
    # - TrainerConfig.feature_set="boosting_optimal" selects a model-appropriate subset
    #   during Phase 6 training. Each model family gets tailored features:
    #     * "boosting_optimal" (~50 features): For XGBoost, LightGBM, CatBoost
    #     * "neural_optimal" (~43 features): For LSTM, GRU, TCN
    #     * "transformer_raw" (~23 features): For Transformer, PatchTST
    #     * "full": Use all available features (no selection)
    #
    # This intentional separation allows per-model feature selection without
    # re-running the data pipeline. See src/phase1/config/feature_sets.py for definitions.
    feature_set: str = "boosting_optimal"  # Controls feature SELECTION, not generation
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
    # Reproducibility settings
    deterministic_mode: bool = False  # Enable deterministic CUDA operations (slower)
    # Numerical stability settings
    nan_check_raise_error: bool = True  # Raise error on NaN/Inf during training
    # Checkpoint settings
    checkpoint_interval: int = 10  # Save checkpoint every N epochs
    keep_n_checkpoints: int = 3  # Number of checkpoints to keep
    checkpoint_dir: str | None = None  # Directory for checkpoints (None = disabled)
    # Experiment tracking settings
    tracking_enabled: bool = True  # Enable experiment tracking
    tracking_backend: str = "local"  # "local", "mlflow", "disabled"
    tracking_uri: str | None = None  # MLflow tracking URI (if using MLflow backend)
    tracking_tags: dict[str, str] = field(default_factory=dict)  # Additional run tags
    # OOM recovery settings
    oom_recovery_enabled: bool = True  # Enable OOM recovery
    oom_max_retries: int = 3  # Max retry attempts on OOM
    oom_batch_reduction_factor: float = 0.5  # Reduce batch by this factor on OOM
    oom_min_batch_size: int = 8  # Minimum batch size after reduction

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
