"""
Consolidated Training Configuration Classes.

This module contains all training-related configuration classes:
- TrainerConfig: Core training settings (re-exported from models)
- OptunaConfig: Hyperparameter optimization settings
- CalibrationConfig: Probability calibration settings
- CheckpointConfig: Model checkpointing settings
- OOMConfig: Out-of-memory recovery settings
- ParallelTrainingConfig: Parallel training settings

These are the CANONICAL locations for these configs.
Import from here:
    from src.config.training import TrainerConfig, OptunaConfig, CalibrationConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config.base import BaseConfig

# =============================================================================
# NOTE: TrainerConfig is NOT imported here to avoid circular imports.
# Import it from src.config directly:
#     from src.config import TrainerConfig
# Or from src.config.models:
#     from src.config.models import TrainerConfig
# =============================================================================


# =============================================================================
# OPTUNA CONFIGURATION
# =============================================================================


@dataclass
class OptunaConfig(BaseConfig):
    """
    Configuration for Optuna hyperparameter optimization.

    This is the CANONICAL OptunaConfig. Use this instead of:
    - OptunaConfig in src/config/global_config.py (deprecated)

    Attributes:
        n_trials: Number of optimization trials
        timeout: Timeout in seconds (0 = no timeout)
        n_jobs: Number of parallel jobs (-1 = all cores)
        study_name: Optional study name for persistence
        storage: Optional database storage URL
        sampler: Sampler type ('tpe', 'random', 'cmaes')
        pruner: Pruner type ('median', 'hyperband', 'none')
        random_state: Random seed for reproducibility

    Example:
        config = OptunaConfig(
            n_trials=100,
            timeout=3600,
            sampler="tpe",
        )
    """

    n_trials: int = 100
    timeout: int = 43200  # 12 hours default (prevents runaway optimization)
    n_startup_trials: int = 10  # Random trials before TPE sampler kicks in
    n_jobs: int = 1
    study_name: str | None = None
    storage: str | None = None
    sampler: str = "tpe"
    pruner: str = "median"
    random_state: int = 42

    # Pruning settings
    pruning_warmup_steps: int = 5
    pruning_interval_steps: int = 1

    # Direction
    direction: str = "maximize"  # maximize or minimize
    metric: str = "f1_weighted"

    # Variance penalty: penalizes cross-fold score variance during tuning.
    # Final objective = mean_score - (variance_penalty * std_score).
    # Higher values favor more stable models across folds.
    variance_penalty: float = 0.1

    # Maximum samples for Optuna subsampling (caps large datasets to control memory).
    # Datasets larger than this are stratified-subsampled before tuning.
    max_samples: int = 50_000

    def validate(self) -> list[str]:
        """Validate Optuna configuration."""
        issues = super().validate()

        if self.n_trials <= 0:
            issues.append(f"n_trials must be positive, got {self.n_trials}")

        if self.timeout < 0:
            issues.append(f"timeout must be non-negative, got {self.timeout}")

        valid_samplers = ["tpe", "random", "cmaes", "grid"]
        if self.sampler not in valid_samplers:
            issues.append(f"sampler must be one of {valid_samplers}, got '{self.sampler}'")

        valid_pruners = ["median", "hyperband", "none", "percentile"]
        if self.pruner not in valid_pruners:
            issues.append(f"pruner must be one of {valid_pruners}, got '{self.pruner}'")

        valid_directions = ["maximize", "minimize"]
        if self.direction not in valid_directions:
            issues.append(f"direction must be one of {valid_directions}, got '{self.direction}'")

        valid_metrics = [
            "f1_weighted",
            "accuracy",
            "sharpe_ratio",
            "sortino_ratio",
            "profit_factor",
            "precision",
            "recall",
            "roc_auc",
            "log_loss",
        ]
        if self.metric not in valid_metrics:
            issues.append(f"metric must be one of {valid_metrics}, got '{self.metric}'")

        if self.variance_penalty < 0:
            issues.append(f"variance_penalty must be non-negative, got {self.variance_penalty}")

        if self.max_samples <= 0:
            issues.append(f"max_samples must be positive, got {self.max_samples}")

        return issues


# =============================================================================
# CALIBRATION CONFIGURATION
# =============================================================================


@dataclass
class CalibrationConfig(BaseConfig):
    """
    Configuration for probability calibration.

    This is the CANONICAL CalibrationConfig. Use this instead of:
    - CalibrationConfig in src/config/global_config.py (deprecated)
    - CalibrationConfig in src/models/calibration/calibrator.py (deprecated)

    Attributes:
        enabled: Whether calibration is enabled
        method: Calibration method ('isotonic', 'platt', 'temperature', 'auto')
        cv_folds: Number of cross-validation folds for calibration
        ensemble_calibration: Whether to calibrate ensemble predictions

    Example:
        config = CalibrationConfig(
            enabled=True,
            method="isotonic",
            cv_folds=5,
        )
    """

    enabled: bool = True
    method: str = "auto"  # auto, isotonic, platt, temperature
    cv_folds: int = 5
    ensemble_calibration: bool = True

    # Temperature scaling specific
    temperature_learning_rate: float = 0.01
    temperature_max_iter: int = 100

    def validate(self) -> list[str]:
        """Validate calibration configuration."""
        issues = super().validate()

        valid_methods = ["auto", "isotonic", "platt", "temperature", "none"]
        if self.method not in valid_methods:
            issues.append(f"method must be one of {valid_methods}, got '{self.method}'")

        if self.cv_folds < 2:
            issues.append(f"cv_folds must be >= 2, got {self.cv_folds}")

        return issues


# =============================================================================
# CONFORMAL CONFIGURATION
# =============================================================================


@dataclass
class ConformalConfig(BaseConfig):
    """
    Configuration for conformal prediction.

    Attributes:
        enabled: Whether conformal prediction is enabled
        alpha: Miscoverage rate (1 - confidence level)
        method: Conformal method ('naive', 'aps', 'raps')
        calibration_size: Size of calibration set

    Example:
        config = ConformalConfig(
            enabled=True,
            alpha=0.1,
            method="aps",
        )
    """

    enabled: bool = False
    alpha: float = 0.1
    method: str = "aps"
    calibration_size: float = 0.2

    def validate(self) -> list[str]:
        """Validate conformal configuration."""
        issues = super().validate()

        if not 0 < self.alpha < 1:
            issues.append(f"alpha must be in (0, 1), got {self.alpha}")

        valid_methods = ["naive", "aps", "raps"]
        if self.method not in valid_methods:
            issues.append(f"method must be one of {valid_methods}, got '{self.method}'")

        if not 0 < self.calibration_size < 1:
            issues.append(f"calibration_size must be in (0, 1), got {self.calibration_size}")

        return issues


# =============================================================================
# CHECKPOINT CONFIGURATION
# =============================================================================


@dataclass
class CheckpointConfig(BaseConfig):
    """
    Configuration for model checkpointing.

    This is the CANONICAL CheckpointConfig. Use this instead of:
    - CheckpointConfig in src/models/neural/checkpointing.py (deprecated)

    Attributes:
        enabled: Whether checkpointing is enabled
        save_interval: Save every N epochs
        keep_n_checkpoints: Number of recent checkpoints to keep
        save_best_only: Only save when validation improves
        checkpoint_dir: Directory for checkpoints
        monitor_metric: Metric to monitor for best model

    Example:
        config = CheckpointConfig(
            enabled=True,
            save_interval=5,
            keep_n_checkpoints=3,
        )
    """

    enabled: bool = True
    save_interval: int = 10
    keep_n_checkpoints: int = 3
    save_best_only: bool = True
    checkpoint_dir: str | Path | None = None
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"  # min or max

    def validate(self) -> list[str]:
        """Validate checkpoint configuration."""
        issues = super().validate()

        if self.save_interval <= 0:
            issues.append(f"save_interval must be positive, got {self.save_interval}")

        if self.keep_n_checkpoints <= 0:
            issues.append(f"keep_n_checkpoints must be positive, got {self.keep_n_checkpoints}")

        valid_modes = ["min", "max"]
        if self.monitor_mode not in valid_modes:
            issues.append(f"monitor_mode must be one of {valid_modes}, got '{self.monitor_mode}'")

        return issues


# =============================================================================
# OOM RECOVERY CONFIGURATION
# =============================================================================


@dataclass
class OOMConfig(BaseConfig):
    """
    Configuration for out-of-memory recovery.

    This is the CANONICAL OOMConfig. Use this instead of:
    - OOMConfig in src/models/neural/oom_recovery.py (deprecated)
    - OOMRecoveryConfig in src/config/global_config.py (deprecated)

    Attributes:
        enabled: Whether OOM recovery is enabled
        max_retries: Maximum number of retry attempts
        batch_reduction_factor: Factor to reduce batch size by
        min_batch_size: Minimum batch size to try
        clear_cache: Whether to clear GPU cache on OOM

    Example:
        config = OOMConfig(
            enabled=True,
            max_retries=3,
            batch_reduction_factor=0.5,
        )
    """

    enabled: bool = True
    max_retries: int = 3
    batch_reduction_factor: float = 0.5
    min_batch_size: int = 8
    clear_cache: bool = True

    def validate(self) -> list[str]:
        """Validate OOM configuration."""
        issues = super().validate()

        if self.max_retries < 0:
            issues.append(f"max_retries must be non-negative, got {self.max_retries}")

        if not 0 < self.batch_reduction_factor < 1:
            issues.append(
                f"batch_reduction_factor must be in (0, 1), got {self.batch_reduction_factor}"
            )

        if self.min_batch_size <= 0:
            issues.append(f"min_batch_size must be positive, got {self.min_batch_size}")

        return issues


# =============================================================================
# PARALLEL TRAINING CONFIGURATION
# =============================================================================


@dataclass
class ParallelTrainingConfig(BaseConfig):
    """
    Configuration for parallel model training.

    This is the CANONICAL ParallelTrainingConfig. Use this instead of:
    - ParallelTrainingConfig in src/models/training/services/parallel_training.py (deprecated)

    Attributes:
        enabled: Whether parallel training is enabled
        n_jobs: Number of parallel jobs (-1 = all cores)
        backend: Parallel backend ('loky', 'multiprocessing', 'threading')
        batch_size: Number of models to train per batch
        memory_per_model_gb: Estimated memory per model in GB

    Example:
        config = ParallelTrainingConfig(
            enabled=True,
            n_jobs=4,
            backend="loky",
        )
    """

    enabled: bool = False
    n_jobs: int = -1
    backend: str = "loky"
    batch_size: int = 4
    memory_per_model_gb: float = 2.0

    # GPU specific
    gpu_memory_fraction: float = 0.9
    allow_multi_gpu: bool = True

    def validate(self) -> list[str]:
        """Validate parallel training configuration."""
        issues = super().validate()

        if self.n_jobs == 0:
            issues.append("n_jobs cannot be 0")

        valid_backends = ["loky", "multiprocessing", "threading"]
        if self.backend not in valid_backends:
            issues.append(f"backend must be one of {valid_backends}, got '{self.backend}'")

        if self.batch_size <= 0:
            issues.append(f"batch_size must be positive, got {self.batch_size}")

        return issues


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================


@dataclass
class ExperimentConfig(BaseConfig):
    """
    Configuration for experiment tracking.

    This is the CANONICAL ExperimentConfig. Use this instead of:
    - ExperimentConfig in src/models/training/config.py (deprecated)

    Attributes:
        name: Experiment name
        description: Experiment description
        tags: Tags for filtering/grouping
        tracking_enabled: Whether to track metrics
        tracking_backend: Backend for tracking ('local', 'mlflow', 'wandb')
        tracking_uri: URI for remote tracking server

    Example:
        config = ExperimentConfig(
            name="mes_xgboost_h20",
            description="XGBoost on MES futures, 20-bar horizon",
            tracking_backend="mlflow",
        )
    """

    name: str = ""
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    tracking_enabled: bool = True
    tracking_backend: str = "mlflow"
    tracking_uri: str | None = None

    # Run configuration
    run_id: str | None = None
    parent_run_id: str | None = None
    log_artifacts: bool = True
    log_models: bool = True

    def validate(self) -> list[str]:
        """Validate experiment configuration."""
        issues = super().validate()

        valid_backends = ["local", "mlflow", "wandb", "none"]
        if self.tracking_backend not in valid_backends:
            issues.append(
                f"tracking_backend must be one of {valid_backends}, "
                f"got '{self.tracking_backend}'"
            )

        return issues


# =============================================================================
# GENETIC ALGORITHM CONFIGURATION
# =============================================================================


@dataclass
class GAConfig(BaseConfig):
    """
    Configuration for genetic algorithm optimization.

    This is the CANONICAL GAConfig. Use this instead of:
    - GAConfig in src/config/global_config.py (deprecated)

    Attributes:
        population_size: Number of individuals per generation
        generations: Number of generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
        elite_size: Number of elite individuals to preserve
        safe_mode: Use safe (slower but stable) operations

    Example:
        config = GAConfig(
            population_size=50,
            generations=100,
            crossover_rate=0.8,
        )
    """

    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 5
    safe_mode: bool = True

    # Selection
    selection_method: str = "tournament"
    tournament_size: int = 3

    def validate(self) -> list[str]:
        """Validate GA configuration."""
        issues = super().validate()

        if self.population_size < 2:
            issues.append(f"population_size must be >= 2, got {self.population_size}")

        if self.generations <= 0:
            issues.append(f"generations must be positive, got {self.generations}")

        if not 0 <= self.crossover_rate <= 1:
            issues.append(f"crossover_rate must be in [0, 1], got {self.crossover_rate}")

        if not 0 <= self.mutation_rate <= 1:
            issues.append(f"mutation_rate must be in [0, 1], got {self.mutation_rate}")

        if self.elite_size < 0:
            issues.append(f"elite_size must be non-negative, got {self.elite_size}")

        if self.elite_size >= self.population_size:
            issues.append(
                f"elite_size ({self.elite_size}) must be < population_size ({self.population_size})"
            )

        return issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Optimization
    "OptunaConfig",
    "GAConfig",
    # Calibration
    "CalibrationConfig",
    "ConformalConfig",
    # Training management
    "CheckpointConfig",
    "OOMConfig",
    "ParallelTrainingConfig",
    "ExperimentConfig",
]
