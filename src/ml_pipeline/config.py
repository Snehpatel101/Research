"""
Unified ML Pipeline Configuration.

Merges PipelineConfig, ExperimentConfig, and TrainerConfig into a single
cohesive configuration dataclass for the entire ML workflow.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Re-export ModelConfig for convenience
from src.training.config import ModelConfig

__all__ = ["MLConfig", "ModelConfig"]


def _generate_run_id() -> str:
    """Generate unique run ID with timestamp and random suffix."""
    import secrets

    return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{secrets.token_hex(2)}"


def _get_global_or_default(attr_path: str, fallback: Any) -> Any:
    """Get value from global config or return fallback."""
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
class MLConfig:
    """
    Unified configuration for the entire ML pipeline.

    Replaces and merges:
    - PipelineConfig (Phase 1 data pipeline)
    - ExperimentConfig (training orchestration)
    - TrainerConfig (model-specific training)

    Usage:
        config = MLConfig(
            symbol="MES",
            horizons=[20],
            models=["xgboost", "lstm"],
            training_mode="standard",
            build_ensemble=True,
        )

        pipeline = MLPipeline(config)
        results = pipeline.run()
    """

    # ========================================================================
    # RUN IDENTIFICATION
    # ========================================================================
    run_id: str = field(default_factory=_generate_run_id)
    description: str = "ML Pipeline Run"

    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    symbol: str = "MES"
    symbols: list[str] | None = None  # For batch processing (future)
    start_date: str | None = None  # YYYY-MM-DD
    end_date: str | None = None  # YYYY-MM-DD

    # ========================================================================
    # TIMEFRAME CONFIGURATION
    # ========================================================================
    target_timeframe: str = field(
        default_factory=lambda: _get_global_or_default("timeframes.default_primary", "5min")
    )
    output_timeframes: list[str] | None = None
    process_all_timeframes: bool = False  # MTF-P1-002: Process all 9 canonical TFs

    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    feature_mode: str = "auto"  # auto, full, minimal, hft_only
    enable_wavelets: bool = True
    enable_microstructure: bool = True
    enable_volume: bool = True
    enable_volatility: bool = True

    # MTF (Multi-Timeframe) features
    enable_mtf: bool = True
    mtf_mode: str = "both"  # bars, indicators, both
    mtf_timeframes: list[str] | None = None  # None = use defaults

    # ========================================================================
    # LABELING CONFIGURATION
    # ========================================================================
    horizons: list[int] = field(default_factory=lambda: [5, 10, 15, 20])
    labeling_method: str = "triple_barrier"
    k_up: float | None = None  # Symbol-specific if None
    k_down: float | None = None  # Symbol-specific if None
    max_bars: int | None = None  # Horizon-specific if None

    # Optuna optimization
    optimize_labeling: bool = True
    labeling_optimization_trials: int = 100

    # ========================================================================
    # SPLIT CONFIGURATION
    # ========================================================================
    train_ratio: float = field(
        default_factory=lambda: _get_global_or_default("splits.train_ratio", 0.70)
    )
    val_ratio: float = field(
        default_factory=lambda: _get_global_or_default("splits.val_ratio", 0.15)
    )
    test_ratio: float = field(
        default_factory=lambda: _get_global_or_default("splits.test_ratio", 0.15)
    )
    purge_bars: int = field(default_factory=lambda: _get_global_or_default("splits.purge_bars", 60))
    embargo_bars: int = field(
        default_factory=lambda: _get_global_or_default("splits.embargo_bars", 1440)
    )

    # ========================================================================
    # SCALING
    # ========================================================================
    scaler_type: str = "robust"  # robust, standard, minmax, none

    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    models: list[str] | list[ModelConfig] = field(default_factory=list)

    # Global optimization settings (applied to all models if not overridden)
    global_feature_optimization: bool = False
    global_hyperparam_optimization: bool = False

    # ========================================================================
    # ENSEMBLE CONFIGURATION
    # ========================================================================
    build_ensemble: bool = False
    ensemble_method: str = "stacking"  # voting, stacking, blending
    meta_learner: str = "ridge_meta"  # ridge_meta, mlp_meta, xgboost_meta, etc.

    # ========================================================================
    # TRAINING MODE
    # ========================================================================
    training_mode: str = "standard"  # standard, walk_forward, regime_aware, meta_labeling

    # Walk-forward specific
    walk_forward_window_size: int = 5000
    walk_forward_step_size: int = 1000
    walk_forward_min_train_size: int = 3000

    # Regime-aware specific
    regime_detection_method: str = "volatility"  # volatility, trend, composite
    train_separate_regimes: bool = True

    # Meta-labeling specific
    meta_labeling_base_model: str | None = None
    meta_labeling_meta_model: str = "logistic"

    # ========================================================================
    # CROSS-VALIDATION
    # ========================================================================
    cross_validate: bool = True
    cv_splits: int = field(
        default_factory=lambda: _get_global_or_default("cross_validation.n_splits", 5)
    )
    cv_purge_bars: int | None = None  # Use purge_bars if None
    cv_embargo_bars: int | None = None  # Use embargo_bars if None

    # ========================================================================
    # EVALUATION METHODS
    # ========================================================================
    evaluation_methods: list[str] = field(default_factory=lambda: ["cv"])
    # Options: cv, walk_forward, cpcv_pbo

    # CPCV-PBO specific
    cpcv_pbo_n_splits: int = 10
    cpcv_pbo_n_tests: int = 16

    # ========================================================================
    # TRAINING CONFIGURATION (from TrainerConfig)
    # ========================================================================
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

    # ========================================================================
    # CALIBRATION
    # ========================================================================
    use_calibration: bool = field(
        default_factory=lambda: _get_global_or_default("calibration.enabled", True)
    )
    calibration_method: str = field(
        default_factory=lambda: _get_global_or_default("calibration.method", "auto")
    )

    # ========================================================================
    # OUTPUT CONFIGURATION
    # ========================================================================
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Tracking (MLflow, Weights & Biases, etc.)
    tracking_enabled: bool = field(
        default_factory=lambda: _get_global_or_default("tracking.enabled", True)
    )
    tracking_backend: str = field(
        default_factory=lambda: _get_global_or_default("tracking.backend", "local")
    )
    tracking_uri: str | None = None
    tracking_tags: dict[str, str] = field(default_factory=dict)

    # ========================================================================
    # CHECKPOINTING & RECOVERY
    # ========================================================================
    checkpoint_interval: int = 10
    keep_n_checkpoints: int = 3
    oom_recovery_enabled: bool = field(
        default_factory=lambda: _get_global_or_default("oom_recovery.enabled", True)
    )
    oom_max_retries: int = field(
        default_factory=lambda: _get_global_or_default("oom_recovery.max_retries", 3)
    )

    # ========================================================================
    # PARALLEL EXECUTION
    # ========================================================================
    parallel_training: bool = False
    n_jobs: int = -1

    # ========================================================================
    # VALIDATION & SAFETY
    # ========================================================================
    deterministic_mode: bool = False
    nan_check_raise_error: bool = True
    validate_pipeline_output: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        if not self.symbols and self.symbol:
            self.symbols = [self.symbol]

        if self.models and isinstance(self.models[0], str):
            self.models = [
                ModelConfig(
                    name=m,
                    optimize_features=self.global_feature_optimization,
                    optimize_hyperparams=self.global_hyperparam_optimization,
                )
                for m in self.models
            ]

        for horizon in self.horizons:
            if horizon <= 0:
                raise ValueError(f"All horizons must be positive, got {horizon}")

        if self.train_ratio + self.val_ratio + self.test_ratio != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0, got "
                f"{self.train_ratio + self.val_ratio + self.test_ratio}"
            )

        if self.cv_purge_bars is None:
            self.cv_purge_bars = self.purge_bars
        if self.cv_embargo_bars is None:
            self.cv_embargo_bars = self.embargo_bars

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLConfig":
        """Create from dictionary."""
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MLConfig":
        """Load from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Save to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
