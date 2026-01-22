"""
THE ONE CONFIG - PipelineConfig

This is THE ONLY config you need. Everything else is deprecated.

Usage:
    from src import PipelineConfig, MLPipeline

    config = PipelineConfig(
        symbol="MES",
        models=["xgboost", "lstm"],
        build_ensemble=True,
    )

    result = MLPipeline(config).run()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _generate_run_id() -> str:
    """Generate unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:6]
    return f"run_{timestamp}_{unique}"


@dataclass
class PipelineConfig:
    """
    THE ONE CONFIG for the entire ML pipeline.

    This single config controls ALL aspects of the pipeline:
    - Data preparation
    - Feature engineering
    - Labeling
    - Training
    - Ensemble
    - Evaluation
    - Bundling
    - Backtesting
    - Inference

    Example:
        # Minimal usage (just symbol required)
        config = PipelineConfig(symbol="MES")

        # Full customization
        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes_1min.parquet",
            output_dir="./experiments",

            # Models
            models=["xgboost", "lightgbm", "lstm"],
            horizons=[5, 10, 15, 20],
            build_ensemble=True,

            # Training
            training_mode="standard",
            cv_method="purged_kfold",
            n_splits=5,

            # Optimization
            optimize_labels=True,
            optimize_features=True,
            optimize_hyperparams=True,
        )
    """

    # =========================================================================
    # REQUIRED
    # =========================================================================
    symbol: str

    # =========================================================================
    # DATA SETTINGS
    # =========================================================================
    data_path: str | Path | None = None  # Auto-resolved from symbol if None
    output_dir: str | Path = field(default_factory=lambda: Path("experiments/runs"))

    # =========================================================================
    # IDENTITY
    # =========================================================================
    run_id: str = field(default_factory=_generate_run_id)
    description: str = ""
    random_seed: int = 42

    # =========================================================================
    # MODELS
    # =========================================================================
    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    horizons: list[int] = field(default_factory=lambda: [20])

    # =========================================================================
    # ENSEMBLE
    # =========================================================================
    build_ensemble: bool = True
    ensemble_method: str = "stacking"  # stacking, voting, blending
    meta_learner: str = "ridge"  # ridge, mlp, xgboost, calibrated

    # =========================================================================
    # TRAINING MODE
    # =========================================================================
    training_mode: str = "standard"  # standard, walk_forward, regime_aware, meta_labeling

    # Walk-forward settings
    walk_forward_window: int = 5000
    walk_forward_step: int = 1000
    walk_forward_min_train: int = 2000

    # Regime-aware settings
    regime_detection_method: str = "volatility_percentile"  # volatility_percentile, trend_adx, combined
    regime_lookback: int = 100
    n_regimes: int = 3
    regime_volatility_window: int = 20
    regime_adx_threshold: float = 25.0
    regime_min_samples: int = 100
    train_separate_regime_models: bool = True

    # Meta-labeling settings
    meta_labeling_primary_model: str = "xgboost"
    meta_labeling_meta_model: str = "lightgbm"
    meta_labeling_threshold: float = 0.5

    # =========================================================================
    # CROSS-VALIDATION
    # =========================================================================
    cv_method: str = "purged_kfold"  # purged_kfold, cpcv, walk_forward
    n_splits: int = 5
    purge_bars: int = 10
    embargo_bars: int = 5

    # CPCV settings
    cpcv_n_splits: int = 5
    cpcv_n_tests: int = 2

    # =========================================================================
    # DATA SPLITS
    # =========================================================================
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # =========================================================================
    # LABELING
    # =========================================================================
    labeling_method: str = "triple_barrier"  # triple_barrier, directional, threshold
    upper_mult: float = 2.0
    lower_mult: float = 2.0
    atr_period: int = 14
    max_holding_bars: int = 20

    # =========================================================================
    # FEATURES
    # =========================================================================
    feature_mode: str = "full"  # full, minimal, custom
    feature_families: list[str] = field(default_factory=lambda: [
        "price", "momentum", "volatility", "volume", "trend"
    ])
    compute_mtf_features: bool = True
    mtf_timeframes: list[str] = field(default_factory=lambda: ["5min", "15min", "1h"])

    # =========================================================================
    # OPTIMIZATION
    # =========================================================================
    optimize_labels: bool = True
    label_optimization_trials: int = 50

    optimize_features: bool = True
    feature_optimization_trials: int = 50

    optimize_hyperparams: bool = True
    hyperparam_optimization_trials: int = 100

    # =========================================================================
    # TRAINING SETTINGS
    # =========================================================================
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "auto"  # auto, cpu, cuda, mps
    parallel_training: bool = False  # Enable parallel model training
    n_jobs: int = -1  # -1 = all cores

    # Neural network specific
    sequence_length: int = 60
    learning_rate: float = 0.001

    # =========================================================================
    # CALIBRATION
    # =========================================================================
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"  # isotonic, platt, temperature

    # =========================================================================
    # OUTPUT SETTINGS
    # =========================================================================
    save_models: bool = True
    save_oof: bool = True
    save_metrics: bool = True
    save_bundle: bool = True

    # =========================================================================
    # TRACKING
    # =========================================================================
    tracking_enabled: bool = False
    tracking_backend: str = "local"  # local, mlflow
    tracking_uri: str | None = None

    # =========================================================================
    # EVALUATION
    # =========================================================================
    run_evaluation: bool = True
    run_backtest: bool = True

    # =========================================================================
    # INTERNAL (auto-populated)
    # =========================================================================
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate and resolve paths."""
        # Resolve data_path from symbol if not provided
        if self.data_path is None:
            self.data_path = Path(f"data/{self.symbol.lower()}_1min.parquet")

        # Convert paths to Path objects
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate ratios sum to 1
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        # Validate training mode
        valid_modes = ["standard", "walk_forward", "regime_aware", "meta_labeling"]
        if self.training_mode not in valid_modes:
            raise ValueError(f"training_mode must be one of {valid_modes}")

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_run_dir(self) -> Path:
        """Get the directory for this run's outputs."""
        return Path(self.output_dir) / self.run_id

    def summary(self) -> str:
        """Get a human-readable summary of the config."""
        lines = [
            f"PipelineConfig Summary",
            f"=" * 50,
            f"Run ID: {self.run_id}",
            f"Symbol: {self.symbol}",
            f"Data: {self.data_path}",
            f"Output: {self.output_dir}",
            f"",
            f"Models: {', '.join(self.models)}",
            f"Horizons: {self.horizons}",
            f"Ensemble: {self.build_ensemble} ({self.ensemble_method})",
            f"",
            f"Training Mode: {self.training_mode}",
            f"CV Method: {self.cv_method} ({self.n_splits} splits)",
            f"",
            f"Optimization:",
            f"  Labels: {self.optimize_labels} ({self.label_optimization_trials} trials)",
            f"  Features: {self.optimize_features} ({self.feature_optimization_trials} trials)",
            f"  Hyperparams: {self.optimize_hyperparams} ({self.hyperparam_optimization_trials} trials)",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
        result = asdict(self)
        # Convert Path objects to strings for serialization
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Save config to file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        """Load config from file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def quick_config(symbol: str, **overrides) -> PipelineConfig:
    """Quick iteration config - fast training, minimal optimization."""
    defaults = {
        "symbol": symbol,
        "models": ["xgboost"],
        "horizons": [20],
        "optimize_labels": False,
        "optimize_features": False,
        "optimize_hyperparams": False,
        "n_splits": 3,
        "build_ensemble": False,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def production_config(symbol: str, **overrides) -> PipelineConfig:
    """Production config - full optimization, robust validation."""
    defaults = {
        "symbol": symbol,
        "models": ["xgboost", "lightgbm", "catboost", "lstm"],
        "horizons": [5, 10, 15, 20],
        "optimize_labels": True,
        "optimize_features": True,
        "optimize_hyperparams": True,
        "label_optimization_trials": 100,
        "feature_optimization_trials": 100,
        "hyperparam_optimization_trials": 200,
        "n_splits": 5,
        "build_ensemble": True,
        "ensemble_method": "stacking",
        "run_evaluation": True,
        "run_backtest": True,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def research_config(symbol: str, **overrides) -> PipelineConfig:
    """Research config - many models, extensive analysis."""
    defaults = {
        "symbol": symbol,
        "models": [
            "xgboost", "lightgbm", "catboost",
            "lstm", "gru", "tcn", "transformer",
        ],
        "horizons": [5, 10, 15, 20, 30, 60],
        "training_mode": "walk_forward",
        "optimize_labels": True,
        "optimize_features": True,
        "optimize_hyperparams": True,
        "build_ensemble": True,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)
