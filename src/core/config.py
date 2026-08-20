"""
Centralized Pipeline Configuration - THE Single Source of Truth.

PHASE_0 + HIGH_LEVEL_ARCHITECTURE: This module defines PipelineConfig,
the ONE configuration object that controls the ENTIRE pipeline.

Design:
- ONE config object controls EVERYTHING
- Set it once in Jupyter, pipeline handles the rest
- All defaults come from src/core/constants.py
- Config is serializable (save/load for reproducibility)

Usage:
    from src.core.config import PipelineConfig

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes_1min.parquet",
        models=["xgboost", "lightgbm", "lstm"],
        horizons=[20],
        build_ensemble=True,
        output_dir="./experiments/exp_001",
    )

    pipeline = MLFactory(config)
    results = pipeline.run()
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.pipeline.data_config import DataConfig

from src.core.constants import (
    ALL_MODELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EMBARGO_BARS,
    DEFAULT_FEATURE_PRUNING_TRIALS,
    DEFAULT_FEATURE_SELECTION_TRIALS,
    DEFAULT_HORIZON,
    DEFAULT_HYPERPARAM_TRIALS,
    DEFAULT_LABEL_OPTIMIZATION_TRIALS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_FEATURES,
    DEFAULT_MTF_TIMEFRAMES,
    DEFAULT_N_SPLITS,
    DEFAULT_OPTUNA_RANDOM_STATE,
    DEFAULT_OPTUNA_TIMEOUT,
    DEFAULT_PURGE_BARS,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SPLIT_RATIOS,
)
from src.core.types import CVMethod, LabelingMethod, TrainingMode
from src.core.validation import ValidationError, validate_model_list, validate_path_exists


@dataclass
class PipelineConfig:
    """
    ONE configuration object that controls the ENTIRE pipeline.

    Set it once in your Jupyter notebook, and the pipeline
    handles everything else automatically.

    Example:
        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes_1min.parquet",
            models=["xgboost", "lightgbm", "lstm"],
            horizons=[20],
            build_ensemble=True,
            output_dir="./experiments/exp_001",
        )
    """

    # =========================================================================
    # REQUIRED FIELDS (no defaults)
    # =========================================================================

    symbol: str  # Trading symbol (e.g., "MES", "ES", "NQ")
    data_path: str | Path  # Path to 1-min OHLCV parquet
    output_dir: str | Path  # Where to save everything

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================

    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    # Available models:
    # Boosting: "xgboost", "lightgbm", "catboost"
    # Classical: "random_forest", "logistic", "svm"
    # Neural: "lstm", "gru", "tcn", "transformer", "patchtst", "itransformer",
    #         "tft", "nbeats", "inceptiontime", "resnet1d"

    horizons: list[int] = field(default_factory=lambda: [DEFAULT_HORIZON])
    # Prediction horizons (in bars). Default: [20]

    # =========================================================================
    # TRAINING CONFIGURATION
    # =========================================================================

    training_mode: str = TrainingMode.STANDARD.value
    # Options: "standard", "walk_forward", "regime_aware", "meta_labeling"

    cv_method: str = CVMethod.PURGED_KFOLD.value
    # Options: "purged_kfold", "cpcv", "walk_forward", "pbo"

    n_splits: int = DEFAULT_N_SPLITS  # CV folds (default: 5)
    purge_bars: int = DEFAULT_PURGE_BARS  # Gap before test (default: 60)
    embargo_bars: int = DEFAULT_EMBARGO_BARS  # Embargo after test (default: 1440)

    # Walk-forward validation settings
    wf_n_windows: int = 5  # Number of walk-forward windows
    wf_window_type: str = "expanding"  # "expanding" or "rolling"
    wf_min_train_pct: float = 0.4  # Minimum training data percentage
    wf_test_pct: float = 0.1  # Test data percentage per window

    # Split ratios
    train_ratio: float = DEFAULT_SPLIT_RATIOS["train"]  # 0.70
    val_ratio: float = DEFAULT_SPLIT_RATIOS["val"]  # 0.15
    test_ratio: float = DEFAULT_SPLIT_RATIOS["test"]  # 0.15

    # =========================================================================
    # ENSEMBLE CONFIGURATION
    # =========================================================================

    build_ensemble: bool = True  # Build stacking ensemble?
    meta_learner: str = "ridge_meta"
    # Options: "ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"

    # =========================================================================
    # META-LABELING CONFIGURATION (Lopez de Prado 2018)
    # =========================================================================
    # Meta-labeling is a two-stage approach:
    # 1. Primary model predicts direction (+1, -1)
    # 2. Meta-model predicts probability primary is correct (bet sizing)
    # Final position = primary_direction * meta_probability

    meta_labeling_primary_model: str = "xgboost"
    # Primary model for side/direction prediction
    # Options: any model from ALL_MODELS (boosting recommended)

    meta_labeling_meta_model: str = "logistic"
    # Meta-model for bet sizing (probability of primary being correct)
    # Options: "logistic", "xgboost", "lightgbm", "random_forest"

    meta_labeling_threshold: float = 0.5
    # Minimum meta-model probability required to take a trade
    # Range: [0.0, 1.0] - higher = more selective (fewer trades, higher accuracy)

    # =========================================================================
    # FEATURE CONFIGURATION
    # =========================================================================

    feature_families: str | list[str] = "auto"
    # "auto" = pipeline selects per model type
    # Or specify: ["momentum", "volatility", "volume", "trend", ...]

    mtf_timeframes: list[str] = field(default_factory=lambda: DEFAULT_MTF_TIMEFRAMES.copy())
    # MTF timeframes: ["5min", "15min", "60min"]

    compute_mtf_features: bool = True  # Whether to compute MTF features

    sequence_length: int = DEFAULT_SEQUENCE_LENGTH  # For neural models (default: 60)

    # =========================================================================
    # LABELING CONFIGURATION
    # =========================================================================

    labeling_method: str = LabelingMethod.TRIPLE_BARRIER.value
    # Options: "triple_barrier", "directional", "threshold", "regression"

    # Triple-barrier overrides. None (default) = use the per-symbol/per-horizon
    # BARRIER_PARAMS table (same source the backtester uses). Set explicitly to
    # override both labeling and backtest barriers.
    upper_mult: float | None = None  # Upper barrier = ATR * upper_mult
    lower_mult: float | None = None  # Lower barrier = ATR * lower_mult
    atr_period: int = 14  # ATR calculation period
    max_holding_bars: int | None = None  # Maximum holding period for triple-barrier

    # =========================================================================
    # OPTUNA OPTIMIZATION CONFIGURATION
    # =========================================================================

    # Label optimization (triple-barrier parameters)
    optimize_labels: bool = True
    label_optimization_trials: int = DEFAULT_LABEL_OPTIMIZATION_TRIALS  # 100
    target_class_distribution: dict[str, float] | None = None  # Default: balanced

    # Feature optimization
    optimize_features: bool = True
    feature_selection_method: str = "optuna"  # Options: "optuna", "shap", "mutual_info"
    feature_selection_trials: int = DEFAULT_FEATURE_SELECTION_TRIALS  # 100
    feature_pruning_trials: int = DEFAULT_FEATURE_PRUNING_TRIALS  # 50
    min_features: int = DEFAULT_MIN_FEATURES  # 20

    # Hyperparameter optimization
    optimize_hyperparams: bool = True
    hyperparam_trials: int = DEFAULT_HYPERPARAM_TRIALS  # 100

    # Optuna settings
    optuna_random_state: int = DEFAULT_OPTUNA_RANDOM_STATE  # 42
    # Wall-clock cap for each Optuna study (seconds). None/0 = unbounded.
    optuna_timeout: int | None = DEFAULT_OPTUNA_TIMEOUT  # 43200 = 12h
    optuna_metric: str = "f1_weighted"  # Optimization metric (from OptunaConfig.metric)

    enforce_dsr_gate: bool = True
    # Enforce Deflated Sharpe Ratio gate after optimization.
    # When True, optimization raises ValueError if DSR is below deployment threshold,
    # preventing deployment of strategies that appear good only due to selection bias.

    dsr_deployment_threshold: float = 0.5
    # Minimum Deflated Sharpe Ratio for deployment (default: 0.5)

    # =========================================================================
    # REGIME-AWARE TRAINING CONFIGURATION
    # =========================================================================

    regime_detection_method: str = "volatility_percentile"
    # Options: "volatility_percentile", "trend_adx", "combined"
    # - volatility_percentile: Uses rolling volatility percentiles
    # - trend_adx: Uses ADX for trend detection
    # - combined: Combines volatility and trend for composite regimes

    regime_lookback: int = 60  # Lookback period for regime detection
    n_regimes: int = 3  # Number of regime states (2=low/high, 3=low/medium/high)

    regime_volatility_window: int = 20  # Rolling window for volatility calculation
    # ADX threshold for trending classification. None (default) = auto: use the
    # per-symbol preset from SymbolConfig (MES=20, MGC=23, MNQ=25). Any explicit
    # float is honored as-is — including 25.0.
    regime_adx_threshold: float | None = None
    regime_min_samples: int = 100  # Minimum samples per regime for training
    train_separate_regime_models: bool = True  # Train separate models per regime

    # =========================================================================
    # NEURAL NETWORK CONFIGURATION
    # =========================================================================

    batch_size: int = DEFAULT_BATCH_SIZE  # 512 (Phase 68)
    max_epochs: int = DEFAULT_MAX_EPOCHS  # 100
    learning_rate: float = DEFAULT_LEARNING_RATE  # 0.001
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE  # 10

    # =========================================================================
    # OUTPUT CONFIGURATION
    # =========================================================================

    save_bundle: bool = True  # Save inference bundle
    save_oof: bool = True  # Save OOF predictions
    save_models: bool = True  # Save individual model artifacts
    save_metrics: bool = True  # Save metrics JSON

    # =========================================================================
    # VALIDATION CONFIGURATION (Phase 1: Contract Enforcement)
    # =========================================================================

    strict_validation: bool = True
    # Enable strict contract validation before training
    # When True, training will fail if data contracts are violated

    check_leakage: bool = True
    # Run leakage detection before training
    # When True, checks for feature-label correlation leakage

    check_lookahead: bool = True
    # Run lookahead audit before training
    # When True, validates that features don't contain future information

    validation_correlation_threshold: float = 0.5
    # Threshold for leakage detection (correlation above this is suspicious)

    validation_fail_on_warning: bool = False
    # When True, warnings from validation (e.g., implicit resample params) also fail
    # When False, only hard failures (actual leakage/lookahead) block training

    # =========================================================================
    # CALIBRATION CONFIGURATION (Phase 4F)
    # =========================================================================

    auto_calibrate: bool = True
    # Automatically calibrate model probabilities after training
    # When True, applies probability calibration to improve reliability

    calibration_method: str = "auto"
    # Calibration method: "isotonic", "sigmoid", or "auto"
    # "auto" selects isotonic for boosting, sigmoid for linear models

    calibration_min_samples: int = 100
    # Minimum samples required per class for calibration

    # =========================================================================
    # BET SIZING CONFIGURATION (Phase 4G)
    # =========================================================================

    use_bet_sizing: bool = False
    # Connect bet sizing to position sizing
    # When True, uses variable position sizing based on model confidence

    bet_sizing_strategy: str = "binary"
    # Bet sizing strategy: "binary", "proportional", "kelly", "half_kelly", "confidence"

    bet_sizing_threshold: float = 0.5
    # Minimum probability to take a trade

    bet_sizing_max_size: float = 1.0
    # Maximum position size (fraction of capital)

    # =========================================================================
    # MISC CONFIGURATION
    # =========================================================================

    random_state: int = 42  # Global random seed
    n_jobs: int = -1  # Use all available cores (-1 = auto-detect)
    verbose: int = 1  # Logging verbosity (0=silent, 1=progress, 2=debug)
    n_classes: int = 3  # Number of classes (2 for binary, 3 for standard)

    # =========================================================================
    # INTERNAL (set automatically)
    # =========================================================================

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization."""
        # Convert paths to Path objects
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)

        # Validate models
        validate_model_list(self.models, "config.models")

        # Validate meta-learner
        valid_meta = ["ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"]
        if self.meta_learner not in valid_meta:
            raise ValidationError(
                f"Invalid meta_learner: {self.meta_learner}",
                field="meta_learner",
                expected=valid_meta,
                actual=self.meta_learner,
            )

        # Validate training mode
        valid_modes = [m.value for m in TrainingMode]
        if self.training_mode not in valid_modes:
            raise ValidationError(
                f"Invalid training_mode: {self.training_mode}",
                field="training_mode",
                expected=valid_modes,
                actual=self.training_mode,
            )

        # Validate CV method
        valid_cv = [m.value for m in CVMethod]
        if self.cv_method not in valid_cv:
            raise ValidationError(
                f"Invalid cv_method: {self.cv_method}",
                field="cv_method",
                expected=valid_cv,
                actual=self.cv_method,
            )

        # Validate labeling method
        valid_labeling = [m.value for m in LabelingMethod]
        if self.labeling_method not in valid_labeling:
            raise ValidationError(
                f"Invalid labeling_method: {self.labeling_method}",
                field="labeling_method",
                expected=valid_labeling,
                actual=self.labeling_method,
            )

        # Validate split ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValidationError(
                "Split ratios must sum to 1.0",
                field="split_ratios",
                expected=1.0,
                actual=total,
            )

        # Validate horizons
        if not self.horizons:
            raise ValidationError("horizons list is empty", field="horizons")

        # Validate regime-aware configuration
        if self.training_mode == TrainingMode.REGIME_AWARE.value:
            valid_regime_methods = ["volatility_percentile", "trend_adx", "combined"]
            if self.regime_detection_method not in valid_regime_methods:
                raise ValidationError(
                    f"Invalid regime_detection_method: {self.regime_detection_method}",
                    field="regime_detection_method",
                    expected=valid_regime_methods,
                    actual=self.regime_detection_method,
                )

            if self.n_regimes not in [2, 3]:
                raise ValidationError(
                    "n_regimes must be 2 or 3",
                    field="n_regimes",
                    expected=[2, 3],
                    actual=self.n_regimes,
                )

        # Validate meta-labeling configuration
        if self.training_mode == TrainingMode.META_LABELING.value:
            # Validate primary model
            try:
                validate_model_list(
                    [self.meta_labeling_primary_model], "meta_labeling_primary_model"
                )
            except ValidationError as err:
                raise ValidationError(
                    f"Invalid meta_labeling_primary_model: {self.meta_labeling_primary_model}",
                    field="meta_labeling_primary_model",
                    expected=ALL_MODELS,
                    actual=self.meta_labeling_primary_model,
                ) from err

            # Validate meta model
            valid_meta_models = ["logistic", "xgboost", "lightgbm", "random_forest", "catboost"]
            if self.meta_labeling_meta_model not in valid_meta_models:
                raise ValidationError(
                    f"Invalid meta_labeling_meta_model: {self.meta_labeling_meta_model}",
                    field="meta_labeling_meta_model",
                    expected=valid_meta_models,
                    actual=self.meta_labeling_meta_model,
                )

            # Validate threshold
            if not 0.0 <= self.meta_labeling_threshold <= 1.0:
                raise ValidationError(
                    "meta_labeling_threshold must be in [0.0, 1.0]",
                    field="meta_labeling_threshold",
                    expected="[0.0, 1.0]",
                    actual=self.meta_labeling_threshold,
                )

        # Auto-populate regime_adx_threshold from SymbolConfig when unset.
        # None is the "auto" sentinel — an explicit user value (even 25.0,
        # the old magic sentinel) is always honored.
        if self.regime_adx_threshold is None:
            from src.config.symbol import SymbolConfig

            sym_cfg = SymbolConfig.from_symbol_or_default(self.symbol)
            self.regime_adx_threshold = sym_cfg.adx_trending_threshold

    def validate_data_path(self) -> None:
        """Validate that data_path exists (call before running pipeline)."""
        validate_path_exists(self.data_path, must_be_file=True, context="data_path")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        d = asdict(self)
        # Convert Path objects to strings
        d["data_path"] = str(d["data_path"])
        d["output_dir"] = str(d["output_dir"])
        return d

    def save(self, path: str | Path | None = None) -> Path:
        """
        Save config to JSON file.

        Args:
            path: Optional path. If None, saves to output_dir/config.json

        Returns:
            Path to saved config file
        """
        if path is None:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            path = output_path / "config.json"
        else:
            path = Path(path)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """
        Load config from JSON file.

        Args:
            path: Path to config JSON file

        Returns:
            Loaded PipelineConfig instance
        """
        path = Path(path)
        with open(path) as f:
            d = json.load(f)

        return cls(**d)

    @property
    def experiment_name(self) -> str:
        """Generate experiment name from config."""
        models_str = "_".join(self.models[:3])
        if len(self.models) > 3:
            models_str += f"_plus{len(self.models)-3}"
        return f"{self.symbol}_{models_str}_h{self.horizons[0]}"

    @property
    def total_optuna_trials(self) -> int:
        """Calculate total Optuna trials for this config."""
        total = 0
        if self.optimize_labels:
            total += self.label_optimization_trials
        if self.optimize_features:
            total += self.feature_selection_trials + self.feature_pruning_trials
        if self.optimize_hyperparams:
            total += self.hyperparam_trials * len(self.models)
        return total

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(\n"
            f"  symbol={self.symbol!r},\n"
            f"  models={self.models},\n"
            f"  horizons={self.horizons},\n"
            f"  training_mode={self.training_mode!r},\n"
            f"  build_ensemble={self.build_ensemble},\n"
            f"  optimize_labels={self.optimize_labels},\n"
            f"  optimize_features={self.optimize_features},\n"
            f"  optimize_hyperparams={self.optimize_hyperparams},\n"
            f"  output_dir={self.output_dir!r},\n"
            f")"
        )

    def to_data_config(self) -> DataConfig:
        """Convert to DataConfig for PipelineRunner."""
        from src.data.pipeline.config_adapter import to_data_config

        return to_data_config(self)

    # Backward compatibility alias
    def to_phase1_config(self) -> DataConfig:
        """Deprecated: Use to_data_config instead."""
        return self.to_data_config()


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================


def quick_config(
    symbol: str,
    data_path: str | Path,
    output_dir: str | Path,
) -> PipelineConfig:
    """
    Quick configuration for fast iteration.

    Uses:
    - 2 boosting models (XGBoost, LightGBM)
    - Single horizon (20)
    - No Optuna optimization
    - No ensemble
    """
    return PipelineConfig(
        symbol=symbol,
        data_path=data_path,
        output_dir=output_dir,
        models=["xgboost", "lightgbm"],
        horizons=[20],
        build_ensemble=False,
        optimize_labels=False,
        optimize_features=False,
        optimize_hyperparams=False,
    )


def production_config(
    symbol: str,
    data_path: str | Path,
    output_dir: str | Path,
) -> PipelineConfig:
    """
    Production configuration for deployment.

    Uses:
    - 5 diverse models (boosting + neural)
    - Walk-forward training
    - Full Optuna optimization
    - Ensemble with XGBoost meta-learner
    """
    return PipelineConfig(
        symbol=symbol,
        data_path=data_path,
        output_dir=output_dir,
        models=["xgboost", "lightgbm", "catboost", "lstm", "transformer"],
        horizons=[20],
        training_mode="walk_forward",
        cv_method="walk_forward",
        build_ensemble=True,
        meta_learner="xgboost_meta",
        optimize_labels=True,
        optimize_features=True,
        optimize_hyperparams=True,
    )


def research_config(
    symbol: str,
    data_path: str | Path,
    output_dir: str | Path,
) -> PipelineConfig:
    """
    Research configuration for experimentation.

    Uses:
    - All available models
    - Multiple horizons
    - Full optimization
    - Ensemble
    """
    return PipelineConfig(
        symbol=symbol,
        data_path=data_path,
        output_dir=output_dir,
        models=[
            "xgboost",
            "lightgbm",
            "catboost",
            "random_forest",
            "lstm",
            "gru",
            "tcn",
            "transformer",
            "tft",
        ],
        horizons=[10, 20],
        build_ensemble=True,
        meta_learner="xgboost_meta",
        optimize_labels=True,
        optimize_features=True,
        optimize_hyperparams=True,
        label_optimization_trials=50,  # Reduced for faster iteration
        hyperparam_trials=50,
    )


def regime_aware_config(
    symbol: str,
    data_path: str | Path,
    output_dir: str | Path,
) -> PipelineConfig:
    """
    Regime-aware configuration for adaptive trading strategies.

    Uses:
    - Regime-aware training mode
    - Combined regime detection (volatility + trend)
    - 3 regime states
    - Separate models per regime
    - Boosting models (work well per-regime)
    - Ensemble of regime-specific predictions

    Example:
        config = regime_aware_config("MES", "./data/mes.parquet", "./experiments")
        orchestrator = UnifiedTrainingOrchestrator(config)
        result = orchestrator.train(df)
    """
    return PipelineConfig(
        symbol=symbol,
        data_path=data_path,
        output_dir=output_dir,
        models=["xgboost", "lightgbm", "catboost"],
        horizons=[20],
        training_mode="regime_aware",
        # Regime-aware settings
        regime_detection_method="combined",
        n_regimes=3,
        regime_lookback=60,
        regime_volatility_window=20,
        # regime_adx_threshold left at None = auto per-symbol preset
        regime_min_samples=100,
        train_separate_regime_models=True,
        # Optimization settings
        build_ensemble=True,
        meta_learner="ridge_meta",
        optimize_labels=True,
        optimize_features=False,  # Faster iteration
        optimize_hyperparams=False,  # Per-regime reduces overfit risk
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PipelineConfig",
    "quick_config",
    "production_config",
    "research_config",
    "regime_aware_config",
]
