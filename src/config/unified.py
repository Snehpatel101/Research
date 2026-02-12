"""
Unified Configuration System.

This module provides a single UnifiedConfig dataclass that consolidates:
- GlobalConfig (from global.yaml)
- MLConfig (84 fields)
- TrainerConfig (63 fields)
- PipelineConfig (88+ fields)
- HorizonConfig

The UnifiedConfig serves as the single source of truth for all configuration
in the ML pipeline, while maintaining backward compatibility with existing
config classes through delegation.

Usage:
    from src.config import UnifiedConfig

    # Load from YAML
    config = UnifiedConfig.from_yaml("config/global.yaml")

    # Or create with overrides
    config = UnifiedConfig(
        symbol="MES",
        horizons=[5, 10, 15, 20],
        training=TrainingSection(batch_size=512),
    )

    # Access nested values
    batch_size = config.training.batch_size
    horizons = config.horizons.active

    # Convert to legacy configs
    trainer_config = config.to_trainer_config(model_name="xgboost")
    pipeline_config = config.to_pipeline_config()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml

from src.config.utils import get_config_value  # noqa: F401 - used in module context
from src.config.validators import (
    ConfigValidationError,
    ValidationResult,
    coerce_types,
    validate_config,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _generate_run_id() -> str:
    """Generate unique run ID with timestamp and random suffix."""
    import secrets

    return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{secrets.token_hex(2)}"


# =============================================================================
# SECTION DEFAULTS - Single source of truth for default values
# =============================================================================

# Import canonical constants
from src.core.constants import (  # noqa: E402
    CANONICAL_TIMEFRAMES,
    DEFAULT_EMBARGO_BARS,
    DEFAULT_HORIZONS,
    DEFAULT_MTF_TIMEFRAMES,
    DEFAULT_SPLIT_RATIOS,
)

# Extended timeframes (beyond canonical 9)
_EXTENDED_TIMEFRAMES: list[str] = ["240min", "1440min"]

# Horizon defaults
_SUPPORTED_HORIZONS: list[int] = [1, 5, 10, 15, 20, 30, 60, 120]


# =============================================================================
# SECTION DATACLASSES
# =============================================================================


@dataclass
class TimeframesSection:
    """Timeframe configuration section."""

    default_primary: str = "5min"
    canonical_ladder: list[str] = field(default_factory=lambda: list(CANONICAL_TIMEFRAMES))
    extended: list[str] = field(default_factory=lambda: list(_EXTENDED_TIMEFRAMES))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeframesSection:
        return cls(
            default_primary=data.get("default_primary", "5min"),
            canonical_ladder=data.get("canonical_ladder") or list(CANONICAL_TIMEFRAMES),
            extended=data.get("extended") or list(_EXTENDED_TIMEFRAMES),
        )


@dataclass
class SplitsSection:
    """Train/val/test split configuration."""

    train: float = field(default_factory=lambda: DEFAULT_SPLIT_RATIOS["train"])
    val: float = field(default_factory=lambda: DEFAULT_SPLIT_RATIOS["val"])
    test: float = field(default_factory=lambda: DEFAULT_SPLIT_RATIOS["test"])

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.3f} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitsSection:
        return cls(
            train=data.get("train", DEFAULT_SPLIT_RATIOS["train"]),
            val=data.get("val", DEFAULT_SPLIT_RATIOS["val"]),
            test=data.get("test", DEFAULT_SPLIT_RATIOS["test"]),
        )


@dataclass
class PurgeEmbargoSection:
    """Purge and embargo configuration."""

    purge_multiplier: float = 3.0
    embargo_time_minutes: int = 7200  # 5 days
    min_embargo_bars: int = DEFAULT_EMBARGO_BARS  # Legacy, for backward compat

    def compute_purge_bars(self, max_horizon: int) -> int:
        """Compute purge bars based on max horizon."""
        return int(max_horizon * self.purge_multiplier)

    def compute_embargo_bars(self, timeframe_minutes: int) -> int:
        """Compute embargo bars based on timeframe."""
        if timeframe_minutes <= 0:
            raise ValueError(f"timeframe_minutes must be positive, got {timeframe_minutes}")
        return max(1, self.embargo_time_minutes // timeframe_minutes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PurgeEmbargoSection:
        return cls(
            purge_multiplier=data.get("purge_multiplier", 3.0),
            embargo_time_minutes=data.get("embargo_time_minutes", 7200),
            min_embargo_bars=data.get("min_embargo_bars", DEFAULT_EMBARGO_BARS),
        )


@dataclass
class HorizonsSection:
    """Horizon configuration."""

    supported: list[int] = field(default_factory=lambda: list(_SUPPORTED_HORIZONS))
    active: list[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    default: list[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))

    def __post_init__(self) -> None:
        # Validate active is subset of supported
        for h in self.active:
            if h not in self.supported:
                raise ValueError(f"Active horizon {h} not in supported horizons {self.supported}")

    @property
    def max_horizon(self) -> int:
        """Get the maximum active horizon."""
        return max(self.active) if self.active else 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HorizonsSection:
        return cls(
            supported=data.get("supported") or list(_SUPPORTED_HORIZONS),
            active=data.get("active") or list(DEFAULT_HORIZONS),
            default=data.get("default") or list(DEFAULT_HORIZONS),
        )


@dataclass
class FeatureSelectionSection:
    """Feature selection configuration."""

    enabled: bool = True
    method: str = "mda"  # mda, mdi, hybrid
    cv_splits: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSelectionSection:
        return cls(
            enabled=data.get("enabled", True),
            method=data.get("method", "mda"),
            cv_splits=data.get("cv_splits", 5),
        )


@dataclass
class FeatureGenerationSection:
    """Feature generation configuration."""

    default: str = "full"
    modes: dict[str, str] = field(
        default_factory=lambda: {
            "full": "Generate complete feature set (~180 features)",
            "minimal": "Generate only essential features (~50 features)",
            "custom": "Generate custom feature set (requires feature_toggles)",
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureGenerationSection:
        default_modes = {
            "full": "Generate complete feature set (~180 features)",
            "minimal": "Generate only essential features (~50 features)",
            "custom": "Generate custom feature set (requires feature_toggles)",
        }
        return cls(
            default=data.get("default", "full"),
            modes=data.get("modes") or default_modes,
        )


@dataclass
class FeaturesSection:
    """Feature engineering configuration."""

    sma_periods: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    ema_periods: list[int] = field(default_factory=lambda: [9, 21, 50])
    atr_periods: list[int] = field(default_factory=lambda: [7, 14, 21])
    rsi_period: int = 14
    macd: dict[str, int] = field(default_factory=lambda: {"fast": 12, "slow": 26, "signal": 9})
    bollinger: dict[str, float | int] = field(default_factory=lambda: {"period": 20, "std": 2.0})
    selection: FeatureSelectionSection = field(default_factory=FeatureSelectionSection)
    generation: FeatureGenerationSection = field(default_factory=FeatureGenerationSection)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeaturesSection:
        default_sma_periods = [10, 20, 50, 100, 200]
        default_ema_periods = [9, 21, 50]
        default_atr_periods = [7, 14, 21]
        default_macd: dict[str, int] = {"fast": 12, "slow": 26, "signal": 9}
        default_bollinger: dict[str, float | int] = {"period": 20, "std": 2.0}
        return cls(
            sma_periods=data.get("sma_periods") or default_sma_periods,
            ema_periods=data.get("ema_periods") or default_ema_periods,
            atr_periods=data.get("atr_periods") or default_atr_periods,
            rsi_period=data.get("rsi_period", 14),
            macd=data.get("macd") or default_macd,
            bollinger=data.get("bollinger") or default_bollinger,
            selection=FeatureSelectionSection.from_dict(data.get("selection", {})),
            generation=FeatureGenerationSection.from_dict(data.get("generation", {})),
        )


@dataclass
class MTFSection:
    """Multi-timeframe configuration."""

    enabled: bool = True
    default_mode: str = "indicators"  # indicators, bars, both
    default_timeframes: list[str] = field(default_factory=lambda: list(DEFAULT_MTF_TIMEFRAMES))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MTFSection:
        return cls(
            enabled=data.get("enabled", True),
            default_mode=data.get("default_mode", "indicators"),
            default_timeframes=data.get("default_timeframes") or list(DEFAULT_MTF_TIMEFRAMES),
        )


@dataclass
class TrainingSection:
    """Model training configuration."""

    sequence_length: int = 60
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 15
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingSection:
        return cls(
            sequence_length=data.get("sequence_length", 60),
            batch_size=data.get("batch_size", 256),
            max_epochs=data.get("max_epochs", 100),
            early_stopping_patience=data.get("early_stopping_patience", 15),
            device=data.get("device", "auto"),
            mixed_precision=data.get("mixed_precision", True),
            num_workers=data.get("num_workers", 4),
            pin_memory=data.get("pin_memory", True),
        )


@dataclass
class CalibrationSection:
    """Calibration configuration."""

    enabled: bool = True
    method: str = "auto"  # auto, isotonic, sigmoid, none

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationSection:
        return cls(
            enabled=data.get("enabled", True),
            method=data.get("method", "auto"),
        )


@dataclass
class GASection:
    """Genetic algorithm configuration."""

    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 5
    safe_mode: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GASection:
        return cls(
            population_size=data.get("population_size", 50),
            generations=data.get("generations", 100),
            crossover_rate=data.get("crossover_rate", 0.8),
            mutation_rate=data.get("mutation_rate", 0.1),
            elite_size=data.get("elite_size", 5),
            safe_mode=data.get("safe_mode", True),
        )


@dataclass
class OptunaSection:
    """Optuna configuration."""

    n_trials: int = 100
    timeout: int = 3600  # 1 hour
    n_jobs: int = -1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptunaSection:
        return cls(
            n_trials=data.get("n_trials", 100),
            timeout=data.get("timeout", 3600),
            n_jobs=data.get("n_jobs", -1),
        )


@dataclass
class OptimizationSection:
    """Optimization configuration (GA + Optuna)."""

    ga: GASection = field(default_factory=GASection)
    optuna: OptunaSection = field(default_factory=OptunaSection)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationSection:
        return cls(
            ga=GASection.from_dict(data.get("ga", {})),
            optuna=OptunaSection.from_dict(data.get("optuna", {})),
        )


@dataclass
class CrossValidationSection:
    """Cross-validation configuration."""

    n_splits: int = 5
    purge_multiplier: float = 3.0
    embargo_time_minutes: int = 7200

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossValidationSection:
        return cls(
            n_splits=data.get("n_splits", 5),
            purge_multiplier=data.get("purge_multiplier", 3.0),
            embargo_time_minutes=data.get("embargo_time_minutes", 7200),
        )


@dataclass
class ProcessingSection:
    """Processing configuration."""

    n_jobs: int = -1
    allow_batch_symbols: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingSection:
        return cls(
            n_jobs=data.get("n_jobs", -1),
            allow_batch_symbols=data.get("allow_batch_symbols", False),
        )


@dataclass
class ScalerSection:
    """Scaler configuration."""

    default: str = "robust"  # robust, standard, minmax, none

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScalerSection:
        return cls(default=data.get("default", "robust"))


@dataclass
class TrackingSection:
    """Experiment tracking configuration."""

    enabled: bool = True
    backend: str = "local"  # local, mlflow, wandb, disabled
    uri: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackingSection:
        return cls(
            enabled=data.get("enabled", True),
            backend=data.get("backend", "local"),
            uri=data.get("uri"),
            tags=data.get("tags", {}),
        )


@dataclass
class OOMRecoverySection:
    """Out-of-memory recovery configuration."""

    enabled: bool = True
    max_retries: int = 3
    batch_reduction_factor: float = 0.5
    min_batch_size: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OOMRecoverySection:
        return cls(
            enabled=data.get("enabled", True),
            max_retries=data.get("max_retries", 3),
            batch_reduction_factor=data.get("batch_reduction_factor", 0.5),
            min_batch_size=data.get("min_batch_size", 8),
        )


@dataclass
class ModelConfigSection:
    """
    Per-model configuration section (Phase 1 SNwH).

    This section provides family-level defaults and per-model overrides
    for timeframe, MTF mode, and feature mode settings.
    """

    # Default configurations by family
    # NOTE: primary_timeframe, mtf_mode, feature_mode are NOT included here
    # because they are model-specific and should come from ModelContract.
    # This section is for settings that genuinely vary by family.
    defaults: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "boosting": {},
            "neural": {},
            "transformer": {},
            "classical": {},
            "ensemble": {},
            "meta_learner": {},
        }
    )

    # Per-model overrides (model_name -> config dict)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_model_config(self, model_name: str, model_family: str) -> dict[str, Any]:
        """
        Get merged configuration for a model.

        Priority: override > family default

        Args:
            model_name: Name of the model
            model_family: Family of the model

        Returns:
            Merged configuration dictionary
        """
        config = dict(self.defaults.get(model_family, {}))
        if model_name in self.overrides:
            config.update(self.overrides[model_name])
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfigSection:
        default_defaults: dict[str, dict[str, Any]] = {
            "boosting": {},
            "neural": {},
            "transformer": {},
            "classical": {},
            "ensemble": {},
            "meta_learner": {},
        }
        return cls(
            defaults=data.get("defaults") or default_defaults,
            overrides=data.get("overrides", {}),
        )


# =============================================================================
# UNIFIED CONFIG
# =============================================================================


@dataclass
class UnifiedConfig:
    """
    Unified configuration for the entire ML pipeline.

    Consolidates all configuration from:
    - GlobalConfig (from global.yaml)
    - MLConfig
    - TrainerConfig
    - PipelineConfig
    - HorizonConfig

    This is the single source of truth for all configuration values.
    """

    # Run identification
    run_id: str = field(default_factory=_generate_run_id)
    description: str = "ML Pipeline Run"

    # Seed
    random_seed: int = 42

    # Data configuration
    symbol: str = "MES"
    symbols: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None

    # Sections (grouped configuration)
    timeframes: TimeframesSection = field(default_factory=TimeframesSection)
    splits: SplitsSection = field(default_factory=SplitsSection)
    purge_embargo: PurgeEmbargoSection = field(default_factory=PurgeEmbargoSection)
    horizons: HorizonsSection = field(default_factory=HorizonsSection)
    features: FeaturesSection = field(default_factory=FeaturesSection)
    mtf: MTFSection = field(default_factory=MTFSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    calibration: CalibrationSection = field(default_factory=CalibrationSection)
    optimization: OptimizationSection = field(default_factory=OptimizationSection)
    cross_validation: CrossValidationSection = field(default_factory=CrossValidationSection)
    processing: ProcessingSection = field(default_factory=ProcessingSection)
    scaler: ScalerSection = field(default_factory=ScalerSection)
    tracking: TrackingSection = field(default_factory=TrackingSection)
    oom_recovery: OOMRecoverySection = field(default_factory=OOMRecoverySection)

    # Phase 1 SNwH: Per-model configuration
    model_config: ModelConfigSection = field(default_factory=ModelConfigSection)

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Ensure paths are Path objects
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        # Ensure symbols list
        if self.symbols is None:
            self.symbols = [self.symbol]
        elif self.symbol not in self.symbols:
            self.symbols = [self.symbol] + list(self.symbols)

    @classmethod
    def from_dict(cls, data: dict[str, Any], validate: bool = True) -> UnifiedConfig:
        """
        Create UnifiedConfig from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Configuration dictionary
        validate : bool, default True
            Whether to validate the configuration

        Returns
        -------
        UnifiedConfig
        """
        # Optionally coerce types
        data, _ = coerce_types(data, warn_on_coercion=True)

        # Optionally validate
        if validate:
            result = validate_config(data, strict=False)
            if not result.is_valid:
                logger.warning(f"Configuration has {len(result.errors)} validation error(s)")
                for error in result.errors:
                    logger.warning(f"  - {error}")

        return cls(
            run_id=data.get("run_id", _generate_run_id()),
            description=data.get("description", "ML Pipeline Run"),
            random_seed=data.get("random_seed", 42),
            symbol=data.get("symbol", "MES"),
            symbols=data.get("symbols"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            timeframes=TimeframesSection.from_dict(data.get("timeframes", {})),
            splits=SplitsSection.from_dict(data.get("splits", {})),
            purge_embargo=PurgeEmbargoSection.from_dict(data.get("purge_embargo", {})),
            horizons=HorizonsSection.from_dict(data.get("horizons", {})),
            features=FeaturesSection.from_dict(data.get("features", {})),
            mtf=MTFSection.from_dict(data.get("mtf", {})),
            training=TrainingSection.from_dict(data.get("training", {})),
            calibration=CalibrationSection.from_dict(data.get("calibration", {})),
            optimization=OptimizationSection.from_dict(data.get("optimization", {})),
            cross_validation=CrossValidationSection.from_dict(data.get("cross_validation", {})),
            processing=ProcessingSection.from_dict(data.get("processing", {})),
            scaler=ScalerSection.from_dict(data.get("scaler", {})),
            tracking=TrackingSection.from_dict(data.get("tracking", {})),
            oom_recovery=OOMRecoverySection.from_dict(data.get("oom_recovery", {})),
            model_config=ModelConfigSection.from_dict(data.get("model_config", {})),
            output_dir=Path(data.get("output_dir", "experiments/runs")),
            data_dir=Path(data.get("data_dir", "data")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path, validate: bool = True) -> UnifiedConfig:
        """
        Load UnifiedConfig from a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML file
        validate : bool, default True
            Whether to validate the configuration

        Returns
        -------
        UnifiedConfig
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls.from_dict(data, validate=validate)

    @classmethod
    def load_default(cls, validate: bool = True) -> UnifiedConfig:
        """
        Load the default configuration from config/global.yaml.

        Returns
        -------
        UnifiedConfig
        """
        default_path = Path(__file__).parent.parent.parent / "config" / "global.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path, validate=validate)
        else:
            logger.warning(f"Default config not found at {default_path}, using hardcoded defaults")
            return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "description": self.description,
            "random_seed": self.random_seed,
            "symbol": self.symbol,
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "timeframes": {
                "default_primary": self.timeframes.default_primary,
                "canonical_ladder": self.timeframes.canonical_ladder,
                "extended": self.timeframes.extended,
            },
            "splits": {
                "train": self.splits.train,
                "val": self.splits.val,
                "test": self.splits.test,
            },
            "purge_embargo": {
                "purge_multiplier": self.purge_embargo.purge_multiplier,
                "embargo_time_minutes": self.purge_embargo.embargo_time_minutes,
                "min_embargo_bars": self.purge_embargo.min_embargo_bars,
            },
            "horizons": {
                "supported": self.horizons.supported,
                "active": self.horizons.active,
                "default": self.horizons.default,
            },
            "features": {
                "sma_periods": self.features.sma_periods,
                "ema_periods": self.features.ema_periods,
                "atr_periods": self.features.atr_periods,
                "rsi_period": self.features.rsi_period,
                "macd": self.features.macd,
                "bollinger": self.features.bollinger,
                "selection": {
                    "enabled": self.features.selection.enabled,
                    "method": self.features.selection.method,
                    "cv_splits": self.features.selection.cv_splits,
                },
                "generation": {
                    "default": self.features.generation.default,
                    "modes": self.features.generation.modes,
                },
            },
            "mtf": {
                "enabled": self.mtf.enabled,
                "default_mode": self.mtf.default_mode,
                "default_timeframes": self.mtf.default_timeframes,
            },
            "training": {
                "sequence_length": self.training.sequence_length,
                "batch_size": self.training.batch_size,
                "max_epochs": self.training.max_epochs,
                "early_stopping_patience": self.training.early_stopping_patience,
                "device": self.training.device,
                "mixed_precision": self.training.mixed_precision,
                "num_workers": self.training.num_workers,
                "pin_memory": self.training.pin_memory,
            },
            "calibration": {
                "enabled": self.calibration.enabled,
                "method": self.calibration.method,
            },
            "optimization": {
                "ga": {
                    "population_size": self.optimization.ga.population_size,
                    "generations": self.optimization.ga.generations,
                    "crossover_rate": self.optimization.ga.crossover_rate,
                    "mutation_rate": self.optimization.ga.mutation_rate,
                    "elite_size": self.optimization.ga.elite_size,
                    "safe_mode": self.optimization.ga.safe_mode,
                },
                "optuna": {
                    "n_trials": self.optimization.optuna.n_trials,
                    "timeout": self.optimization.optuna.timeout,
                    "n_jobs": self.optimization.optuna.n_jobs,
                },
            },
            "cross_validation": {
                "n_splits": self.cross_validation.n_splits,
                "purge_multiplier": self.cross_validation.purge_multiplier,
                "embargo_time_minutes": self.cross_validation.embargo_time_minutes,
            },
            "processing": {
                "n_jobs": self.processing.n_jobs,
                "allow_batch_symbols": self.processing.allow_batch_symbols,
            },
            "scaler": {
                "default": self.scaler.default,
            },
            "tracking": {
                "enabled": self.tracking.enabled,
                "backend": self.tracking.backend,
                "uri": self.tracking.uri,
                "tags": self.tracking.tags,
            },
            "oom_recovery": {
                "enabled": self.oom_recovery.enabled,
                "max_retries": self.oom_recovery.max_retries,
                "batch_reduction_factor": self.oom_recovery.batch_reduction_factor,
                "min_batch_size": self.oom_recovery.min_batch_size,
            },
            "model_config": {
                "defaults": self.model_config.defaults,
                "overrides": self.model_config.overrides,
            },
            "output_dir": str(self.output_dir),
            "data_dir": str(self.data_dir),
        }

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> ValidationResult:
        """Validate the configuration."""
        return validate_config(self.to_dict())

    def validate_strict(self) -> None:
        """Validate the configuration, raising on errors."""
        result = self.validate()
        if not result.is_valid:
            raise ConfigValidationError(result)

    # =========================================================================
    # BACKWARD COMPATIBILITY: Convert to legacy config types
    # =========================================================================

    def to_trainer_config(
        self,
        model_name: str,
        horizon: int | None = None,
        feature_set: str = "boosting_optimal",
        **overrides: Any,
    ) -> Any:
        """
        Convert to TrainerConfig for backward compatibility.

        Parameters
        ----------
        model_name : str
            Name of the model to train
        horizon : int, optional
            Training horizon (defaults to max active horizon)
        feature_set : str
            Feature set to use
        **overrides
            Additional overrides for TrainerConfig

        Returns
        -------
        TrainerConfig
        """
        from src.models.config.trainer_config import TrainerConfig

        horizon = horizon or self.horizons.max_horizon

        return TrainerConfig(
            model_name=model_name,
            horizon=horizon,
            feature_set=feature_set,
            pipeline_run_id=self.run_id,
            sequence_length=self.training.sequence_length,
            batch_size=self.training.batch_size,
            max_epochs=self.training.max_epochs,
            early_stopping_patience=self.training.early_stopping_patience,
            random_seed=self.random_seed,
            output_dir=self.output_dir,
            device=self.training.device,
            mixed_precision=self.training.mixed_precision,
            num_workers=self.training.num_workers,
            pin_memory=self.training.pin_memory,
            use_calibration=self.calibration.enabled,
            calibration_method=self.calibration.method,
            use_feature_selection=self.features.selection.enabled,
            feature_selection_method=self.features.selection.method,
            feature_selection_cv_splits=self.features.selection.cv_splits,
            tracking_enabled=self.tracking.enabled,
            tracking_backend=self.tracking.backend,
            tracking_uri=self.tracking.uri,
            tracking_tags=self.tracking.tags,
            oom_recovery_enabled=self.oom_recovery.enabled,
            oom_max_retries=self.oom_recovery.max_retries,
            oom_batch_reduction_factor=self.oom_recovery.batch_reduction_factor,
            oom_min_batch_size=self.oom_recovery.min_batch_size,
            **overrides,
        )

    def get_trainer_config_for_model(
        self,
        model_name: str,
        horizon: int | None = None,
        **overrides: Any,
    ) -> Any:
        """
        Get TrainerConfig for a specific model using contract-based defaults.

        This is the SNwH-compatible way to create TrainerConfig. It uses the
        model's contract to set appropriate defaults for timeframe, MTF mode,
        and other settings.

        Args:
            model_name: Name of the model (e.g., "xgboost", "lstm", "patchtst")
            horizon: Training horizon (defaults to max active horizon)
            **overrides: Additional overrides for any TrainerConfig field

        Returns:
            TrainerConfig with contract-based defaults

        Example:
            config = unified_config.get_trainer_config_for_model("lstm", horizon=20)
        """
        from src.core.contracts import get_model_contract
        from src.models.config.trainer_config import TrainerConfig

        horizon = horizon or self.horizons.max_horizon
        contract = get_model_contract(model_name)

        # Get family-level and model-level overrides from model_config section
        model_overrides = self.model_config.get_model_config(model_name, contract.model_family)

        # Merge: contract defaults -> model_config overrides -> user overrides
        merged_overrides = {
            # Base settings from UnifiedConfig
            "batch_size": self.training.batch_size,
            "max_epochs": self.training.max_epochs,
            "early_stopping_patience": self.training.early_stopping_patience,
            "random_seed": self.random_seed,
            "output_dir": self.output_dir,
            "device": self.training.device,
            "mixed_precision": self.training.mixed_precision,
            "num_workers": self.training.num_workers,
            "pin_memory": self.training.pin_memory,
            "use_calibration": self.calibration.enabled,
            "calibration_method": self.calibration.method,
            "use_feature_selection": self.features.selection.enabled,
            "feature_selection_method": self.features.selection.method,
            "feature_selection_cv_splits": self.features.selection.cv_splits,
            "tracking_enabled": self.tracking.enabled,
            "tracking_backend": self.tracking.backend,
            "tracking_uri": self.tracking.uri,
            "tracking_tags": self.tracking.tags,
            "oom_recovery_enabled": self.oom_recovery.enabled,
            "oom_max_retries": self.oom_recovery.max_retries,
            "oom_batch_reduction_factor": self.oom_recovery.batch_reduction_factor,
            "oom_min_batch_size": self.oom_recovery.min_batch_size,
        }

        # Apply model_config section overrides
        merged_overrides.update(model_overrides)

        # Apply user-provided overrides
        merged_overrides.update(overrides)

        return TrainerConfig.from_model_contract(
            model_name=model_name,
            horizon=horizon,
            **merged_overrides,
        )

    def to_pipeline_config(self, **overrides: Any) -> Any:
        """
        Convert to PipelineConfig for backward compatibility.

        Parameters
        ----------
        **overrides
            Additional overrides for PipelineConfig

        Returns
        -------
        PipelineConfig
        """
        # Parse timeframe minutes for embargo calculation
        from src.core.common.timeframes import get_timeframe_minutes
        from src.data.pipeline.data_config import PipelineConfig

        tf_minutes = get_timeframe_minutes(self.timeframes.default_primary)
        purge_bars = self.purge_embargo.compute_purge_bars(self.horizons.max_horizon)
        embargo_bars = self.purge_embargo.compute_embargo_bars(tf_minutes)

        return PipelineConfig(
            run_id=self.run_id,
            description=self.description,
            symbols=self.symbols or [self.symbol],
            start_date=self.start_date,
            end_date=self.end_date,
            target_timeframe=self.timeframes.default_primary,
            feature_generation=self.features.generation.default,
            sma_periods=self.features.sma_periods,
            ema_periods=self.features.ema_periods,
            atr_periods=self.features.atr_periods,
            rsi_period=self.features.rsi_period,
            macd_params=self.features.macd,
            bb_period=int(self.features.bollinger.get("period", 20)),
            bb_std=self.features.bollinger.get("std", 2.0),
            mtf_timeframes=self.mtf.default_timeframes,
            mtf_mode=self.mtf.default_mode,
            label_horizons=self.horizons.active,
            train_ratio=self.splits.train,
            val_ratio=self.splits.val,
            test_ratio=self.splits.test,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            auto_scale_purge_embargo=False,  # We already computed them
            ga_population_size=self.optimization.ga.population_size,
            ga_generations=self.optimization.ga.generations,
            ga_crossover_rate=self.optimization.ga.crossover_rate,
            ga_mutation_rate=self.optimization.ga.mutation_rate,
            ga_elite_size=self.optimization.ga.elite_size,
            ga_safe_mode=self.optimization.ga.safe_mode,
            n_jobs=self.processing.n_jobs,
            random_seed=self.random_seed,
            allow_batch_symbols=self.processing.allow_batch_symbols,
            scaler_type=self.scaler.default,
            **overrides,
        )

    def to_ml_config(self, models: list[str] | None = None, **overrides: Any) -> Any:
        """
        Convert to MLConfig for backward compatibility.

        DEPRECATED: MLConfig from src.ml_pipeline.config no longer exists.
        This method now returns a dictionary with the same structure for
        backwards compatibility with any code that might use this API.

        Parameters
        ----------
        models : list[str], optional
            List of model names
        **overrides
            Additional overrides for the config

        Returns
        -------
        dict[str, Any]
            Configuration dictionary (MLConfig is deprecated)
        """
        import warnings

        warnings.warn(
            "to_ml_config() is deprecated. MLConfig no longer exists. "
            "Use to_pipeline_config() or to_trainer_config() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        config_dict = {
            "run_id": self.run_id,
            "description": self.description,
            "symbol": self.symbol,
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "target_timeframe": self.timeframes.default_primary,
            "horizons": self.horizons.active,
            "train_ratio": self.splits.train,
            "val_ratio": self.splits.val,
            "test_ratio": self.splits.test,
            "sequence_length": self.training.sequence_length,
            "batch_size": self.training.batch_size,
            "max_epochs": self.training.max_epochs,
            "early_stopping_patience": self.training.early_stopping_patience,
            "random_seed": self.random_seed,
            "device": self.training.device,
            "mixed_precision": self.training.mixed_precision,
            "num_workers": self.training.num_workers,
            "pin_memory": self.training.pin_memory,
            "use_calibration": self.calibration.enabled,
            "calibration_method": self.calibration.method,
            "output_dir": str(self.output_dir),
            "data_dir": str(self.data_dir),
            "tracking_enabled": self.tracking.enabled,
            "tracking_backend": self.tracking.backend,
            "tracking_uri": self.tracking.uri,
            "tracking_tags": self.tracking.tags,
            "cv_splits": self.cross_validation.n_splits,
            "n_jobs": self.processing.n_jobs,
            "models": models or [],
        }
        config_dict.update(overrides)
        return config_dict

    def to_horizon_config(self) -> Any:
        """
        Convert to HorizonConfig for backward compatibility.

        Returns
        -------
        HorizonConfig
        """
        from src.core.common.horizon_config import HorizonConfig

        return HorizonConfig(
            horizons=self.horizons.active,
            source_timeframe=self.timeframes.default_primary,
            auto_scale_purge_embargo=True,
            purge_multiplier=self.purge_embargo.purge_multiplier,
            # Legacy: 72.0 multiplier for embargo period calculation (horizon * 72 = embargo bars)
            embargo_multiplier=72.0,
        )


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_unified_config: UnifiedConfig | None = None


def get_unified_config() -> UnifiedConfig:
    """
    Get the global UnifiedConfig singleton.

    Loads from config/global.yaml on first access.
    """
    global _unified_config
    if _unified_config is None:
        _unified_config = UnifiedConfig.load_default()
    return _unified_config


def set_unified_config(config: UnifiedConfig) -> None:
    """Set the global UnifiedConfig singleton."""
    global _unified_config
    _unified_config = config


def reset_unified_config() -> None:
    """Reset the global UnifiedConfig singleton."""
    global _unified_config
    _unified_config = None


__all__ = [
    # Main class
    "UnifiedConfig",
    # Section classes
    "TimeframesSection",
    "SplitsSection",
    "PurgeEmbargoSection",
    "HorizonsSection",
    "FeaturesSection",
    "FeatureSelectionSection",
    "FeatureGenerationSection",
    "MTFSection",
    "TrainingSection",
    "CalibrationSection",
    "OptimizationSection",
    "GASection",
    "OptunaSection",
    "CrossValidationSection",
    "ProcessingSection",
    "ScalerSection",
    "TrackingSection",
    "OOMRecoverySection",
    "ModelConfigSection",  # Phase 1 SNwH
    # Singleton functions
    "get_unified_config",
    "set_unified_config",
    "reset_unified_config",
]
