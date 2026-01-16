from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TimeframeConfig:
    default_primary: str
    canonical_ladder: list[str]
    extended: list[str]


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float


@dataclass
class PurgeEmbargoConfig:
    purge_multiplier: float
    embargo_time_minutes: int
    min_embargo_bars: int


@dataclass
class HorizonsConfig:
    supported: list[int]
    active: list[int]
    default: list[int]


@dataclass
class FeatureSelectionConfig:
    enabled: bool
    method: str
    cv_splits: int


@dataclass
class FeatureGenerationConfig:
    default: str
    modes: dict[str, str]


@dataclass
class FeaturesConfig:
    sma_periods: list[int]
    ema_periods: list[int]
    atr_periods: list[int]
    rsi_period: int
    macd: dict[str, int]
    bollinger: dict[str, float | int]
    selection: FeatureSelectionConfig
    generation: FeatureGenerationConfig


@dataclass
class MTFConfig:
    default_mode: str
    default_timeframes: list[str]
    enabled: bool


@dataclass
class TrainingConfig:
    sequence_length: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    device: str
    mixed_precision: bool
    num_workers: int
    pin_memory: bool


@dataclass
class CalibrationConfig:
    enabled: bool
    method: str


@dataclass
class GAConfig:
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    elite_size: int
    safe_mode: bool


@dataclass
class OptunaConfig:
    n_trials: int
    timeout: int
    n_jobs: int


@dataclass
class OptimizationConfig:
    ga: GAConfig
    optuna: OptunaConfig


@dataclass
class CrossValidationConfig:
    n_splits: int
    purge_multiplier: float
    embargo_time_minutes: int


@dataclass
class ProcessingConfig:
    n_jobs: int
    allow_batch_symbols: bool


@dataclass
class ScalerConfig:
    default: str


@dataclass
class TrackingConfig:
    enabled: bool
    backend: str


@dataclass
class OOMRecoveryConfig:
    enabled: bool
    max_retries: int
    batch_reduction_factor: float
    min_batch_size: int


@dataclass
class GlobalConfig:
    random_seed: int
    timeframes: TimeframeConfig
    splits: SplitConfig
    purge_embargo: PurgeEmbargoConfig
    horizons: HorizonsConfig
    features: FeaturesConfig
    mtf: MTFConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    optimization: OptimizationConfig
    cross_validation: CrossValidationConfig
    processing: ProcessingConfig
    scaler: ScalerConfig
    tracking: TrackingConfig
    oom_recovery: OOMRecoveryConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlobalConfig:
        return cls(
            random_seed=data["random_seed"],
            timeframes=TimeframeConfig(**data["timeframes"]),
            splits=SplitConfig(**data["splits"]),
            purge_embargo=PurgeEmbargoConfig(**data["purge_embargo"]),
            horizons=HorizonsConfig(**data["horizons"]),
            features=FeaturesConfig(
                sma_periods=data["features"]["sma_periods"],
                ema_periods=data["features"]["ema_periods"],
                atr_periods=data["features"]["atr_periods"],
                rsi_period=data["features"]["rsi_period"],
                macd=data["features"]["macd"],
                bollinger=data["features"]["bollinger"],
                selection=FeatureSelectionConfig(**data["features"]["selection"]),
                generation=FeatureGenerationConfig(**data["features"]["generation"]),
            ),
            mtf=MTFConfig(**data["mtf"]),
            training=TrainingConfig(**data["training"]),
            calibration=CalibrationConfig(**data["calibration"]),
            optimization=OptimizationConfig(
                ga=GAConfig(**data["optimization"]["ga"]),
                optuna=OptunaConfig(**data["optimization"]["optuna"]),
            ),
            cross_validation=CrossValidationConfig(**data["cross_validation"]),
            processing=ProcessingConfig(**data["processing"]),
            scaler=ScalerConfig(**data["scaler"]),
            tracking=TrackingConfig(**data["tracking"]),
            oom_recovery=OOMRecoveryConfig(**data["oom_recovery"]),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> GlobalConfig:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "random_seed": self.random_seed,
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
                "default_mode": self.mtf.default_mode,
                "default_timeframes": self.mtf.default_timeframes,
                "enabled": self.mtf.enabled,
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
            },
            "oom_recovery": {
                "enabled": self.oom_recovery.enabled,
                "max_retries": self.oom_recovery.max_retries,
                "batch_reduction_factor": self.oom_recovery.batch_reduction_factor,
                "min_batch_size": self.oom_recovery.min_batch_size,
            },
        }


def load_global_config(
    path: Path | str | None = None,
) -> GlobalConfig:
    if path is None:
        path = Path(__file__).parent.parent.parent / "config" / "global.yaml"
    return GlobalConfig.from_yaml(path)


_global_config: GlobalConfig | None = None


def get_global_config() -> GlobalConfig:
    global _global_config
    if _global_config is None:
        _global_config = load_global_config()
    return _global_config


def set_global_config(config: GlobalConfig) -> None:
    global _global_config
    _global_config = config
