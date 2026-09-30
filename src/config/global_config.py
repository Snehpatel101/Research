"""
GlobalConfig - typed view of src/config/global.yaml (shipped as package data).

Holds only sections that code actually reads: via ``get_config_value()``
(TrainerConfig field defaults) or ``get_global_config()``
(horizon lists). Keep the YAML and these dataclasses in lock-step: a section
here that nothing reads is a "settable but ignored" knob.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    ema_periods: list[int]
    atr_periods: list[int]
    rsi_period: int
    macd: dict[str, int]
    bollinger: dict[str, float | int]
    selection: FeatureSelectionConfig
    generation: FeatureGenerationConfig


@dataclass
class TrainingConfig:
    sequence_length: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    device: str
    mixed_precision: bool
    num_workers: int | None  # None = auto (4 on CUDA, 0 on CPU)
    pin_memory: bool | None  # None = auto (pin only on CUDA)


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
class OptimizationConfig:
    ga: GAConfig


@dataclass
class ProcessingConfig:
    n_jobs: int
    allow_batch_symbols: bool


@dataclass
class ScalerConfig:
    default: str


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
    horizons: HorizonsConfig
    features: FeaturesConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    optimization: OptimizationConfig
    processing: ProcessingConfig
    scaler: ScalerConfig
    oom_recovery: OOMRecoveryConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlobalConfig:
        return cls(
            random_seed=data["random_seed"],
            timeframes=TimeframeConfig(**data["timeframes"]),
            splits=SplitConfig(**data["splits"]),
            horizons=HorizonsConfig(**data["horizons"]),
            features=FeaturesConfig(
                ema_periods=data["features"]["ema_periods"],
                atr_periods=data["features"]["atr_periods"],
                rsi_period=data["features"]["rsi_period"],
                macd=data["features"]["macd"],
                bollinger=data["features"]["bollinger"],
                selection=FeatureSelectionConfig(**data["features"]["selection"]),
                generation=FeatureGenerationConfig(**data["features"]["generation"]),
            ),
            training=TrainingConfig(**data["training"]),
            calibration=CalibrationConfig(**data["calibration"]),
            optimization=OptimizationConfig(
                ga=GAConfig(**data["optimization"]["ga"]),
            ),
            processing=ProcessingConfig(**data["processing"]),
            scaler=ScalerConfig(**data["scaler"]),
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
            "horizons": {
                "supported": self.horizons.supported,
                "active": self.horizons.active,
                "default": self.horizons.default,
            },
            "features": {
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
            },
            "processing": {
                "n_jobs": self.processing.n_jobs,
                "allow_batch_symbols": self.processing.allow_batch_symbols,
            },
            "scaler": {
                "default": self.scaler.default,
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
        path = Path(__file__).parent / "global.yaml"
    return GlobalConfig.from_yaml(path)


_global_config: GlobalConfig | None = None


def get_global_config() -> GlobalConfig:
    global _global_config
    if _global_config is None:
        _global_config = load_global_config()
    return _global_config
