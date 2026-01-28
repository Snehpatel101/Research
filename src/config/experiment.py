"""
ExperimentConfig - Single Source of Truth for ML Factory Experiments.

This module provides ExperimentConfig, which consolidates all configuration
needed for a complete ML Factory experiment run. It composes existing config
classes rather than duplicating fields.

ExperimentConfig is the top-level config used by MLFactory. It provides
conversion methods to legacy config formats for backward compatibility.

Example:
    from src.config.experiment import ExperimentConfig

    config = ExperimentConfig(
        name="mes_xgboost_experiment",
        symbol="MES",
        models=["xgboost", "lightgbm", "lstm"],
        horizons=[5, 10, 15, 20],
    )

    # Use with MLFactory
    from src.factory import MLFactory
    factory = MLFactory(config)
    result = factory.run()

    # Or convert to legacy formats
    pipeline_config = config.to_pipeline_config()
    trainer_config = config.to_trainer_config(model_name="xgboost")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.config.data import (
    FeatureConfig,
    LabelingConfig,
    MTFConfig,
    ScalerConfig,
    SequenceConfig,
    SplitConfig,
)
from src.config.inference import BacktestConfig, BundleConfig
from src.config.training import CalibrationConfig, CheckpointConfig, OptunaConfig

logger = logging.getLogger(__name__)


def _generate_run_id() -> str:
    """Generate unique run ID with timestamp."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# =============================================================================
# SUB-CONFIGS
# =============================================================================


@dataclass
class DataSection:
    """
    Data-related configuration section.

    Composes existing data config classes.
    """

    # Data source
    symbol: str = "MES"
    data_path: str | Path | None = None
    start_date: str | None = None
    end_date: str | None = None

    # Sub-configs
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    mtf: MTFConfig = field(default_factory=MTFConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)


@dataclass
class TrainingSection:
    """
    Training-related configuration section.

    Composes existing training config classes.
    """

    # Model selection
    models: list[str] = field(default_factory=lambda: ["xgboost"])
    horizons: list[int] = field(default_factory=lambda: [5, 10, 15, 20])

    # Training settings
    training_mode: str = "single_horizon"  # single_horizon, multi_horizon, oof_only
    cv_method: str = "purged_kfold"
    n_splits: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440

    # Sub-configs
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Device settings
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 15

    # Ensemble
    build_ensemble: bool = True
    meta_learner: str = "ridge_meta"


@dataclass
class EvaluationSection:
    """
    Evaluation-related configuration section.
    """

    run_backtest: bool = False
    compute_shap: bool = False
    generate_report: bool = True
    position_sizing: str = "fixed"


@dataclass
class BundlingSection:
    """
    Bundling-related configuration section.
    """

    create_bundle: bool = True
    bundle_format: str = "directory"  # directory, tar.gz
    include_oof: bool = True
    include_feature_importance: bool = True


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================


@dataclass
class ExperimentConfig:
    """
    Single source of truth for ML Factory experiment configuration.

    This is the top-level config class that composes all other configs.
    It's designed to be simple and intuitive while providing access to
    all configuration options through sub-configs.

    Attributes:
        name: Experiment name
        description: Experiment description
        run_id: Unique run identifier (auto-generated)
        output_dir: Output directory for artifacts
        random_seed: Random seed for reproducibility
        verbose: Logging verbosity (0=silent, 1=info, 2=debug)

        data: Data configuration section
        training: Training configuration section
        evaluation: Evaluation configuration section
        bundling: Bundling configuration section

    Example:
        config = ExperimentConfig(
            name="mes_xgboost_experiment",
            symbol="MES",
            models=["xgboost", "lightgbm"],
            horizons=[5, 10, 15, 20],
        )
    """

    # Core settings
    name: str = "ml_factory_experiment"
    description: str = ""
    run_id: str = field(default_factory=_generate_run_id)
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
    random_seed: int = 42
    verbose: int = 1

    # Configuration sections
    data: DataSection = field(default_factory=DataSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    bundling: BundlingSection = field(default_factory=BundlingSection)

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure paths are Path objects
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Create full output path with run_id
        self.output_dir = self.output_dir / self.run_id

    @property
    def symbol(self) -> str:
        """Convenience accessor for data.symbol."""
        return self.data.symbol

    @property
    def models(self) -> list[str]:
        """Convenience accessor for training.models."""
        return self.training.models

    @property
    def horizons(self) -> list[int]:
        """Convenience accessor for training.horizons."""
        return self.training.horizons

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """
        Create ExperimentConfig from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            ExperimentConfig instance
        """
        # Extract top-level fields
        config_dict = {
            "name": data.get("name", "ml_factory_experiment"),
            "description": data.get("description", ""),
            "run_id": data.get("run_id", _generate_run_id()),
            "output_dir": Path(data.get("output_dir", "experiments/runs")),
            "random_seed": data.get("random_seed", 42),
            "verbose": data.get("verbose", 1),
        }

        # Parse data section
        data_section_dict = data.get("data", {})
        config_dict["data"] = DataSection(
            symbol=data_section_dict.get("symbol", "MES"),
            data_path=data_section_dict.get("data_path"),
            start_date=data_section_dict.get("start_date"),
            end_date=data_section_dict.get("end_date"),
            features=FeatureConfig(**data_section_dict.get("features", {})),
            labeling=LabelingConfig(**data_section_dict.get("labeling", {})),
            scaler=ScalerConfig(**data_section_dict.get("scaler", {})),
            sequence=SequenceConfig(**data_section_dict.get("sequence", {})),
            mtf=MTFConfig(**data_section_dict.get("mtf", {})),
            splits=SplitConfig(**data_section_dict.get("splits", {})),
        )

        # Parse training section
        training_section_dict = data.get("training", {})
        config_dict["training"] = TrainingSection(
            models=training_section_dict.get("models", ["xgboost"]),
            horizons=training_section_dict.get("horizons", [5, 10, 15, 20]),
            training_mode=training_section_dict.get("training_mode", "single_horizon"),
            cv_method=training_section_dict.get("cv_method", "purged_kfold"),
            n_splits=training_section_dict.get("n_splits", 5),
            purge_bars=training_section_dict.get("purge_bars", 60),
            embargo_bars=training_section_dict.get("embargo_bars", 1440),
            optuna=OptunaConfig(**training_section_dict.get("optuna", {})),
            calibration=CalibrationConfig(**training_section_dict.get("calibration", {})),
            checkpoint=CheckpointConfig(**training_section_dict.get("checkpoint", {})),
            device=training_section_dict.get("device", "auto"),
            batch_size=training_section_dict.get("batch_size", 256),
            max_epochs=training_section_dict.get("max_epochs", 100),
            early_stopping_patience=training_section_dict.get("early_stopping_patience", 15),
            build_ensemble=training_section_dict.get("build_ensemble", True),
            meta_learner=training_section_dict.get("meta_learner", "ridge_meta"),
        )

        # Parse evaluation section
        eval_section_dict = data.get("evaluation", {})
        config_dict["evaluation"] = EvaluationSection(
            run_backtest=eval_section_dict.get("run_backtest", False),
            compute_shap=eval_section_dict.get("compute_shap", False),
            generate_report=eval_section_dict.get("generate_report", True),
            position_sizing=eval_section_dict.get("position_sizing", "fixed"),
        )

        # Parse bundling section
        bundle_section_dict = data.get("bundling", {})
        config_dict["bundling"] = BundlingSection(
            create_bundle=bundle_section_dict.get("create_bundle", True),
            bundle_format=bundle_section_dict.get("bundle_format", "directory"),
            include_oof=bundle_section_dict.get("include_oof", True),
            include_feature_importance=bundle_section_dict.get("include_feature_importance", True),
        )

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """
        Load ExperimentConfig from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            ExperimentConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "data": {
                "symbol": self.data.symbol,
                "data_path": str(self.data.data_path) if self.data.data_path else None,
                "start_date": self.data.start_date,
                "end_date": self.data.end_date,
            },
            "training": {
                "models": self.training.models,
                "horizons": self.training.horizons,
                "training_mode": self.training.training_mode,
                "cv_method": self.training.cv_method,
                "n_splits": self.training.n_splits,
                "purge_bars": self.training.purge_bars,
                "embargo_bars": self.training.embargo_bars,
                "device": self.training.device,
                "batch_size": self.training.batch_size,
                "max_epochs": self.training.max_epochs,
                "early_stopping_patience": self.training.early_stopping_patience,
                "build_ensemble": self.training.build_ensemble,
                "meta_learner": self.training.meta_learner,
            },
            "evaluation": {
                "run_backtest": self.evaluation.run_backtest,
                "compute_shap": self.evaluation.compute_shap,
                "generate_report": self.evaluation.generate_report,
                "position_sizing": self.evaluation.position_sizing,
            },
            "bundling": {
                "create_bundle": self.bundling.create_bundle,
                "bundle_format": self.bundling.bundle_format,
                "include_oof": self.bundling.include_oof,
                "include_feature_importance": self.bundling.include_feature_importance,
            },
        }

    def save_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    # =========================================================================
    # CONVERSION TO LEGACY CONFIGS (BACKWARD COMPATIBILITY)
    # =========================================================================

    def to_pipeline_config(self) -> Any:
        """
        Convert to PipelineConfig for backward compatibility.

        This is used by MLFactory to interface with existing components
        that expect the legacy PipelineConfig format.

        Returns:
            PipelineConfig instance
        """
        from src.core import PipelineConfig

        return PipelineConfig(
            symbol=self.data.symbol,
            data_path=str(self.data.data_path) if self.data.data_path else "",
            output_dir=self.output_dir,
            models=self.training.models,
            horizons=self.training.horizons,
            build_ensemble=self.training.build_ensemble,
            meta_learner=self.training.meta_learner,
            training_mode=self.training.training_mode,
            cv_method=self.training.cv_method,
            n_splits=self.training.n_splits,
            purge_bars=self.training.purge_bars,
            embargo_bars=self.training.embargo_bars,
            random_state=self.random_seed,
            hyperparam_trials=self.training.optuna.n_trials,
            label_optimization_trials=self.training.optuna.n_trials,
            feature_selection_trials=self.training.optuna.n_trials,
            optuna_random_state=self.training.optuna.random_state,
        )

    def to_trainer_config(self, model_name: str, horizon: int | None = None) -> Any:
        """
        Convert to TrainerConfig for a specific model.

        Args:
            model_name: Name of the model
            horizon: Training horizon (defaults to first horizon)

        Returns:
            TrainerConfig instance
        """
        from src.models.config.trainer_config import TrainerConfig

        horizon = horizon or self.training.horizons[0] if self.training.horizons else 20

        return TrainerConfig(
            model_name=model_name,
            horizon=horizon,
            sequence_length=self.data.sequence.seq_len,
            batch_size=self.training.batch_size,
            max_epochs=self.training.max_epochs,
            early_stopping_patience=self.training.early_stopping_patience,
            random_seed=self.random_seed,
            output_dir=self.output_dir,
            device=self.training.device,
        )

    def to_backtest_config(self) -> BacktestConfig:
        """
        Convert to BacktestConfig for backtesting.

        Returns:
            BacktestConfig instance
        """
        return BacktestConfig(
            start_date=self.data.start_date,
            end_date=self.data.end_date,
            position_sizing=self.evaluation.position_sizing,
        )

    def to_bundle_config(self, model_name: str, horizon: int) -> BundleConfig:
        """
        Convert to BundleConfig for a specific model.

        Args:
            model_name: Name of the model
            horizon: Prediction horizon

        Returns:
            BundleConfig instance
        """
        bundle_path = self.output_dir / "bundles" / f"{model_name}_h{horizon}"
        return BundleConfig(
            bundle_path=bundle_path,
            model_name=model_name,
            horizon=horizon,
            symbol=self.data.symbol,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ExperimentConfig",
    "DataSection",
    "TrainingSection",
    "EvaluationSection",
    "BundlingSection",
]
