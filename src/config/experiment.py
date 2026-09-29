"""
ExperimentConfig - Single Source of Truth for ML Factory Experiments.

ExperimentConfig is the top-level config used by MLFactory. Every field is
either read by MLFactory directly or reaches the training stack through
``to_pipeline_config()`` — there are no settable-but-ignored knobs. Loading a
dict/YAML that carries keys from older versions logs a warning per unknown key
and ignores it instead of failing.

Example:
    from src.config.experiment import ExperimentConfig

    config = ExperimentConfig(name="mes_xgboost_experiment")
    config.data.symbol = "MES"
    config.data.data_path = "data/mes_1min.parquet"
    config.training.models = ["xgboost", "lightgbm", "lstm"]

    from src.factory import MLFactory
    result = MLFactory(config).run()
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml

from src.config.cv import WalkForwardConfig
from src.config.data import (
    FeatureConfig,
    LabelingConfig,
    MTFConfig,
    SequenceConfig,
    SplitConfig,
)
from src.config.training import CalibrationConfig, OptunaConfig

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _generate_run_id() -> str:
    """Generate unique run ID with timestamp."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _dataclass_from_dict(cls: type[_T], raw: dict[str, Any], where: str) -> _T:  # noqa: UP047
    """Build dataclass ``cls`` from a nested dict, warning on (and dropping) unknown keys.

    Nested config sections are recognised by their ``default_factory`` being a
    dataclass type, and are built recursively. A section given as ``None``
    (an empty YAML mapping) falls back to its defaults.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"{where} must be a mapping, got {type(raw).__name__}")

    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - set(known))
    if unknown:
        logger.warning(
            f"Ignoring unknown config key(s) in {where}: {unknown} "
            "(misspelled, or removed because they never reached the pipeline)"
        )

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        if name not in known:
            continue
        factory = known[name].default_factory
        if isinstance(factory, type) and is_dataclass(factory):
            if value is None:
                continue
            value = _dataclass_from_dict(factory, value, f"{where}.{name}")
        kwargs[name] = value
    return cls(**kwargs)


def _plain(value: Any) -> Any:
    """Recursively convert Paths/tuples so the dict is safe_dump/JSON friendly."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


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
    # Training bar timeframe. None = use the input bars as-is (auto-detected);
    # e.g. "5min" resamples 1-minute input to 5-minute bars before features.
    bar_timeframe: str | None = None

    # Sub-configs
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    mtf: MTFConfig = field(default_factory=MTFConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)


@dataclass
class RegimeSettings:
    """Regime-aware mode (training_mode="regime_aware"): one model per market regime."""

    detection_method: str = "volatility_percentile"  # volatility_percentile, trend_adx, combined
    n_regimes: int = 3  # 2 = low/high, 3 = low/medium/high
    lookback: int = 60  # bars of history for the rolling regime statistic


@dataclass
class MetaLabelingSettings:
    """Meta-labeling mode (training_mode="meta_labeling").

    The primary (direction) model is ``training.models[0]``; the meta-model
    learns P(primary correct) and trades are taken only above ``threshold``.
    """

    meta_model: str = "logistic"  # logistic, random_forest, xgboost, lightgbm, catboost
    threshold: float = 0.5


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
    training_mode: str = "standard"  # standard, walk_forward, regime_aware, meta_labeling
    cv_method: str = "purged_kfold"
    n_splits: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440

    # Walk-forward validation settings (used when training_mode="walk_forward")
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    regime: RegimeSettings = field(default_factory=RegimeSettings)
    meta_labeling: MetaLabelingSettings = field(default_factory=MetaLabelingSettings)

    # Sub-configs
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Neural network settings (device is auto-detected per model)
    batch_size: int = 512
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
    position_sizing: str = "fixed"

    # Transaction cost overrides (passed to BacktestConfig)
    commission_per_contract: float | None = None
    slippage_ticks: float | None = None
    initial_equity: float = 100000.0


@dataclass
class BundlingSection:
    """
    Bundling-related configuration section.
    """

    create_bundle: bool = True
    deploy_artifact: bool = True


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
        description: Free-text description (metadata, saved with the run)
        run_id: Unique run identifier (auto-generated)
        output_dir: Output directory for artifacts
        random_seed: Random seed for reproducibility
        verbose: Logging verbosity (0=silent, 1=info, 2=debug); MLFactory's
            default when its own ``verbose`` argument is omitted

        data: Data configuration section
        training: Training configuration section
        evaluation: Evaluation configuration section
        bundling: Bundling configuration section

    Example:
        config = ExperimentConfig(
            name="mes_xgboost_experiment",
            data=DataSection(symbol="MES", data_path="data/mes_1min.parquet"),
            training=TrainingSection(models=["xgboost", "lightgbm"], horizons=[5, 20]),
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

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Ensure paths are Path objects
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Create full output path with run_id (idempotent — skip if already appended)
        if self.output_dir.name != self.run_id:
            self.output_dir = self.output_dir / self.run_id

        # Validate purge_bars >= max(horizons) to prevent label leakage
        if self.training.horizons:
            max_h = max(self.training.horizons)
            if self.training.purge_bars < max_h:
                logger.warning(
                    f"purge_bars ({self.training.purge_bars}) < max horizon ({max_h}). "
                    f"This may cause label leakage. Auto-correcting to {max_h}."
                )
                self.training.purge_bars = max_h

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
        Create ExperimentConfig from a (possibly nested) dictionary.

        Missing keys take their defaults. Unknown keys — typos, or fields
        removed in later versions (e.g. Phase 116 pruned every setting that
        never reached the pipeline) — are logged with a warning and ignored,
        so YAML written by older versions still loads.

        Args:
            data: Configuration dictionary

        Returns:
            ExperimentConfig instance
        """
        return _dataclass_from_dict(cls, data or {}, "ExperimentConfig")

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
        """Convert to a plain (YAML/JSON-safe) nested dictionary."""
        return _plain(asdict(self))

    def save_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # safe_dump pairs with the safe_load in from_yaml: the full Dumper
            # emits python-specific tags (e.g. !!python/tuple) that safe_load
            # then refuses to parse, breaking every YAML round-trip.
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    # =========================================================================
    # CONVERSION TO THE TRAINING STACK'S CONFIG
    # =========================================================================

    def to_pipeline_config(self) -> Any:
        """
        Convert to the PipelineConfig consumed by the training orchestrator
        and BundleBuilder.

        Returns:
            PipelineConfig instance
        """
        from src.core import PipelineConfig

        # Derive optimization flags from trial count
        _do_optimize = self.training.optuna.n_trials > 0

        # Binary mode: 2 classes instead of 3
        _n_classes = 2 if self.data.labeling.binary_mode else 3

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
            # Chronological split ratios (purge/embargo gaps sit between them)
            train_ratio=self.data.splits.train_ratio,
            val_ratio=self.data.splits.val_ratio,
            test_ratio=self.data.splits.test_ratio,
            # Walk-forward validation settings
            wf_n_windows=self.training.walk_forward.n_windows,
            wf_window_type=self.training.walk_forward.window_type,
            wf_min_train_pct=self.training.walk_forward.min_train_pct,
            wf_test_pct=self.training.walk_forward.test_pct,
            # Regime-aware settings
            regime_detection_method=self.training.regime.detection_method,
            n_regimes=self.training.regime.n_regimes,
            regime_lookback=self.training.regime.lookback,
            # Meta-labeling: the first listed model is the primary
            meta_labeling_primary_model=self.training.models[0],
            meta_labeling_meta_model=self.training.meta_labeling.meta_model,
            meta_labeling_threshold=self.training.meta_labeling.threshold,
            random_state=self.random_seed,
            # Optimization flags — Optuna-based ones disabled when n_trials=0.
            # Feature selection is NOT Optuna-based (MDA ranking) and follows
            # features.selection_enabled independently of the trial count.
            optimize_hyperparams=_do_optimize,
            optimize_labels=_do_optimize,
            optimize_features=self.data.features.selection_enabled,
            # Optuna trial counts - all driven by OptunaConfig.n_trials
            hyperparam_trials=self.training.optuna.n_trials,
            label_optimization_trials=self.training.optuna.n_trials,
            feature_selection_trials=self.training.optuna.n_trials,
            feature_pruning_trials=self.training.optuna.n_trials,
            optuna_metric=self.training.optuna.metric,
            optuna_timeout=self.training.optuna.timeout,
            # MTF configuration (empty list disables MTF features)
            mtf_timeframes=self.data.mtf.timeframes if self.data.mtf.enabled else [],
            # Sequence configuration
            sequence_length=self.data.sequence.seq_len,
            # Neural network configuration
            batch_size=self.training.batch_size,
            max_epochs=self.training.max_epochs,
            early_stopping_patience=self.training.early_stopping_patience,
            # Probability calibration (fitted on each model's validation split)
            auto_calibrate=self.training.calibration.enabled,
            calibration_method=self.training.calibration.method,
            # Classification mode
            n_classes=_n_classes,
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
