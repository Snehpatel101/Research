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
import secrets
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

# Top-level fields that name, place or report a run without changing its
# results; excluded from ExperimentConfig.config_hash
RESULT_NEUTRAL_FIELDS = frozenset(
    {"name", "description", "run_id", "output_dir", "verbose", "tracking"}
)

# Derived embargo: one trading day. CME equity/metal futures trade ~23h a day,
# so 1440 minutes of bars spans one session at any bar timeframe.
EMBARGO_SPAN_MINUTES = 1440
# A derived embargo never takes more than this share of a CV fold.
MAX_EMBARGO_FOLD_FRACTION = 0.25


def _generate_run_id() -> str:
    """Unique run ID: timestamp to the microsecond plus a short random suffix.

    Two runs started in the same second (parallel CLI invocations, scripted loops)
    must not share an output directory.
    """
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{secrets.token_hex(2)}"


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
    # CV / split gaps in bars. None = derived by ExperimentConfig.resolve_cv_gaps:
    # purge = longest triple-barrier label span (max_bars over the horizons),
    # embargo = one trading day of bars at the bar timeframe, capped at 25% of a
    # CV fold. Explicit values are kept (a purge shorter than the label span is
    # raised to it). CV additionally purges on every label's actual end bar.
    purge_bars: int | None = None
    embargo_bars: int | None = None
    # Default training sample weights: "uniqueness" (AFML average uniqueness of
    # overlapping labels, from the training split) or "none" (all 1.0)
    sample_weighting: str = "uniqueness"

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


@dataclass
class TrackingSection:
    """
    Experiment tracking: one parent run per ``MLFactory.run`` plus one child
    run per trained model.

    - ``backend``: "none" (default), "local" or "mlflow"
    - ``tracking_uri``: local = directory the runs are written to (default
      ``<output root>/tracking``, shared by every run under that root);
      mlflow = tracking server URI or store (default: MLflow's own default,
      ``MLFLOW_TRACKING_URI`` or ``./mlruns``)
    - ``experiment_name``: tracker experiment (default: ``ExperimentConfig.name``)
    """

    backend: str = "none"
    tracking_uri: str | None = None
    experiment_name: str | None = None


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
        random_seed: Seed for every random number generator of the run (Python,
            NumPy, torch, Optuna samplers, model ``random_state``)
        deterministic: Force deterministic torch kernels (slower on GPU; CPU
            runs of the same config are bit-identical without it)
        verbose: Logging verbosity (0=silent, 1=info, 2=debug); MLFactory's
            default when its own ``verbose`` argument is omitted

        data: Data configuration section
        training: Training configuration section
        evaluation: Evaluation configuration section
        bundling: Bundling configuration section
        tracking: Experiment-tracking configuration section

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
    deterministic: bool = False
    verbose: int = 1

    # Configuration sections
    data: DataSection = field(default_factory=DataSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    bundling: BundlingSection = field(default_factory=BundlingSection)
    tracking: TrackingSection = field(default_factory=TrackingSection)

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Ensure paths are Path objects
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Create full output path with run_id (idempotent — skip if already appended)
        if self.output_dir.name != self.run_id:
            self.output_dir = self.output_dir / self.run_id

        from src.core.config import SAMPLE_WEIGHTING_MODES, TRACKING_BACKENDS

        if self.training.sample_weighting not in SAMPLE_WEIGHTING_MODES:
            raise ValueError(
                f"training.sample_weighting must be one of {SAMPLE_WEIGHTING_MODES}, "
                f"got {self.training.sample_weighting!r}"
            )
        for name in ("purge_bars", "embargo_bars"):
            value = getattr(self.training, name)
            if value is not None and value < 0:
                raise ValueError(f"training.{name} must be >= 0 or None, got {value}")
        from src.core.reproducibility import MAX_RANDOM_SEED

        if not 0 <= self.random_seed <= MAX_RANDOM_SEED:
            raise ValueError(
                f"random_seed must be in [0, {MAX_RANDOM_SEED}] (derived per-fold and "
                f"per-feature seeds must stay below 2**32), got {self.random_seed}"
            )
        if self.tracking.backend not in TRACKING_BACKENDS:
            raise ValueError(
                f"tracking.backend must be one of {TRACKING_BACKENDS}, "
                f"got {self.tracking.backend!r}"
            )

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
    # LABEL SPAN AND CV GAPS
    # =========================================================================

    def resolve_barrier_params(self, horizon: int) -> tuple[float, float, int, str]:
        """
        Resolve triple-barrier parameters for one horizon.

        Single source of truth shared by labeling, the backtester and the CV
        purge derivation, so all three always play the same game.

        Priority per field:
          1. Explicit LabelingConfig override (upper_mult / lower_mult /
             max_holding_bars set to a non-None value)
          2. Per-symbol, per-horizon BARRIER_PARAMS table

        Returns:
            (k_up, k_down, max_bars, source) where source describes which
            fields came from config overrides vs the barriers table.
        """
        from src.data.pipeline.config.barriers_config import get_barrier_params

        labeling = self.data.labeling
        table = get_barrier_params(self.data.symbol.upper(), horizon)

        k_up = labeling.upper_mult if labeling.upper_mult is not None else float(table["k_up"])
        k_down = labeling.lower_mult if labeling.lower_mult is not None else float(table["k_down"])
        max_bars = (
            labeling.max_holding_bars
            if labeling.max_holding_bars is not None
            else int(table["max_bars"])
        )

        overridden = any(
            v is not None
            for v in (labeling.upper_mult, labeling.lower_mult, labeling.max_holding_bars)
        )
        source = "labeling-config override" if overridden else "barriers table"
        return k_up, k_down, max_bars, source

    def label_span_bars(self) -> int:
        """Longest label span in bars: a triple-barrier label resolves within max_bars."""
        return max(self.resolve_barrier_params(h)[2] for h in self.training.horizons)

    def resolve_cv_gaps(
        self,
        bar_timeframe: str | None = None,
        n_rows: int | None = None,
    ) -> tuple[int, int]:
        """
        Purge and embargo bars for CV and the train/val/test split gaps.

        - purge: ``training.purge_bars`` if set, else the longest label span
          (``label_span_bars``). An explicit purge shorter than the span is
          raised to it with a warning — a shorter purge leaks labels that
          resolve inside the next block.
        - embargo: ``training.embargo_bars`` if set, else one trading day
          (``EMBARGO_SPAN_MINUTES``) of bars at the bar timeframe, capped at
          ``MAX_EMBARGO_FOLD_FRACTION`` of a CV fold when ``n_rows`` is known.

        Args:
            bar_timeframe: Training bar timeframe (detected or declared). Falls
                back to ``data.bar_timeframe``.
            n_rows: Rows of the labeled training frame (for the fold cap).

        Returns:
            (purge_bars, embargo_bars)
        """
        span = self.label_span_bars()
        purge = self.training.purge_bars
        if purge is None:
            purge = span
        elif purge < span:
            logger.warning(
                f"training.purge_bars={purge} is shorter than the longest label span "
                f"({span} bars = max_bars over horizons {self.training.horizons}); "
                f"raising it to {span} to prevent label leakage"
            )
            purge = span

        embargo = self.training.embargo_bars
        if embargo is None:
            embargo = self._derive_embargo_bars(bar_timeframe or self.data.bar_timeframe, n_rows)
        return int(purge), int(embargo)

    def _derive_embargo_bars(self, bar_timeframe: str | None, n_rows: int | None) -> int:
        """One trading day of bars, capped at a fraction of a CV fold."""
        from src.core.common.horizon_config import (
            DEFAULT_TIMEFRAME_MINUTES,
            compute_embargo_bars,
        )

        if bar_timeframe is None:
            bar_timeframe = f"{DEFAULT_TIMEFRAME_MINUTES}min"
            logger.warning(
                f"Bar timeframe unknown; deriving the embargo assuming {bar_timeframe} bars"
            )
        embargo = compute_embargo_bars(bar_timeframe, embargo_time_minutes=EMBARGO_SPAN_MINUTES)
        derivation = f"{EMBARGO_SPAN_MINUTES} min of {bar_timeframe} bars"
        if n_rows is not None:
            fold = int(n_rows * self.data.splits.train_ratio) // self.training.n_splits
            cap = int(fold * MAX_EMBARGO_FOLD_FRACTION)
            if embargo > cap:
                derivation += f", capped at {MAX_EMBARGO_FOLD_FRACTION:.0%} of a {fold}-bar fold"
                embargo = cap
        logger.info(f"Derived embargo_bars={embargo} ({derivation})")
        return embargo

    # =========================================================================
    # IDENTITY AND TRACKING
    # =========================================================================

    def config_hash(self) -> str:
        """
        SHA-256 of every setting that can change a run's results.

        Two runs with the same hash, input data and code produce the same
        output. Fields that only name, place or report the run
        (``RESULT_NEUTRAL_FIELDS``: name, description, run_id, output_dir,
        verbose, tracking) are excluded, so re-running an experiment under a
        new run ID keeps its hash.
        """
        from src.core.run_manifest import canonical_json_sha256

        definition = {k: v for k, v in self.to_dict().items() if k not in RESULT_NEUTRAL_FIELDS}
        return canonical_json_sha256(definition)

    def tracking_experiment_name(self) -> str:
        """Tracker experiment the run is logged to (``tracking.experiment_name`` or ``name``)."""
        return self.tracking.experiment_name or self.name

    def tracking_location(self) -> str | None:
        """
        Where the tracker writes: ``tracking.tracking_uri`` when set; for the
        local backend ``<output root>/tracking`` (next to the run directories);
        None for mlflow (MLflow's default) and "none".
        """
        if self.tracking.tracking_uri:
            return self.tracking.tracking_uri
        if self.tracking.backend == "local":
            return str(self.output_dir.parent / "tracking")
        return None

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

    def to_pipeline_config(
        self,
        cv_gaps: tuple[int, int] | None = None,
        bar_timeframe: str | None = None,
        n_rows: int | None = None,
        tracking_parent_run_id: str | None = None,
    ) -> Any:
        """
        Convert to the PipelineConfig consumed by the training orchestrator
        and BundleBuilder.

        Args:
            cv_gaps: Already-resolved (purge_bars, embargo_bars). When None
                they are resolved here via ``resolve_cv_gaps``.
            bar_timeframe: Training bar timeframe, for the derived embargo.
            n_rows: Rows of the labeled training frame, for the embargo cap.
            tracking_parent_run_id: Tracker run of the factory; every trained
                model logs a child run under it.

        Returns:
            PipelineConfig instance
        """
        from src.core import PipelineConfig

        purge_bars, embargo_bars = cv_gaps or self.resolve_cv_gaps(bar_timeframe, n_rows)

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
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            sample_weighting=self.training.sample_weighting,
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
            deterministic=self.deterministic,
            # Experiment tracking: trainers log one child run per model under
            # the factory's parent run
            tracking_backend=self.tracking.backend,
            tracking_uri=self.tracking_location(),
            tracking_experiment=self.tracking_experiment_name(),
            tracking_parent_run_id=tracking_parent_run_id,
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
    "TrackingSection",
]
