"""
Data-section configuration classes for ExperimentConfig.

Every field of these classes reaches the pipeline (see
``ExperimentConfig.to_pipeline_config()`` and ``MLFactory.prepare_data``):

- FeatureConfig: feature-selection switch + governance diagnostics + fractional differentiation
- LabelingConfig: triple-barrier overrides + binary mode + CUSUM event sampling
- SequenceConfig: window length for sequence models
- MTFConfig: multi-timeframe feature switch + timeframes
- SplitConfig: chronological train/val/test ratios

The enums below are shared with the data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from src.config.base import BaseConfig
from src.core.constants import FRAC_DIFF_PRICE_COLUMNS

# =============================================================================
# ENUMS
# =============================================================================


class ScalerType(StrEnum):
    """Supported scaler types for data normalization."""

    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    QUANTILE = "quantile"


class FeatureCategory(StrEnum):
    """Feature categories for scaling strategy selection."""

    RETURNS = "returns"
    OSCILLATOR = "oscillator"
    PRICE_LEVEL = "price_level"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TEMPORAL = "temporal"
    BINARY = "binary"
    UNKNOWN = "unknown"


class MTFMode(StrEnum):
    """Multi-timeframe aggregation modes."""

    NONE = "none"
    BARS = "bars"
    INDICATORS = "indicators"
    BOTH = "both"
    MULTI_STREAM = "multi_stream"


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================


@dataclass
class FeatureGovernanceConfig(BaseConfig):
    """
    Opt-in feature-governance diagnostics run after feature selection.

    Everything here is READ-ONLY with respect to the selection: the selected
    features are identical with the report on or off. All diagnostics use the
    TRAIN split rows only, and every ranking inside them is the same purged-CV
    (label-span purge + embargo) out-of-sample MDA the selection itself uses.
    Cost: about ``n_bootstrap + len(barrier_scales)`` extra MDA rankings.

    Attributes:
        report: Master switch. Writes ``<output_dir>/feature_governance/h{h}.json``
            (h = the ranking horizon) with per-feature MDA importance, selection
            stability across contiguous blocks, label-perturbation rank shifts,
            robustness score and per-model selection.
        bootstrap_stability: Include block-subsample selection frequency
            (stability selection). Only used when ``report`` is on.
        label_perturbation: Include rank shifts when triple-barrier widths change
            by ``barrier_scales``. Only used when ``report`` is on.
        registry: Persist a cross-run FeatureRegistry (lifecycle state per
            feature) updated from this run's selection and stability verdicts.
            Only used when ``report`` is on.
        registry_path: Registry JSON path. None = ``<runs dir>/feature_registry_<SYMBOL>.json``
            next to the run directories, so every run of a symbol shares it.
        n_bootstrap: Number of contiguous blocks for the stability estimate.
        stability_threshold: Minimum share of blocks a feature must rank in the
            top-K (K = the largest per-model feature budget) to count as stable.
        window_fraction: Length of each block as a share of the train rows.
        barrier_scales: Multipliers applied to the label's k_up and k_down for
            the perturbed label variants (e.g. 0.75 = tighter, 1.25 = wider).
        max_degraded_runs: Consecutive failing runs in DEGRADED before the
            registry retires a feature (retirement is recorded, never applied).
    """

    report: bool = False
    bootstrap_stability: bool = True
    label_perturbation: bool = True
    registry: bool = True
    registry_path: str | None = None
    n_bootstrap: int = 8
    stability_threshold: float = 0.6
    window_fraction: float = 0.5
    barrier_scales: list[float] = field(default_factory=lambda: [0.75, 1.25])
    max_degraded_runs: int = 3

    def validate(self) -> list[str]:
        """Validate governance configuration."""
        issues = super().validate()
        if self.n_bootstrap < 1:
            issues.append(f"n_bootstrap must be >= 1, got {self.n_bootstrap}")
        if not 0.0 < self.stability_threshold <= 1.0:
            issues.append(f"stability_threshold must be in (0, 1], got {self.stability_threshold}")
        if not 0.0 < self.window_fraction <= 1.0:
            issues.append(f"window_fraction must be in (0, 1], got {self.window_fraction}")
        if any(scale <= 0 for scale in self.barrier_scales):
            issues.append(f"barrier_scales must all be positive, got {self.barrier_scales}")
        if self.max_degraded_runs < 1:
            issues.append(f"max_degraded_runs must be >= 1, got {self.max_degraded_runs}")
        return issues

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ValueError("Invalid features.governance config: " + "; ".join(issues))


@dataclass
class FracDiffConfig(BaseConfig):
    """
    Fractionally differentiated log-price features (AFML ch. 5), opt-in.

    Adds ``ffd_log_<column>`` features: a fixed-width-window fractional
    difference of the log price. The window does not depend on the data
    length, and the value at a bar uses only earlier bars (lagged one bar like
    every other feature), so training and inference agree exactly.

    Attributes:
        enabled: Add the FFD features (default False = features unchanged)
        d: Differentiation order in (0, 1], or ``"auto"``: the smallest d whose
            FFD log close passes the ADF stationarity test (at least 0.05, even
            for a series that is already stationary), fitted on the leading
            training bars only (same prefix as the CUSUM threshold) and then
            frozen into the feature spec (inference replays the same d).
            ``"auto"`` needs the ``stats`` extra (statsmodels).
        columns: Price columns (of open/high/low/close) to differentiate
        window: Fixed FFD window cap in bars (the first ``window`` bars of a
            series are warmup NaN)
        threshold: FFD weight truncation threshold
    """

    enabled: bool = False
    d: float | str = "auto"
    columns: list[str] = field(default_factory=lambda: ["close", "open", "high", "low"])
    window: int = 100
    threshold: float = 1e-5

    def validate(self) -> list[str]:
        """Validate fractional-differentiation configuration."""
        issues = super().validate()

        if isinstance(self.d, str):
            if self.d != "auto":
                issues.append(f"frac_diff.d must be a number in (0, 1] or 'auto', got {self.d!r}")
        elif isinstance(self.d, bool) or not 0.0 < float(self.d) <= 1.0:
            issues.append(f"frac_diff.d must be in (0, 1] or 'auto', got {self.d}")

        if self.enabled and not self.columns:
            issues.append("frac_diff.columns must not be empty when frac_diff is enabled")
        unknown = [c for c in self.columns if c not in FRAC_DIFF_PRICE_COLUMNS]
        if unknown:
            issues.append(
                f"frac_diff.columns must be among {list(FRAC_DIFF_PRICE_COLUMNS)}, got {unknown}"
            )
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 2:
            issues.append(f"frac_diff.window must be an integer >= 2, got {self.window!r}")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not 0.0 < self.threshold < 1.0
        ):
            issues.append(f"frac_diff.threshold must be a number in (0, 1), got {self.threshold!r}")

        return issues


@dataclass
class FeatureConfig(BaseConfig):
    """
    Feature configuration.

    The feature set itself is fixed (``FeatureEngineer`` computes every
    family; MTF is controlled by ``MTFConfig``). What is configurable is
    whether the MDA feature-selection pipeline prunes it per model (feature
    counts come from each model's contract), whether governance diagnostics
    are written alongside it, and whether fractionally differentiated price
    features are added.

    Attributes:
        selection_enabled: Run train-only MDA feature selection per model
        governance: Opt-in stability / label-perturbation / registry diagnostics
        frac_diff: Fractionally differentiated log-price features (opt-in)
    """

    selection_enabled: bool = True
    governance: FeatureGovernanceConfig = field(default_factory=FeatureGovernanceConfig)
    frac_diff: FracDiffConfig = field(default_factory=FracDiffConfig)

    def validate(self) -> list[str]:
        """Validate feature configuration."""
        return super().validate() + self.frac_diff.validate()


# =============================================================================
# LABELING CONFIGURATION
# =============================================================================


@dataclass
class LabelingConfig(BaseConfig):
    """
    Triple-barrier labeling configuration.

    Attributes:
        upper_mult: Upper barrier multiplier in ATR units. None (default) means
            "auto": use the per-symbol/per-horizon BARRIER_PARAMS table, which is
            the same source the backtester uses — keeping labels and backtest
            playing the same game. Set explicitly to override both.
        lower_mult: Lower barrier multiplier (ATR units). None = auto, see upper_mult.
        atr_period: ATR calculation period
        max_holding_bars: Maximum holding period (time barrier, bars).
            None = auto from BARRIER_PARAMS, see upper_mult.
        binary_mode: Remap labels to {0: time-out, 1: barrier hit}
        event_sampling: ``"none"`` (default: every bar is labeled) or
            ``"cusum"`` (AFML ch. 2): only bars where a symmetric CUSUM filter
            on the log returns fires carry a label; all other bars are marked
            invalid (-99) and dropped from training, CV and the ensemble.
            Features are still computed on every bar. Label spans stay in bar
            coordinates, so purging and uniqueness weights follow the events.
            The backtest acts on event bars only; ``predict_from_raw`` still
            predicts every bar and flags the event bars in
            ``metadata["is_event"]``.
        cusum_threshold: CUSUM threshold in log-return units, or ``"auto"``:
            ``cusum_vol_multiple`` x the per-bar return volatility of the
            leading bars only — the training split (walk-forward: the bars
            before the first test window) — frozen into the deployment bundle.
            The validation/test holdout never influences it; purged-CV folds
            inside the training split see a value fitted on all of it (the
            same convention as the labeler's cost calibration).
        cusum_vol_multiple: Multiple for the ``"auto"`` threshold. For i.i.d.
            returns an event fires about every ``multiple^2`` bars.

    Example:
        config = LabelingConfig(
            upper_mult=2.0,
            lower_mult=2.0,
            max_holding_bars=20,
        )
    """

    upper_mult: float | None = None
    lower_mult: float | None = None
    atr_period: int = 14
    max_holding_bars: int | None = None

    # Binary classification mode
    binary_mode: bool = False  # If True, remap labels to binary: 0=neutral, 1=significant_move

    # Event sampling (AFML ch. 2)
    event_sampling: str = "none"
    cusum_threshold: float | str = "auto"
    cusum_vol_multiple: float = 3.0

    def validate(self) -> list[str]:
        """Validate labeling configuration."""
        issues = super().validate()

        if self.event_sampling not in ("none", "cusum"):
            issues.append(f"event_sampling must be 'none' or 'cusum', got {self.event_sampling!r}")
        if isinstance(self.cusum_threshold, str):
            if self.cusum_threshold != "auto":
                issues.append(
                    f"cusum_threshold must be a positive number or 'auto', "
                    f"got {self.cusum_threshold!r}"
                )
        elif isinstance(self.cusum_threshold, bool) or self.cusum_threshold <= 0:
            issues.append(f"cusum_threshold must be positive or 'auto', got {self.cusum_threshold}")
        if (
            isinstance(self.cusum_vol_multiple, bool)
            or not isinstance(self.cusum_vol_multiple, (int, float))
            or self.cusum_vol_multiple <= 0
        ):
            issues.append(
                f"cusum_vol_multiple must be a positive number, got {self.cusum_vol_multiple!r}"
            )

        if self.upper_mult is not None and self.upper_mult <= 0:
            issues.append(f"upper_mult must be positive, got {self.upper_mult}")

        if self.lower_mult is not None and self.lower_mult <= 0:
            issues.append(f"lower_mult must be positive, got {self.lower_mult}")

        if self.atr_period <= 0:
            issues.append(f"atr_period must be positive, got {self.atr_period}")

        if self.max_holding_bars is not None and self.max_holding_bars <= 0:
            issues.append(f"max_holding_bars must be positive, got {self.max_holding_bars}")

        return issues


# =============================================================================
# SEQUENCE CONFIGURATION
# =============================================================================


@dataclass
class SequenceConfig(BaseConfig):
    """
    Sequence-window configuration for 3D/4D models.

    Attributes:
        seq_len: Window length (time steps per sample) for EVERY sequence model.
            None (default) gives each model its contract length (e.g. TCN 64,
            Transformer 128, most others 60).
    """

    seq_len: int | None = None

    def validate(self) -> list[str]:
        """Validate sequence configuration."""
        issues = super().validate()

        if self.seq_len is not None and self.seq_len <= 0:
            issues.append(f"seq_len must be positive, got {self.seq_len}")

        return issues


# =============================================================================
# MTF CONFIGURATION
# =============================================================================


@dataclass
class MTFConfig(BaseConfig):
    """
    Multi-timeframe feature configuration.

    Attributes:
        enabled: Whether MTF features are computed
        timeframes: Higher timeframes to aggregate (also the streams fed to
            multi-stream 4D models)

    Example:
        config = MTFConfig(enabled=True, timeframes=["15min", "60min"])
    """

    enabled: bool = True
    timeframes: list[str] = field(default_factory=lambda: ["5min", "15min", "60min"])


# =============================================================================
# SPLITS CONFIGURATION
# =============================================================================


@dataclass
class SplitConfig(BaseConfig):
    """
    Chronological train/val/test split ratios.

    The purge/embargo gaps between splits come from
    ``training.purge_bars`` / ``training.embargo_bars``.

    Attributes:
        train_ratio: Proportion for training
        val_ratio: Proportion for validation (calibration, early stopping)
        test_ratio: Proportion held out for testing
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def validate(self) -> list[str]:
        """Validate split configuration."""
        issues = super().validate()

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            issues.append(f"Split ratios must sum to 1.0, got {total}")

        if self.train_ratio <= 0:
            issues.append(f"train_ratio must be positive, got {self.train_ratio}")

        if self.val_ratio < 0:
            issues.append(f"val_ratio must be non-negative, got {self.val_ratio}")

        if self.test_ratio < 0:
            issues.append(f"test_ratio must be non-negative, got {self.test_ratio}")

        return issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ScalerType",
    "FeatureCategory",
    "MTFMode",
    # Configs
    "FeatureConfig",
    "FeatureGovernanceConfig",
    "FracDiffConfig",
    "LabelingConfig",
    "SequenceConfig",
    "MTFConfig",
    "SplitConfig",
]
