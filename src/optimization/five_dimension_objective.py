"""
5-Dimension Optuna Objective - Unified search across all optimization dimensions.

Phase 3 of the SNwH (Unified Multi-Timeframe Model Factory) implementation.

This module extends Optuna optimization from 1 dimension (hyperparameters only)
to all 5 dimensions:
    1. Triple Barrier Parameters - profit_threshold, loss_threshold, max_holding_bars
    2. Feature Selection - which features from the base set to use
    3. Feature Parameters - tunable parameters per feature (RSI period, ATR window, etc.)
    4. Feature Timeframes - which timeframe each feature uses
    5. Model Hyperparameters - model-specific tuning parameters (existing)

The objective function samples all 5 dimensions from Optuna and builds a FeatureSpec
that captures the complete configuration. This FeatureSpec can then be:
    - Stored in the trial for later retrieval
    - Saved to disk for reproducibility
    - Used to configure training and inference

Key Design Decisions:
    - Feature selection uses categorical trials (True/False) for each base feature
    - Feature parameters scale around defaults (0.5x to 2x)
    - Timeframe assignment is categorical across available timeframes
    - Hyperparameters use the existing HYPERPARAMETER_SPACES definitions
    - Labels are generated INSIDE each trial using the trial's barrier parameters
      (this makes labels model-specific rather than pre-computed once)

Usage:
    from src.optimization.five_dimension_objective import (
        create_5d_objective,
        FiveDimensionResult,
    )

    objective = create_5d_objective(
        model_family=ModelFamily.BOOSTING,
        train_data=train_df,
        val_data=val_df,
        timeframes=["5min", "15min", "30min"],
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Get best FeatureSpec
    best_spec = FiveDimensionResult.from_trial(study.best_trial).feature_spec

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.core.constants import (
    DEFAULT_MAX_FEATURES_TO_SEARCH,
    DEFAULT_MIN_TRADE_RATE,
    DEFAULT_PREFERRED_TRADE_RATE,
)
from src.core.contracts import FeatureSpec
from src.core.types import ModelFamily
from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler
from src.optimization.base_feature_sets import (
    DEFAULT_FEATURE_PARAMS,
    get_base_features,
)
from src.optimization.hyperparameters import (
    HYPERPARAMETER_SPACES,
    suggest_hyperparameters,
)
from src.validation.deflated_sharpe import (
    compute_dsr_from_optuna_study,
    dsr_gate,
    is_sharpe_like_metric,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ANNUALIZATION HELPER
# =============================================================================


def _infer_annualization_factor(df: pd.DataFrame) -> float:
    """
    Infer the annualization factor from a DataFrame's DatetimeIndex.

    Computes bars_per_year from the median timedelta between rows.
    Falls back to 252*78 (US equity 5-min bars) if inference fails.

    Common results:
        - 5-min equity:  252 * 78  = 19,656
        - 1-min equity:  252 * 390 = 98,280
        - 1-hour crypto: 365 * 24  = 8,760
        - Daily equity:  252

    Args:
        df: DataFrame, ideally with a DatetimeIndex

    Returns:
        Annualization factor (bars per year)
    """
    _DEFAULT_FACTOR = 252 * 78  # US equity 5-min bars

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.debug(
            "DataFrame index is not DatetimeIndex, using default annualization factor "
            f"({_DEFAULT_FACTOR})"
        )
        return float(_DEFAULT_FACTOR)

    if len(df) < 3:
        logger.debug(
            f"DataFrame too short ({len(df)} rows) to infer frequency, "
            f"using default annualization factor ({_DEFAULT_FACTOR})"
        )
        return float(_DEFAULT_FACTOR)

    # Compute median timedelta (robust to gaps like weekends/holidays)
    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        return float(_DEFAULT_FACTOR)

    median_delta = deltas.median()
    total_seconds = median_delta.total_seconds()

    if total_seconds <= 0:
        logger.warning(
            f"Non-positive median timedelta ({median_delta}), "
            f"using default annualization factor ({_DEFAULT_FACTOR})"
        )
        return float(_DEFAULT_FACTOR)

    # Compute bars per year
    seconds_per_year = 365.25 * 24 * 3600  # ~31,557,600
    bars_per_year = seconds_per_year / total_seconds

    logger.debug(
        f"Inferred bar frequency: {median_delta} -> " f"annualization factor: {bars_per_year:.0f}"
    )

    return float(bars_per_year)


# =============================================================================
# CONSTANTS
# =============================================================================

# Triple barrier parameter bounds (ATR multipliers)
# These are multipliers for ATR, not percentage thresholds
# Upper barrier = entry_price + (ATR * upper_mult)
# Lower barrier = entry_price - (ATR * lower_mult)
UPPER_MULT_RANGE = (1.0, 3.0)  # ATR multiplier for profit barrier
LOWER_MULT_RANGE = (0.5, 2.0)  # ATR multiplier for loss barrier
MAX_HOLDING_BARS_RANGE = (30, 240)  # 30 bars to 240 bars

# Feature selection constraints — aligned with src.core.constants.
# MIN_FEATURES is intentionally lower than DEFAULT_MIN_FEATURES (20) because
# the 5D objective explores aggressive feature subsets during search.
MIN_FEATURES = 5
MAX_FEATURES_TO_SEARCH = DEFAULT_MAX_FEATURES_TO_SEARCH

# Default timeframes
DEFAULT_TIMEFRAMES = ["5min", "15min", "30min", "60min"]

# Label cache for trials with same barrier params (optimization)
# Phase 29: Bounded LRU cache to prevent OOM in long optimization runs
LABEL_CACHE_MAXSIZE = 128  # Maximum cached label configurations
_label_cache: OrderedDict[tuple[float, float, int, str], np.ndarray] = OrderedDict()
_label_cache_lock = threading.Lock()


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class FiveDimensionResult:
    """
    Result container for 5-dimension optimization.

    Stores the FeatureSpec and associated metrics from an Optuna trial.

    Attributes:
        feature_spec: The complete 5-dimension specification
        validation_metric: The objective metric value (e.g., Sharpe ratio)
        metric_name: Name of the validation metric
        trial_number: Optuna trial number
        optimization_time: Time taken for this trial in seconds
        additional_metrics: Optional dict of additional metrics
    """

    feature_spec: FeatureSpec
    validation_metric: float
    metric_name: str = "validation_score"
    trial_number: int = -1
    optimization_time: float = 0.0
    additional_metrics: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"FiveDimensionResult(\n"
            f"  model={self.feature_spec.model_name},\n"
            f"  n_features={self.feature_spec.n_features},\n"
            f"  {self.metric_name}={self.validation_metric:.4f},\n"
            f"  trial={self.trial_number}\n"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature_spec": self.feature_spec.to_dict(),
            "validation_metric": self.validation_metric,
            "metric_name": self.metric_name,
            "trial_number": self.trial_number,
            "optimization_time": self.optimization_time,
            "additional_metrics": self.additional_metrics,
        }

    @classmethod
    def from_trial(cls, trial: optuna.Trial) -> FiveDimensionResult:
        """
        Reconstruct FiveDimensionResult from an Optuna trial.

        Args:
            trial: Completed Optuna trial with user attributes set

        Returns:
            FiveDimensionResult instance
        """
        spec_dict = trial.user_attrs.get("feature_spec", {})
        if not spec_dict:
            raise ValueError("Trial does not contain feature_spec user attribute")

        feature_spec = FeatureSpec.from_dict(spec_dict)
        return cls(
            feature_spec=feature_spec,
            validation_metric=trial.value if trial.value is not None else 0.0,
            metric_name=trial.user_attrs.get("metric_name", "validation_score"),
            trial_number=trial.number,
            optimization_time=trial.user_attrs.get("optimization_time", 0.0),
            additional_metrics=trial.user_attrs.get("additional_metrics", {}),
        )


# =============================================================================
# HYPERPARAMETER SUGGESTION BY MODEL FAMILY
# =============================================================================


def suggest_hyperparameters_by_family(
    trial: optuna.Trial,
    model_family: ModelFamily,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Suggest hyperparameters based on model family or specific model.

    This function provides sensible defaults for each model family,
    with the option to use specific model search spaces.

    Args:
        trial: Optuna trial
        model_family: Model family (BOOSTING, NEURAL, etc.)
        model_name: Optional specific model name for precise search space

    Returns:
        Dictionary of suggested hyperparameters
    """
    # If specific model name is provided and exists in HYPERPARAMETER_SPACES, use it
    if model_name and model_name.lower() in HYPERPARAMETER_SPACES:
        return suggest_hyperparameters(trial, model_name.lower())

    # Otherwise, use family-specific defaults
    if model_family == ModelFamily.BOOSTING:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    elif model_family == ModelFamily.NEURAL:
        return {
            "hidden_size": trial.suggest_int("hidden_size", 32, 256),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        }
    elif model_family == ModelFamily.TRANSFORMER:
        return {
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256]),
            "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 6),
            "dim_feedforward": trial.suggest_int("dim_feedforward", 64, 512),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    elif model_family == ModelFamily.CLASSICAL:
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
        }
    elif model_family == ModelFamily.ENSEMBLE:
        return {
            "passthrough": trial.suggest_categorical("passthrough", [True, False]),
        }
    elif model_family == ModelFamily.META_LEARNER:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
    else:
        return {}


# =============================================================================
# 5-DIMENSION OBJECTIVE FACTORY
# =============================================================================


def generate_labels_for_trial(
    data: pd.DataFrame,
    upper_mult: float,
    lower_mult: float,
    max_holding_bars: int,
    atr_period: int = 14,
    cache_key: str = "",
    use_cache: bool = True,
) -> np.ndarray:
    """
    Generate triple barrier labels for a specific trial's barrier parameters.

    This function is called inside each Optuna trial to generate labels
    using that trial's sampled barrier parameters, making labels model-specific.

    Args:
        data: DataFrame with OHLCV data (must have 'open', 'high', 'low', 'close')
        upper_mult: ATR multiplier for upper (profit) barrier
        lower_mult: ATR multiplier for lower (stop) barrier
        max_holding_bars: Maximum bars to hold before timeout
        atr_period: Period for ATR calculation (default: 14)
        cache_key: Unique identifier for this data split (e.g., "train" or "val")
        use_cache: Whether to use label cache (default: True)

    Returns:
        np.ndarray: Array of labels (-1=short, 0=neutral, +1=long, -99=invalid)

    Note:
        Labels with value -99 are invalid and should be filtered before training.
    """
    # Build cache key from barrier params and data identifier
    full_cache_key = (upper_mult, lower_mult, max_holding_bars, cache_key)
    if use_cache:
        with _label_cache_lock:
            if full_cache_key in _label_cache:
                # Move to end (most recently used) for LRU behavior
                _label_cache.move_to_end(full_cache_key)
                cached = _label_cache[full_cache_key]
                # Return cached labels if data length matches (basic sanity check)
                if len(cached) == len(data):
                    return cached

    # Ensure data has ATR column (compute if missing)
    atr_col = f"atr_{atr_period}"
    if atr_col not in data.columns:
        # Compute ATR inline
        data = data.copy()  # Don't modify original
        data[atr_col] = _compute_atr(data, atr_period)

    # Create labeler config
    config = TripleBarrierConfig(
        upper_mult=upper_mult,
        lower_mult=lower_mult,
        horizon=max_holding_bars,
        atr_period=atr_period,
        atr_column=atr_col,
        apply_transaction_costs=True,  # Include costs for realistic optimization
    )

    labeler = TripleBarrierLabeler(config)

    # Generate labels
    result = labeler.compute_labels(data, horizon=max_holding_bars)
    labels = result.labels

    # Cache for reuse with LRU eviction (Phase 29: bounded cache)
    if use_cache:
        with _label_cache_lock:
            _label_cache[full_cache_key] = labels.copy()
            # Evict oldest entry if cache exceeds maxsize
            while len(_label_cache) > LABEL_CACHE_MAXSIZE:
                _label_cache.popitem(last=False)  # Remove oldest (first) entry

    return labels


def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """
    Compute Average True Range (ATR) for a DataFrame.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default: 14)

    Returns:
        np.ndarray: ATR values
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = np.maximum(np.maximum(tr1, tr2), tr3)

    # EMA of true range
    alpha = 1.0 / period  # Wilder's EMA, matching triple_barrier.py and backtest.py
    atr = np.zeros_like(true_range)
    atr[0] = true_range[0]
    for i in range(1, len(true_range)):
        atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i - 1]

    return atr


def clear_label_cache() -> None:
    """Clear the label cache. Call between optimization runs to free memory."""
    with _label_cache_lock:
        _label_cache.clear()


def create_5d_objective(
    model_family: ModelFamily,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    train_labels: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    timeframes: list[str] | None = None,
    model_name: str | None = None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    metric_name: str = "validation_score",
    min_features: int = MIN_FEATURES,
    max_features_to_search: int = MAX_FEATURES_TO_SEARCH,
    study_name: str = "",
    generate_labels_per_trial: bool = True,
    atr_period: int = 14,
) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function that optimizes all 5 dimensions.

    This factory creates an Optuna objective that samples:
    1. Triple barrier parameters (upper_mult, lower_mult, max_holding_bars)
    2. Feature selection from base set
    3. Feature parameters (scaling around defaults)
    4. Feature timeframes
    5. Model hyperparameters

    CRITICAL: When generate_labels_per_trial=True (default), labels are generated
    INSIDE each trial using that trial's barrier parameters. This makes labels
    model-specific rather than fixed across all trials.

    Args:
        model_family: Model family to optimize for
        train_data: Training DataFrame with OHLCV columns (required for label generation)
        val_data: Validation DataFrame with OHLCV columns
        train_labels: Pre-computed training labels (fallback if generate_labels_per_trial=False)
        val_labels: Pre-computed validation labels (fallback if generate_labels_per_trial=False)
        timeframes: Available timeframes (default: ["5min", "15min", "30min", "60min"])
        model_name: Specific model name (optional, for precise hyperparameter space)
        metric_fn: Validation metric function (y_true, y_pred) -> float
        metric_name: Name of the metric for logging
        min_features: Minimum number of features to select
        max_features_to_search: Maximum features to include in search space
        study_name: Optuna study name for attribution
        generate_labels_per_trial: If True (default), generate labels inside each trial
            using the trial's barrier parameters. If False, use pre-computed labels.
        atr_period: Period for ATR calculation when generating labels (default: 14)

    Returns:
        Optuna objective function that returns validation metric

    Example:
        >>> # Train and val data must be DataFrames with OHLCV for label generation
        >>> objective = create_5d_objective(
        ...     model_family=ModelFamily.BOOSTING,
        ...     train_data=train_df,  # DataFrame with 'open', 'high', 'low', 'close'
        ...     val_data=val_df,
        ...     timeframes=["5min", "15min"],
        ... )
        >>> study = optuna.create_study(direction="maximize")
        >>> study.optimize(objective, n_trials=100)
    """
    # Set default timeframes
    timeframes = timeframes or DEFAULT_TIMEFRAMES

    # Get base features for this model family
    base_features = get_base_features(model_family)

    # Limit search space for efficiency
    searchable_features = base_features[:max_features_to_search]

    # Default metric function (Sharpe ratio-based for trading profitability)
    # CRITICAL: Optimize for Sharpe ratio, not F1 score (Phase 12A-1)
    if metric_fn is None:
        # Infer annualization factor from data frequency instead of hardcoding 252*78
        _annualization = _infer_annualization_factor(train_data)

        def default_metric_sharpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """
            Sharpe-like metric for trading profitability optimization.

            Instead of optimizing F1 (classification accuracy), we optimize a
            Sharpe-like metric that rewards:
            - Higher directional accuracy
            - Consistent returns (low variance)
            - Selective trading (not over-trading)

            Returns:
                Sharpe-like ratio (mean return / std return), annualized.
            """
            # Penalize if model doesn't trade enough (too many neutral predictions)
            trade_mask = y_pred != 0  # Assuming 0 is neutral
            trade_rate = trade_mask.mean()

            if trade_rate < DEFAULT_MIN_TRADE_RATE:
                # Model is too selective, penalize
                return 0.0

            # Simulate returns: +1 for correct, -1 for incorrect
            # Scale by confidence (simplified: all trades equal size)
            returns = np.where(y_pred == y_true, 1.0, -1.0)

            # Only count trades (non-neutral predictions)
            if trade_mask.sum() > 0:
                trade_returns = returns[trade_mask]
            else:
                return 0.0

            # Calculate Sharpe-like ratio
            mean_return = trade_returns.mean()
            std_return = trade_returns.std()

            if std_return < 1e-8:
                # No variance - return mean if positive, else 0
                return max(0.0, mean_return)

            # Annualized Sharpe (derived from data frequency)
            sharpe = mean_return / std_return * np.sqrt(_annualization)

            # Penalize very low trade rates (linear ramp between min and preferred)
            if trade_rate < DEFAULT_PREFERRED_TRADE_RATE:
                sharpe *= trade_rate / DEFAULT_PREFERRED_TRADE_RATE

            return float(max(0.0, sharpe))

        metric_fn = default_metric_sharpe

    # Validate data inputs when label generation is enabled
    if generate_labels_per_trial:
        required_cols = {"open", "high", "low", "close"}
        if isinstance(train_data, pd.DataFrame):
            missing = required_cols - set(train_data.columns)
            if missing:
                raise ValueError(
                    f"train_data missing required OHLCV columns for label generation: {missing}. "
                    f"Either provide a DataFrame with OHLCV columns or set generate_labels_per_trial=False."
                )
        else:
            raise ValueError(
                "train_data must be a DataFrame with OHLCV columns when generate_labels_per_trial=True. "
                "Either provide a DataFrame or set generate_labels_per_trial=False."
            )

        if isinstance(val_data, pd.DataFrame):
            missing = required_cols - set(val_data.columns)
            if missing:
                raise ValueError(
                    f"val_data missing required OHLCV columns for label generation: {missing}. "
                    f"Either provide a DataFrame with OHLCV columns or set generate_labels_per_trial=False."
                )
        else:
            raise ValueError(
                "val_data must be a DataFrame with OHLCV columns when generate_labels_per_trial=False. "
                "Either provide a DataFrame or set generate_labels_per_trial=False."
            )

        # Pre-compute ATR on train and val data ONCE before trials start
        # This avoids recomputing ATR inside each trial (significant speedup)
        atr_col = f"atr_{atr_period}"
        if atr_col not in train_data.columns:
            train_data = train_data.copy()
            train_data[atr_col] = _compute_atr(train_data, atr_period)
        if atr_col not in val_data.columns:
            val_data = val_data.copy()
            val_data[atr_col] = _compute_atr(val_data, atr_period)

    def objective(trial: optuna.Trial) -> float:
        """
        5-dimension Optuna objective function.

        Samples all 5 dimensions and builds a FeatureSpec.
        Labels are generated per-trial when generate_labels_per_trial=True.
        """
        start_time = time.time()

        try:
            # ==================================================================
            # Dimension 1: Triple Barrier Parameters
            # ==================================================================
            upper_mult = trial.suggest_float("upper_mult", *UPPER_MULT_RANGE)
            lower_mult = trial.suggest_float("lower_mult", *LOWER_MULT_RANGE)
            max_holding_bars = trial.suggest_int("max_holding_bars", *MAX_HOLDING_BARS_RANGE)

            # ==================================================================
            # Dimension 2: Feature Selection
            # ==================================================================
            selected_features: list[str] = []

            for feat in searchable_features:
                # Use categorical for feature inclusion
                include = trial.suggest_categorical(f"use_{feat}", [True, False])
                if include:
                    selected_features.append(feat)

            # Ensure minimum features
            if len(selected_features) < min_features:
                # Add features to meet minimum (deterministically based on trial)
                # Use set difference for O(1) lookup instead of O(n) list comprehension
                remaining = list(set(searchable_features) - set(selected_features))
                rng = np.random.RandomState(trial.number)
                rng.shuffle(remaining)
                needed = min_features - len(selected_features)
                selected_features.extend(remaining[:needed])

            # ==================================================================
            # Dimension 3: Feature Parameters
            # ==================================================================
            feature_params: dict[str, dict[str, Any]] = {}

            for feat in selected_features:
                default_params = DEFAULT_FEATURE_PARAMS.get(feat, {})
                if not default_params:
                    continue

                feat_params: dict[str, Any] = {}
                for param_name, default_value in default_params.items():
                    # Scale around default value (0.5x to 2x for integers, same for floats)
                    if isinstance(default_value, int):
                        low = max(1, default_value // 2)
                        high = default_value * 2
                        feat_params[param_name] = trial.suggest_int(
                            f"{feat}_{param_name}", low, high
                        )
                    elif isinstance(default_value, float):
                        low = default_value * 0.5
                        high = default_value * 2.0
                        feat_params[param_name] = trial.suggest_float(
                            f"{feat}_{param_name}", low, high
                        )
                    # Skip non-numeric parameters

                if feat_params:
                    feature_params[feat] = feat_params

            # ==================================================================
            # Dimension 4: Feature Timeframes
            # ==================================================================
            feature_timeframes: dict[str, str] = {}

            for feat in selected_features:
                tf = trial.suggest_categorical(f"{feat}_tf", timeframes)
                feature_timeframes[feat] = tf

            # ==================================================================
            # Dimension 5: Model Hyperparameters
            # ==================================================================
            hyperparameters = suggest_hyperparameters_by_family(trial, model_family, model_name)

            # ==================================================================
            # Build FeatureSpec
            # Note: We store upper_mult/lower_mult as profit_threshold/loss_threshold
            # since FeatureSpec uses these names. The values are ATR multipliers.
            # ==================================================================
            spec = FeatureSpec(
                profit_threshold=upper_mult,  # ATR multiplier for upper barrier
                loss_threshold=lower_mult,  # ATR multiplier for lower barrier
                max_holding_bars=max_holding_bars,
                selected_features=selected_features,
                feature_params=feature_params,
                feature_timeframes=feature_timeframes,
                hyperparameters=hyperparameters,
                model_name=model_name or model_family.value,
                optuna_trial_id=trial.number,
                optuna_study_name=study_name,
            )

            # Store spec in trial for later retrieval
            trial.set_user_attr("feature_spec", spec.to_dict())
            trial.set_user_attr("metric_name", metric_name)

            # ==================================================================
            # Generate Labels (per-trial or use pre-computed)
            # ==================================================================
            if generate_labels_per_trial:
                # Generate labels INSIDE the trial using this trial's barrier params
                # This makes labels model-specific!
                trial_train_labels = generate_labels_for_trial(
                    data=train_data,
                    upper_mult=upper_mult,
                    lower_mult=lower_mult,
                    max_holding_bars=max_holding_bars,
                    atr_period=atr_period,
                    cache_key=f"train_{study_name}",
                    use_cache=True,
                )
                trial_val_labels = generate_labels_for_trial(
                    data=val_data,
                    upper_mult=upper_mult,
                    lower_mult=lower_mult,
                    max_holding_bars=max_holding_bars,
                    atr_period=atr_period,
                    cache_key=f"val_{study_name}",
                    use_cache=True,
                )

                # Filter out invalid labels (-99)
                train_valid_mask = trial_train_labels != -99
                val_valid_mask = trial_val_labels != -99

                # Store label statistics
                n_train_valid = int(train_valid_mask.sum())
                n_val_valid = int(val_valid_mask.sum())

                if n_train_valid < 100 or n_val_valid < 50:
                    # Degenerate trial — signal to TPE sampler
                    logger.warning(
                        f"Trial {trial.number}: Insufficient valid labels "
                        f"(train={n_train_valid}, val={n_val_valid}). Skipping."
                    )
                    return float("-inf")

                current_train_labels = trial_train_labels
                current_val_labels = trial_val_labels

            elif train_labels is not None and val_labels is not None:
                # Use pre-computed labels (fallback mode)
                current_train_labels = train_labels
                current_val_labels = val_labels
                train_valid_mask = current_train_labels != -99
                val_valid_mask = current_val_labels != -99
            else:
                # Degenerate trial — signal to TPE sampler
                logger.warning(
                    f"Trial {trial.number}: No labels available. "
                    "Set generate_labels_per_trial=True or provide pre-computed labels."
                )
                return float("-inf")

            # ==================================================================
            # Compute Validation Metric
            # ==================================================================
            validation_score = _compute_validation_metric(
                spec=spec,
                train_data=train_data,
                val_data=val_data,
                train_labels=current_train_labels,
                val_labels=current_val_labels,
                train_valid_mask=train_valid_mask,
                val_valid_mask=val_valid_mask,
                metric_fn=metric_fn,
                model_family=model_family,
            )

            # Store optimization time
            optimization_time = time.time() - start_time
            trial.set_user_attr("optimization_time", optimization_time)

            # Store additional metrics including label distribution
            additional_metrics: dict[str, Any] = {
                "n_features": len(selected_features),
                "n_timeframes": len(spec.unique_timeframes),
            }

            # Add label distribution if we generated labels
            if generate_labels_per_trial:
                additional_metrics["n_train_valid"] = int(train_valid_mask.sum())
                additional_metrics["n_val_valid"] = int(val_valid_mask.sum())
                # Distribution of labels
                for label_val, label_name in [(-1, "short"), (0, "neutral"), (1, "long")]:
                    additional_metrics[f"val_{label_name}_pct"] = float(
                        (current_val_labels[val_valid_mask] == label_val).mean() * 100
                    )

            trial.set_user_attr("additional_metrics", additional_metrics)

            return validation_score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("-inf")

    return objective


def _compute_validation_metric(
    spec: FeatureSpec,
    train_data: pd.DataFrame | np.ndarray,
    val_data: pd.DataFrame | np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    train_valid_mask: np.ndarray,
    val_valid_mask: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    model_family: ModelFamily = ModelFamily.BOOSTING,
) -> float:
    """
    Compute validation metric using a model matched to the target family.

    For BOOSTING targets, uses XGBoost with the trial's actual hyperparameters
    so the optimization signal matches the model being tuned.
    For NEURAL/TRANSFORMER targets, uses LightGBM as a fast proxy since
    training a neural network per trial is prohibitively slow.
    For CLASSICAL targets, uses LogisticRegression.

    Args:
        spec: FeatureSpec with configuration (including hyperparameters)
        train_data: Training data (DataFrame or array)
        val_data: Validation data (DataFrame or array)
        train_labels: Training labels (may contain -99 invalid markers)
        val_labels: Validation labels (may contain -99 invalid markers)
        train_valid_mask: Boolean mask for valid training samples
        val_valid_mask: Boolean mask for valid validation samples
        metric_fn: Metric function (y_true, y_pred) -> float
        model_family: Target model family for proxy selection

    Returns:
        Validation metric value
    """
    try:
        # Extract feature columns (exclude OHLCV for training)
        ohlcv_cols = {"open", "high", "low", "close", "volume"}

        if isinstance(train_data, pd.DataFrame):
            # Use the trial's selected features if available, falling back to all non-OHLCV
            feature_cols: list[str] = []
            if spec.selected_features:
                feature_cols = [f for f in spec.selected_features if f in train_data.columns]
            if not spec.selected_features or not feature_cols:
                feature_cols = [c for c in train_data.columns if c.lower() not in ohlcv_cols]
            if not feature_cols:
                # If no feature columns, use all numeric columns
                feature_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
            X_train = train_data[feature_cols].values
            X_val = val_data[feature_cols].values
        else:
            X_train = train_data
            X_val = val_data

        # Filter to valid samples only
        X_train_valid = X_train[train_valid_mask]
        y_train_valid = train_labels[train_valid_mask]
        X_val_valid = X_val[val_valid_mask]
        y_val_valid = val_labels[val_valid_mask]

        # Limit features for speed during optimization.
        # Select top features by importance (not positional index) so that
        # features at index >50 are not permanently excluded.
        max_proxy_features = 50
        if X_train_valid.shape[1] <= max_proxy_features:
            X_train_sub = X_train_valid
            X_val_sub = X_val_valid
        else:
            try:
                import lightgbm as lgb

                quick_model = lgb.LGBMClassifier(
                    n_estimators=10,
                    max_depth=4,
                    learning_rate=0.1,
                    num_leaves=15,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                quick_model.fit(X_train_valid, y_train_valid)
                importances = quick_model.feature_importances_
                top_indices = np.argsort(importances)[::-1][:max_proxy_features]
                top_indices = np.sort(top_indices)  # preserve column order
                X_train_sub = X_train_valid[:, top_indices]
                X_val_sub = X_val_valid[:, top_indices]
            except Exception:
                # Fallback: random column sampling (unbiased, unlike positional)
                rng = np.random.RandomState(42)
                rand_indices = rng.choice(
                    X_train_valid.shape[1],
                    size=max_proxy_features,
                    replace=False,
                )
                rand_indices = np.sort(rand_indices)
                X_train_sub = X_train_valid[:, rand_indices]
                X_val_sub = X_val_valid[:, rand_indices]

        # Use a model from the same family as the optimization target.
        # For boosting, use XGBoost with the trial's actual hyperparameters.
        # For neural/transformer, use LightGBM as a fast proxy (training a
        # neural net per trial is too slow; the optimization signal for
        # feature/label quality transfers well from boosting proxies).
        if model_family == ModelFamily.BOOSTING:
            import xgboost as xgb

            hp = spec.hyperparameters
            clf = xgb.XGBClassifier(
                n_estimators=hp.get("n_estimators", 100),
                max_depth=hp.get("max_depth", 5),
                learning_rate=hp.get("learning_rate", 0.1),
                min_child_weight=hp.get("min_child_weight", 1),
                subsample=hp.get("subsample", 0.8),
                colsample_bytree=hp.get("colsample_bytree", 0.8),
                reg_alpha=hp.get("reg_alpha", 0.0),
                reg_lambda=hp.get("reg_lambda", 1.0),
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        elif model_family in (ModelFamily.NEURAL, ModelFamily.TRANSFORMER):
            import lightgbm as lgb

            clf = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            # CLASSICAL, META_LEARNER, ENSEMBLE - use logistic regression
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(
                max_iter=500,
                random_state=42,
                n_jobs=-1,
            )

        clf.fit(X_train_sub, y_train_valid)
        predictions = clf.predict(X_val_sub)

        return metric_fn(y_val_valid, predictions)

    except Exception as e:
        logger.error(f"Validation metric computation failed: {e}")
        return float("-inf")


# =============================================================================
# STUDY HELPERS
# =============================================================================


def run_5d_optimization(
    model_family: ModelFamily,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    train_labels: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    n_trials: int = 100,
    timeframes: list[str] | None = None,
    model_name: str | None = None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    metric_name: str = "validation_score",
    study_name: str = "5d_optimization",
    random_state: int = 42,
    timeout: int | None = None,
    verbose: int = 1,
    generate_labels_per_trial: bool = True,
    atr_period: int = 14,
    # Artifact saving options
    save_artifacts: bool = True,
    run_id: str | None = None,
    output_dir: str = "experiments",
    save_top_k: int = 5,
    # DSR gate enforcement
    enforce_dsr_gate: bool = True,
) -> tuple[optuna.Study, FiveDimensionResult]:
    """
    Run complete 5-dimension optimization study.

    Convenience function that creates and runs an Optuna study
    with the 5D objective. Optionally saves FeatureSpec artifacts
    to disk for reproducibility.

    CRITICAL: When generate_labels_per_trial=True (default), labels are generated
    INSIDE each trial using that trial's barrier parameters. This requires
    train_data and val_data to be DataFrames with OHLCV columns.

    Args:
        model_family: Model family to optimize
        train_data: Training DataFrame with OHLCV columns (for label generation)
        val_data: Validation DataFrame with OHLCV columns
        train_labels: Pre-computed training labels (optional, fallback mode)
        val_labels: Pre-computed validation labels (optional, fallback mode)
        n_trials: Number of optimization trials
        timeframes: Available timeframes
        model_name: Specific model name
        metric_fn: Custom validation metric function (y_true, y_pred) -> float.
            When None (default), uses a Sharpe-like trading metric.
        metric_name: Name of the metric for logging and DSR gate decisions.
            DSR gate is only applied when the metric is Sharpe-like.
        study_name: Name for the study
        random_state: Random seed
        timeout: Optional timeout in seconds
        verbose: Verbosity level
        generate_labels_per_trial: If True (default), generate labels inside each trial
        atr_period: Period for ATR calculation (default: 14)
        save_artifacts: If True (default), save FeatureSpecs to disk after optimization
        run_id: Experiment run identifier (auto-generated if None)
        output_dir: Base output directory for artifacts (default: "experiments")
        save_top_k: Number of top trials to save (default: 5)

    Returns:
        Tuple of (study, best_result)

    Example:
        >>> study, best = run_5d_optimization(
        ...     model_family=ModelFamily.BOOSTING,
        ...     train_data=train_df,  # DataFrame with OHLCV columns
        ...     val_data=val_df,
        ...     n_trials=50,
        ...     save_artifacts=True,
        ...     run_id="exp_001",
        ... )
        >>> print(best.feature_spec)
        >>> # Specs saved to experiments/exp_001/feature_specs/
    """
    # Configure Optuna logging
    if verbose == 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    elif verbose == 1:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

    if verbose >= 1:
        print(f"\n{'='*60}")
        print("5-Dimension Optuna Optimization")
        print(f"  Model family: {model_family.value}")
        print(f"  Model name: {model_name or 'auto'}")
        print(f"  Trials: {n_trials}")
        print(f"  Timeframes: {timeframes or DEFAULT_TIMEFRAMES}")
        print(f"  Generate labels per trial: {generate_labels_per_trial}")
        print(f"{'='*60}")

    # Clear label cache before starting (ensures fresh labels)
    clear_label_cache()

    # Track whether the default Sharpe-like metric is used.
    # When metric_fn is None, create_5d_objective uses a built-in Sharpe proxy,
    # so DSR deflation is valid regardless of metric_name.
    _using_default_sharpe_metric = metric_fn is None

    # Create objective
    objective = create_5d_objective(
        model_family=model_family,
        train_data=train_data,
        val_data=val_data,
        train_labels=train_labels,
        val_labels=val_labels,
        timeframes=timeframes,
        model_name=model_name,
        metric_fn=metric_fn,
        metric_name=metric_name,
        study_name=study_name,
        generate_labels_per_trial=generate_labels_per_trial,
        atr_period=atr_period,
    )

    # Create study
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    # Run optimization with parallel trials (Phase 12A-7)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=-1,  # Use all available CPU cores for parallel trials
        show_progress_bar=(verbose >= 1),
    )

    # Get best result
    best_result = FiveDimensionResult.from_trial(study.best_trial)

    if verbose >= 1:
        print("\nOptimization complete:")
        print(f"  Best value: {study.best_value:.4f}")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best spec: {best_result.feature_spec}")

    # =========================================================================
    # Phase 4D: Deflated Sharpe Ratio Validation
    # DSR is only valid when the optimized metric is Sharpe-like (unbounded).
    # For bounded metrics (F1, accuracy), DSR math is statistically invalid.
    # =========================================================================
    _dsr_applicable = _using_default_sharpe_metric or is_sharpe_like_metric(metric_name)

    if _dsr_applicable:
        try:
            # Bonferroni correction: pass completed trial count so DSR
            # raises significance/deployment thresholds proportionally,
            # controlling family-wise error rate across all trials.
            n_completed = len(
                [t for t in study.trials if t.state.is_finished() and t.value is not None]
            )
            dsr_result = compute_dsr_from_optuna_study(
                study=study,
                deployment_threshold=0.5,
                num_tests=max(1, n_completed),
            )

            # Store DSR metrics in best_result
            best_result.additional_metrics.update(
                {
                    "dsr_raw_sharpe": dsr_result.sharpe_ratio,
                    "dsr_deflated_sharpe": dsr_result.deflated_sharpe,
                    "dsr_n_trials": dsr_result.n_trials,
                    "dsr_is_significant": dsr_result.is_significant,
                    "dsr_should_deploy": dsr_result.should_deploy,
                    "dsr_deflation_pct": dsr_result.get_deflation_pct(),
                    "dsr_risk_level": dsr_result.get_risk_level(),
                }
            )

            # Log DSR results
            logger.info(
                f"DSR Validation: Raw Sharpe={dsr_result.sharpe_ratio:.3f}, "
                f"Deflated Sharpe={dsr_result.deflated_sharpe:.3f} "
                f"({dsr_result.get_deflation_pct():.1f}% deflation), "
                f"Risk: {dsr_result.get_risk_level()}"
            )

            # Enforce DSR gate: reject strategies that appear good only due to selection bias
            if enforce_dsr_gate:
                proceed, reason = dsr_gate(dsr_result)
                if not proceed:
                    logger.warning(f"DSR gate FAILED: {reason}")
                    raise ValueError(
                        f"DSR gate rejected optimization result: {reason}\n"
                        f"The observed Sharpe ratio ({dsr_result.sharpe_ratio:.3f}) is likely "
                        f"inflated by selection bias across {dsr_result.n_trials} trials.\n"
                        f"To disable this gate, set enforce_dsr_gate=False in PipelineConfig."
                    )
            elif not dsr_result.should_deploy:
                logger.warning(
                    f"DSR gate advisory: Deflated Sharpe ({dsr_result.deflated_sharpe:.3f}) "
                    f"below deployment threshold (0.5). Risk: {dsr_result.get_risk_level()}"
                )

            if verbose >= 1:
                print("\nDeflated Sharpe Ratio Analysis:")
                print(f"  Raw Sharpe: {dsr_result.sharpe_ratio:.3f}")
                print(f"  Deflated Sharpe: {dsr_result.deflated_sharpe:.3f}")
                print(f"  Deflation: {dsr_result.get_deflation_pct():.1f}%")
                print(f"  Risk Level: {dsr_result.get_risk_level()}")
                print(f"  Deploy Recommended: {dsr_result.should_deploy}")

        except Exception as e:
            logger.error(f"DSR computation failed: {e}")
            if verbose >= 1:
                print(f"\nDSR validation failed: {e}")
    else:
        logger.info(
            f"DSR gate skipped: metric '{metric_name}' is not a Sharpe-like ratio. "
            f"DSR deflation is only valid for unbounded Sharpe-like distributions."
        )
        if verbose >= 1:
            print(f"\nDSR skipped: metric '{metric_name}' is not Sharpe-like")

    # Save artifacts if requested
    if save_artifacts:
        from src.optimization.artifact_saver import save_optimization_results

        actual_run_id, saved_paths = save_optimization_results(
            study=study,
            run_id=run_id,
            output_dir=output_dir,
            save_all_trials=False,
            top_k=save_top_k,
            auto_generate_run_id=True,
        )

        if verbose >= 1:
            print(f"\nArtifacts saved to: {output_dir}/{actual_run_id}/feature_specs/")
            print(f"  Saved {len(saved_paths)} spec(s)")
            if "best" in saved_paths:
                print(f"  Best spec: {saved_paths['best']}")

    return study, best_result


def get_best_feature_spec(study: optuna.Study) -> FeatureSpec:
    """
    Extract the best FeatureSpec from a completed study.

    Args:
        study: Completed Optuna study

    Returns:
        FeatureSpec from the best trial
    """
    spec_dict = study.best_trial.user_attrs.get("feature_spec", {})
    if not spec_dict:
        raise ValueError("Study best trial does not contain feature_spec")
    return FeatureSpec.from_dict(spec_dict)


def get_top_k_specs(study: optuna.Study, k: int = 5) -> list[FeatureSpec]:
    """
    Get top-K FeatureSpecs from a study.

    Args:
        study: Completed Optuna study
        k: Number of top specs to return

    Returns:
        List of FeatureSpecs from top-K trials
    """
    # Get completed trials sorted by value
    completed = [t for t in study.trials if t.value is not None]
    sorted_trials = sorted(completed, key=lambda t: t.value or 0, reverse=True)

    specs = []
    for trial in sorted_trials[:k]:
        spec_dict = trial.user_attrs.get("feature_spec", {})
        if spec_dict:
            specs.append(FeatureSpec.from_dict(spec_dict))

    return specs


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Result container
    "FiveDimensionResult",
    # Main factory
    "create_5d_objective",
    # Label generation (per-trial)
    "generate_labels_for_trial",
    "clear_label_cache",
    # Convenience runner
    "run_5d_optimization",
    # Study helpers
    "get_best_feature_spec",
    "get_top_k_specs",
    # Hyperparameter suggestion
    "suggest_hyperparameters_by_family",
    # Constants (for customization)
    "UPPER_MULT_RANGE",
    "LOWER_MULT_RANGE",
    "MAX_HOLDING_BARS_RANGE",
    "MIN_FEATURES",
    "MAX_FEATURES_TO_SEARCH",
    "DEFAULT_TIMEFRAMES",
]
