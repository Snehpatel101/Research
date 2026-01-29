"""
ML for Dummies Configuration System.

Dead simple for beginners, full power for experts.

Users make HIGH-LEVEL choices:
    - Which model(s)?
    - Which symbol?
    - Which horizon(s)?
    - Optimize what?
    - Build ensemble?

The system figures out ALL the details automatically based on the model type.

Usage:
    # Absolute beginner - just works
    train("xgboost")

    # Pick symbol and horizon
    train("xgboost", symbol="MGC", horizon=20)

    # Want optimization
    train("xgboost", optimize="hyperparameters")

    # Multiple models with ensemble
    train(["xgboost", "lstm"], ensemble=True)

    # Advanced: override a smart default
    train("lstm", sequence_length=120)  # Override the default 60
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL DEFAULTS - ALL SMART DEFAULTS IN ONE PLACE
# =============================================================================

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # BOOSTING MODELS - Fast, interpretable, good baselines
    # -------------------------------------------------------------------------
    "xgboost": {
        "family": "boosting",
        "timeframe": "15min",
        "features": "full",  # ~100 engineered indicators + MTF
        "sequence_length": None,  # Not a sequence model
        "batch_size": None,  # Not batched (fits in memory)
        "description": "Gradient boosting. Fast and interpretable. Great baseline.",
    },
    "lightgbm": {
        "family": "boosting",
        "timeframe": "15min",
        "features": "full",
        "sequence_length": None,
        "batch_size": None,
        "description": "Light gradient boosting. Faster than XGBoost, similar accuracy.",
    },
    "catboost": {
        "family": "boosting",
        "timeframe": "15min",
        "features": "full",
        "sequence_length": None,
        "batch_size": None,
        "description": "Categorical boosting. Handles categoricals natively.",
    },
    # -------------------------------------------------------------------------
    # CLASSICAL ML - Simple, interpretable baselines
    # -------------------------------------------------------------------------
    "random_forest": {
        "family": "classical",
        "timeframe": "15min",
        "features": "standard",  # ~70 features, no MTF
        "sequence_length": None,
        "batch_size": None,
        "description": "Random forest. Simple, robust baseline.",
    },
    "logistic": {
        "family": "classical",
        "timeframe": "15min",
        "features": "minimal",  # ~30 core features
        "sequence_length": None,
        "batch_size": None,
        "description": "Logistic regression. Simplest baseline, highly interpretable.",
    },
    "svm": {
        "family": "classical",
        "timeframe": "15min",
        "features": "minimal",
        "sequence_length": None,
        "batch_size": None,
        "description": "Support vector machine. Good for small datasets.",
    },
    # -------------------------------------------------------------------------
    # RECURRENT NEURAL NETWORKS - Sequence learning
    # -------------------------------------------------------------------------
    "lstm": {
        "family": "neural",
        "timeframe": "5min",
        "features": "sequence",  # ~80 features, includes wavelets
        "sequence_length": 60,  # 60 timesteps
        "batch_size": 256,
        "description": "LSTM network. Learns long-term dependencies in sequences.",
    },
    "gru": {
        "family": "neural",
        "timeframe": "5min",
        "features": "sequence",
        "sequence_length": 60,
        "batch_size": 256,
        "description": "GRU network. Faster LSTM variant with similar performance.",
    },
    # -------------------------------------------------------------------------
    # CONVOLUTIONAL NEURAL NETWORKS - Pattern detection
    # -------------------------------------------------------------------------
    "tcn": {
        "family": "neural",
        "timeframe": "5min",
        "features": "sequence",
        "sequence_length": 120,  # Longer context for dilated convolutions
        "batch_size": 256,
        "description": "Temporal convolutional network. Fast parallel training.",
    },
    "inceptiontime": {
        "family": "cnn",
        "timeframe": "5min",
        "features": "sequence",
        "sequence_length": 96,
        "batch_size": 128,  # More memory intensive
        "description": "InceptionTime. Multi-scale pattern detection.",
    },
    "resnet1d": {
        "family": "cnn",
        "timeframe": "5min",
        "features": "sequence",
        "sequence_length": 96,
        "batch_size": 128,
        "description": "1D ResNet. Deep residual learning for time series.",
    },
    # -------------------------------------------------------------------------
    # TRANSFORMERS - State-of-the-art attention-based models
    # -------------------------------------------------------------------------
    "transformer": {
        "family": "transformer",
        "timeframe": "1min",
        "features": "raw",  # Raw OHLCV only (5 features)
        "sequence_length": 96,
        "batch_size": 128,
        "description": "Vanilla transformer. Attention-based sequence model.",
    },
    "patchtst": {
        "family": "transformer",
        "timeframe": "1min",
        "features": "raw",  # Raw OHLCV only
        "sequence_length": 96,  # Patch-based, longer context
        "batch_size": 64,  # Memory intensive
        "description": "PatchTST. State-of-the-art time series transformer.",
    },
    "itransformer": {
        "family": "transformer",
        "timeframe": "1min",
        "features": "raw",
        "sequence_length": 96,
        "batch_size": 64,
        "description": "iTransformer. Inverted transformer for time series.",
    },
    "tft": {
        "family": "transformer",
        "timeframe": "5min",
        "features": "tft",  # Limited features (raw + some indicators)
        "sequence_length": 60,
        "batch_size": 128,
        "description": "Temporal Fusion Transformer. Interpretable attention.",
    },
    "nbeats": {
        "family": "transformer",
        "timeframe": "5min",
        "features": "minimal_raw",  # Just close + volume
        "sequence_length": 60,
        "batch_size": 128,
        "description": "N-BEATS. Pure deep learning forecaster.",
    },
    # -------------------------------------------------------------------------
    # META-LEARNERS - For ensemble stacking
    # -------------------------------------------------------------------------
    "ridge_meta": {
        "family": "meta_learner",
        "timeframe": None,  # Uses base model predictions
        "features": None,
        "sequence_length": None,
        "batch_size": None,
        "description": "Ridge meta-learner. Fast linear stacking.",
    },
    "mlp_meta": {
        "family": "meta_learner",
        "timeframe": None,
        "features": None,
        "sequence_length": None,
        "batch_size": 256,
        "description": "MLP meta-learner. Non-linear stacking.",
    },
    "calibrated_meta": {
        "family": "meta_learner",
        "timeframe": None,
        "features": None,
        "sequence_length": None,
        "batch_size": None,
        "description": "Calibrated meta-learner. Probability calibration.",
    },
    "xgboost_meta": {
        "family": "meta_learner",
        "timeframe": None,
        "features": None,
        "sequence_length": None,
        "batch_size": None,
        "description": "XGBoost meta-learner. Non-linear feature interactions.",
    },
    # -------------------------------------------------------------------------
    # ENSEMBLES
    # -------------------------------------------------------------------------
    "voting": {
        "family": "ensemble",
        "timeframe": None,
        "features": None,
        "sequence_length": None,
        "batch_size": None,
        "description": "Voting ensemble. Simple average of base models.",
    },
    "stacking": {
        "family": "ensemble",
        "timeframe": None,
        "features": None,
        "sequence_length": None,
        "batch_size": None,
        "description": "Stacking ensemble. Meta-learner on OOF predictions.",
    },
    "blending": {
        "family": "ensemble",
        "timeframe": None,
        "features": None,
        "sequence_length": None,
        "batch_size": None,
        "description": "Blending ensemble. Holdout-based stacking.",
    },
}

# Feature set definitions
FEATURE_SETS = {
    "full": "~100 engineered indicators + MTF features",
    "standard": "~70 engineered indicators, no MTF",
    "sequence": "~80 indicators + wavelets, optimized for RNNs",
    "minimal": "~30 core momentum/volatility features",
    "raw": "Raw OHLCV only (5 features) - model learns representations",
    "minimal_raw": "Just close + volume (2 features)",
    "tft": "~20 features: raw OHLCV + core indicators",
}

# Default optimization settings
OPTIMIZATION_DEFAULTS = {
    "features": {"n_trials": 30, "timeout": 600},  # 10 min max
    "hyperparameters": {"n_trials": 50, "timeout": 1800},  # 30 min max
    "both": {"feature_trials": 30, "hyperparam_trials": 50, "timeout": 3600},
}


# =============================================================================
# USER-FACING CONFIG - ONLY WHAT USERS NEED TO SET
# =============================================================================


@dataclass
class SmartConfig:
    """
    ML for Dummies configuration.

    Only contains what users actually need to decide.
    Everything else is figured out automatically.
    """

    # What model(s) to train?
    models: list[str] = field(default_factory=lambda: ["xgboost"])

    # What symbol to trade?
    symbol: str = "MES"

    # How far ahead to predict? (bars)
    horizons: list[int] = field(default_factory=lambda: [20])

    # Should the system tune things?
    optimize: Literal["none", "features", "hyperparameters", "both"] = "none"

    # Combine models into an ensemble?
    ensemble: bool = False

    # Which meta-learner for ensemble?
    meta_learner: str = "ridge_meta"

    # Where is the data?
    data_dir: Path = field(default_factory=lambda: Path("data/splits/scaled"))

    # Where to save outputs?
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))

    # Advanced overrides (experts only)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure models is a list
        if isinstance(self.models, str):
            self.models = [self.models]

        # Ensure horizons is a list
        if isinstance(self.horizons, int):
            self.horizons = [self.horizons]

        # Convert paths
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate models
        for model in self.models:
            if model not in MODEL_DEFAULTS:
                available = list(MODEL_DEFAULTS.keys())
                raise ValueError(f"Unknown model '{model}'. " f"Available: {available}")


# =============================================================================
# RESOLUTION LOGIC - MERGE USER CHOICES + SMART DEFAULTS
# =============================================================================


@dataclass
class ResolvedModelConfig:
    """
    Fully resolved configuration for a single model.

    Created by merging user choices with smart defaults.
    """

    name: str
    family: str
    timeframe: str
    features: str
    sequence_length: int | None
    batch_size: int | None
    optimize_features: bool
    optimize_hyperparams: bool
    feature_opt_trials: int
    hyperparam_opt_trials: int
    description: str

    # Any user overrides applied
    overrides_applied: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for compatibility."""
        return {
            "name": self.name,
            "family": self.family,
            "timeframe": self.timeframe,
            "features": self.features,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "optimize_features": self.optimize_features,
            "optimize_hyperparams": self.optimize_hyperparams,
            "feature_opt_trials": self.feature_opt_trials,
            "hyperparam_opt_trials": self.hyperparam_opt_trials,
        }


def resolve_model_config(
    model_name: str,
    optimize: str = "none",
    overrides: dict[str, Any] | None = None,
) -> ResolvedModelConfig:
    """
    Resolve a model's configuration by merging smart defaults with user overrides.

    Args:
        model_name: Name of the model (e.g., "xgboost", "lstm")
        optimize: What to optimize ("none", "features", "hyperparameters", "both")
        overrides: User overrides for specific settings

    Returns:
        ResolvedModelConfig with all settings filled in
    """
    overrides = overrides or {}

    # Get smart defaults for this model
    if model_name not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model '{model_name}'")

    defaults = MODEL_DEFAULTS[model_name]

    # Determine optimization flags
    optimize_features = optimize in ("features", "both")
    optimize_hyperparams = optimize in ("hyperparameters", "both")

    # Get optimization trial counts
    opt_settings = OPTIMIZATION_DEFAULTS.get(optimize, {})
    feature_trials = opt_settings.get("feature_trials", opt_settings.get("n_trials", 30))
    hyperparam_trials = opt_settings.get("hyperparam_trials", opt_settings.get("n_trials", 50))

    # Build resolved config, applying overrides
    return ResolvedModelConfig(
        name=model_name,
        family=defaults["family"],
        timeframe=overrides.get("timeframe", defaults["timeframe"]),
        features=overrides.get("features", defaults["features"]),
        sequence_length=overrides.get("sequence_length", defaults["sequence_length"]),
        batch_size=overrides.get("batch_size", defaults["batch_size"]),
        optimize_features=overrides.get("optimize_features", optimize_features),
        optimize_hyperparams=overrides.get("optimize_hyperparams", optimize_hyperparams),
        feature_opt_trials=overrides.get("feature_opt_trials", feature_trials),
        hyperparam_opt_trials=overrides.get("hyperparam_opt_trials", hyperparam_trials),
        description=defaults["description"],
        overrides_applied=overrides,
    )


def resolve_config(config: SmartConfig) -> dict[str, Any]:
    """
    Fully resolve a SmartConfig into all the details needed for training.

    This is where the magic happens - user makes simple choices,
    system figures out all the details.

    Args:
        config: User's SmartConfig with high-level choices

    Returns:
        Dict with fully resolved configuration for all models
    """
    resolved_models = []

    for model_name in config.models:
        # Get user overrides for this model
        model_overrides = config.overrides.get(model_name, {})

        # Resolve the model config
        resolved = resolve_model_config(
            model_name=model_name,
            optimize=config.optimize,
            overrides=model_overrides,
        )

        resolved_models.append(resolved)

    return {
        "symbol": config.symbol,
        "horizons": config.horizons,
        "models": resolved_models,
        "ensemble": {
            "enabled": config.ensemble,
            "meta_learner": config.meta_learner,
        },
        "paths": {
            "data_dir": config.data_dir,
            "output_dir": config.output_dir,
        },
    }


# =============================================================================
# THE SIMPLE API - train() function
# =============================================================================


def train(
    models: str | list[str],
    *,
    symbol: str = "MES",
    horizon: int | list[int] = 20,
    optimize: Literal["none", "features", "hyperparameters", "both"] = "none",
    ensemble: bool = False,
    meta_learner: str = "ridge_meta",
    data_dir: str | Path = "data/splits/scaled",
    output_dir: str | Path = "experiments/runs",
    # Advanced overrides
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Train ML model(s) on OHLCV time series data.

    This is the "ML for Dummies" entry point. Just tell it what you want,
    it figures out all the details.

    Args:
        models: Model name(s) to train. Examples:
            - "xgboost" (single model)
            - ["xgboost", "lstm"] (multiple models)
            - ["xgboost", "lstm", "patchtst"] (heterogeneous ensemble)

        symbol: Trading symbol. Default "MES".
            Examples: "MES", "MGC", "ES", "GC"

        horizon: Bars ahead to predict. Default 20.
            Examples: 5, 10, 15, 20, or [5, 10, 15, 20]

        optimize: What should the system optimize? Default "none".
            - "none": Use defaults (fast, good baseline)
            - "features": Find optimal feature subset per model
            - "hyperparameters": Tune model hyperparameters
            - "both": Full optimization (slowest, best results)

        ensemble: Combine models into an ensemble? Default False.
            If True and multiple models provided, builds a stacking ensemble.

        meta_learner: Which meta-learner for ensemble? Default "ridge_meta".
            Options: "ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"

        data_dir: Where is the data? Default "data/splits/scaled".

        output_dir: Where to save outputs? Default "experiments/runs".

        **kwargs: Advanced overrides. Can override any smart default.
            Examples:
                - sequence_length=120 (override LSTM sequence length)
                - batch_size=64 (override batch size)
                - timeframe="5min" (override training timeframe)

    Returns:
        Dict with training results, including:
            - metrics: Performance metrics for each model
            - models: Trained model objects
            - config: Resolved configuration that was used

    Examples:
        # Absolute beginner - just works
        >>> results = train("xgboost")

        # Pick symbol and horizon
        >>> results = train("xgboost", symbol="MGC", horizon=20)

        # Want optimization
        >>> results = train("xgboost", optimize="hyperparameters")

        # Multiple models
        >>> results = train(["xgboost", "lstm"])

        # Multiple models with ensemble
        >>> results = train(["xgboost", "lstm"], ensemble=True)

        # Full optimization
        >>> results = train(["xgboost", "lstm", "patchtst"],
        ...                 optimize="both",
        ...                 ensemble=True)

        # Advanced: override a smart default
        >>> results = train("lstm", sequence_length=120)

        # Advanced: per-model overrides
        >>> results = train(
        ...     ["xgboost", "lstm"],
        ...     xgboost={"timeframe": "5min"},  # Override xgboost's timeframe
        ...     lstm={"sequence_length": 120},   # Override lstm's sequence length
        ... )
    """
    # Normalize models to list
    if isinstance(models, str):
        models = [models]

    # Normalize horizons to list
    horizons = [horizon] if isinstance(horizon, int) else list(horizon)

    # Extract per-model overrides from kwargs
    overrides = {}
    remaining_kwargs = {}

    for key, value in kwargs.items():
        if key in MODEL_DEFAULTS:
            # This is a per-model override dict
            overrides[key] = value
        else:
            # This is a global override (applies to all models)
            remaining_kwargs[key] = value

    # If there are global overrides, apply them to all models
    if remaining_kwargs:
        for model_name in models:
            if model_name not in overrides:
                overrides[model_name] = {}
            overrides[model_name].update(remaining_kwargs)

    # Build the SmartConfig
    config = SmartConfig(
        models=models,
        symbol=symbol,
        horizons=horizons,
        optimize=optimize,
        ensemble=ensemble,
        meta_learner=meta_learner,
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        overrides=overrides,
    )

    # Resolve to full configuration
    resolved = resolve_config(config)

    # Log what we're doing
    logger.info("=" * 60)
    logger.info("ML FOR DUMMIES - Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Symbol: {config.symbol}")
    logger.info(f"Horizons: {config.horizons}")
    logger.info(f"Models: {[m.name for m in resolved['models']]}")
    logger.info(f"Optimize: {config.optimize}")
    logger.info(f"Ensemble: {config.ensemble}")
    logger.info("")

    for model_cfg in resolved["models"]:
        logger.info(f"  {model_cfg.name}:")
        logger.info(f"    Timeframe: {model_cfg.timeframe}")
        logger.info(
            f"    Features: {model_cfg.features} ({FEATURE_SETS.get(model_cfg.features, 'custom')})"
        )
        if model_cfg.sequence_length:
            logger.info(f"    Sequence length: {model_cfg.sequence_length}")
        if model_cfg.batch_size:
            logger.info(f"    Batch size: {model_cfg.batch_size}")
        if model_cfg.overrides_applied:
            logger.info(f"    Overrides: {model_cfg.overrides_applied}")
        logger.info("")

    logger.info("=" * 60)

    # Actually run training via unified orchestrator
    from src.core.config import PipelineConfig

    # Convert resolved config to PipelineConfig
    pipeline_config = PipelineConfig(
        symbol=config.symbol,
        data_path=str(config.data_dir),
        output_dir=str(config.output_dir),
        models=[m.name for m in resolved["models"]],
        horizons=config.horizons,
        build_ensemble=config.ensemble,
        meta_learner=config.meta_learner if config.ensemble else None,
    )

    # Note: smart_config.py uses deprecated training flow
    # UnifiedTrainingOrchestrator requires DataFrame input which this flow doesn't provide
    # For now, return a stub result indicating migration needed
    logger.warning(
        "smart_config.py uses deprecated training flow. "
        "Consider migrating to MLPipeline for full functionality. "
        "UnifiedTrainingOrchestrator requires DataFrame input - use MLPipeline instead."
    )
    results = {
        "status": "deprecated_flow_requires_migration",
        "config": pipeline_config,
        "message": "Use MLPipeline or UnifiedTrainingOrchestrator directly with data",
    }

    # Add resolved config to results
    results["_config"] = resolved
    results["_smart_config"] = config

    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def list_models(family: str | None = None) -> list[str]:
    """
    List available models.

    Args:
        family: Filter by family (e.g., "boosting", "neural", "transformer")
                If None, returns all models.

    Returns:
        List of model names

    Examples:
        >>> list_models()
        ['xgboost', 'lightgbm', 'catboost', 'lstm', 'gru', ...]

        >>> list_models("boosting")
        ['xgboost', 'lightgbm', 'catboost']
    """
    if family is None:
        return list(MODEL_DEFAULTS.keys())

    return [name for name, defaults in MODEL_DEFAULTS.items() if defaults["family"] == family]


def describe_model(model_name: str) -> str:
    """
    Get a description of a model.

    Args:
        model_name: Name of the model

    Returns:
        Human-readable description

    Examples:
        >>> describe_model("xgboost")
        'Gradient boosting. Fast and interpretable. Great baseline.'
    """
    if model_name not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model '{model_name}'")

    return str(MODEL_DEFAULTS[model_name]["description"])


def show_defaults(model_name: str) -> dict[str, Any]:
    """
    Show the smart defaults for a model.

    Args:
        model_name: Name of the model

    Returns:
        Dict of smart defaults

    Examples:
        >>> show_defaults("lstm")
        {
            'family': 'neural',
            'timeframe': '5min',
            'features': 'sequence',
            'sequence_length': 60,
            'batch_size': 256,
            'description': 'LSTM network. Learns long-term dependencies in sequences.'
        }
    """
    if model_name not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model '{model_name}'")

    return MODEL_DEFAULTS[model_name].copy()


def preview_config(
    models: str | list[str],
    optimize: str = "none",
    **kwargs: Any,
) -> None:
    """
    Preview what configuration will be used without actually training.

    Useful for understanding what the system will do.

    Args:
        models: Model name(s)
        optimize: Optimization setting
        **kwargs: Any overrides

    Examples:
        >>> preview_config("lstm")
        lstm:
          Family: neural
          Timeframe: 5min
          Features: sequence (~80 indicators + wavelets)
          Sequence length: 60
          Batch size: 256
          Optimize features: False
          Optimize hyperparams: False

        >>> preview_config("lstm", optimize="both", sequence_length=120)
        lstm:
          Family: neural
          Timeframe: 5min
          Features: sequence (~80 indicators + wavelets)
          Sequence length: 120 (overridden from 60)
          Batch size: 256
          Optimize features: True (30 trials)
          Optimize hyperparams: True (50 trials)
    """
    if isinstance(models, str):
        models = [models]

    # Extract per-model overrides
    overrides = {}
    remaining = {}
    for key, value in kwargs.items():
        if key in MODEL_DEFAULTS:
            overrides[key] = value
        else:
            remaining[key] = value

    # Apply global overrides to all models
    if remaining:
        for model_name in models:
            if model_name not in overrides:
                overrides[model_name] = {}
            overrides[model_name].update(remaining)

    print("\nConfiguration Preview")
    print("=" * 50)

    for model_name in models:
        model_overrides = overrides.get(model_name, {})
        resolved = resolve_model_config(model_name, optimize, model_overrides)
        defaults = MODEL_DEFAULTS[model_name]

        print(f"\n{model_name}:")
        print(f"  Family: {resolved.family}")

        # Show timeframe (with override indicator)
        if resolved.timeframe != defaults["timeframe"] and defaults["timeframe"]:
            print(f"  Timeframe: {resolved.timeframe} (overridden from {defaults['timeframe']})")
        else:
            print(f"  Timeframe: {resolved.timeframe}")

        # Show features
        feature_desc = FEATURE_SETS.get(resolved.features, "custom")
        print(f"  Features: {resolved.features} ({feature_desc})")

        # Show sequence length (with override indicator)
        if resolved.sequence_length:
            if resolved.sequence_length != defaults["sequence_length"]:
                print(
                    f"  Sequence length: {resolved.sequence_length} (overridden from {defaults['sequence_length']})"
                )
            else:
                print(f"  Sequence length: {resolved.sequence_length}")

        # Show batch size
        if resolved.batch_size:
            if resolved.batch_size != defaults["batch_size"]:
                print(
                    f"  Batch size: {resolved.batch_size} (overridden from {defaults['batch_size']})"
                )
            else:
                print(f"  Batch size: {resolved.batch_size}")

        # Show optimization settings
        if resolved.optimize_features:
            print(f"  Optimize features: Yes ({resolved.feature_opt_trials} trials)")
        else:
            print("  Optimize features: No")

        if resolved.optimize_hyperparams:
            print(f"  Optimize hyperparams: Yes ({resolved.hyperparam_opt_trials} trials)")
        else:
            print("  Optimize hyperparams: No")

    print("\n" + "=" * 50)


def quick_compare(models: list[str] | None = None) -> None:
    """
    Quick comparison of model defaults.

    Args:
        models: List of models to compare. If None, shows common models.

    Examples:
        >>> quick_compare()
        Model           Family      Timeframe  Features   Seq Len  Batch
        --------------- ----------- ---------- ---------- -------- ------
        xgboost         boosting    15min      full       -        -
        lstm            neural      5min       sequence   60       256
        patchtst        transformer 1min       raw        96       64
    """
    if models is None:
        models = ["xgboost", "lightgbm", "lstm", "gru", "tcn", "patchtst", "itransformer"]

    # Header
    print(
        f"\n{'Model':<15} {'Family':<11} {'Timeframe':<10} {'Features':<10} {'Seq Len':<8} {'Batch':<6}"
    )
    print(
        "-" * 15 + " " + "-" * 11 + " " + "-" * 10 + " " + "-" * 10 + " " + "-" * 8 + " " + "-" * 6
    )

    for model_name in models:
        if model_name not in MODEL_DEFAULTS:
            continue

        d = MODEL_DEFAULTS[model_name]
        seq = str(d["sequence_length"]) if d["sequence_length"] else "-"
        batch = str(d["batch_size"]) if d["batch_size"] else "-"
        tf = d["timeframe"] if d["timeframe"] else "-"
        feat = d["features"] if d["features"] else "-"

        print(f"{model_name:<15} {d['family']:<11} {tf:<10} {feat:<10} {seq:<8} {batch:<6}")

    print()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core API
    "train",
    "SmartConfig",
    "ResolvedModelConfig",
    # Configuration data
    "MODEL_DEFAULTS",
    "FEATURE_SETS",
    "OPTIMIZATION_DEFAULTS",
    # Resolution
    "resolve_model_config",
    "resolve_config",
    # Helpers
    "list_models",
    "describe_model",
    "show_defaults",
    "preview_config",
    "quick_compare",
]
