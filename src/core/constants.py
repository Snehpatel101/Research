"""
Core constants - Canonical values for the ML Factory.

PHASE_0: This is the SINGLE SOURCE OF TRUTH for all constant values.
All other modules should import from here.

Defines:
- CANONICAL_TIMEFRAMES: 9 intraday timeframes
- MODEL_FAMILIES: 23 models across 5 families
- MODEL_DATA_RANKS: Model -> tensor rank mapping
- MODEL_ADAPTER_MAP: Model -> adapter type mapping
- DEFAULT_* values for pipeline configuration
"""

# =============================================================================
# TIMEFRAMES - 9 canonical intraday timeframes
# =============================================================================

CANONICAL_TIMEFRAMES: list[str] = [
    "1min",
    "5min",
    "10min",
    "15min",
    "20min",
    "25min",
    "30min",
    "45min",
    "60min",
]

# Base timeframe (source data resolution)
BASE_TIMEFRAME: str = "1min"

# Default MTF timeframes for multi-timeframe features
# Canonical default: covers short, medium, and long-term patterns
# All other modules should import from here
DEFAULT_MTF_TIMEFRAMES: list[str] = ["1min", "5min", "15min", "60min"]


# =============================================================================
# HORIZONS - Prediction horizons (in bars)
# =============================================================================

DEFAULT_HORIZONS: list[int] = [5, 10, 15, 20]
DEFAULT_HORIZON: int = 20


# =============================================================================
# DATA SPLIT RATIOS
# =============================================================================

DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


# =============================================================================
# PURGE/EMBARGO - Leakage prevention
# =============================================================================

DEFAULT_PURGE_BARS: int = 60  # Gap before test fold
DEFAULT_EMBARGO_BARS: int = 1440  # Gap after test fold (1 day at 1min)
MINUTES_PER_DAY: int = 1440  # 24 * 60 minutes


# =============================================================================
# FINANCIAL CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR: int = 252  # Standard US equity trading days
DEFAULT_ANNUALIZATION_FACTOR: float = 252.0  # For Sharpe ratio annualization
DEFAULT_BOOTSTRAP_SAMPLES: int = 1000  # Bootstrap resampling iterations


# =============================================================================
# MODEL FAMILIES - 23 models across 5 families
# =============================================================================

MODEL_FAMILIES: dict[str, list[str]] = {
    "boosting": ["xgboost", "lightgbm", "catboost"],
    "classical": ["random_forest", "logistic", "svm"],
    "neural": [
        "lstm",
        "gru",
        "tcn",
        "tft",
        "nbeats",
        "inceptiontime",
        "resnet1d",
    ],
    "transformer": ["transformer", "patchtst", "itransformer"],
    "ensemble": ["voting", "stacking", "blending"],
    "meta_learner": ["ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"],
}

# Total model count verification
ALL_MODELS: list[str] = [model for models in MODEL_FAMILIES.values() for model in models]
assert len(ALL_MODELS) == 23, f"Expected 23 models, got {len(ALL_MODELS)}"


# =============================================================================
# MODEL -> FAMILY MAPPING
# =============================================================================

MODEL_TO_FAMILY: dict[str, str] = {
    model: family for family, models in MODEL_FAMILIES.items() for model in models
}


# =============================================================================
# MODEL -> DATA RANK MAPPING (2D/3D/4D)
# =============================================================================
# Derived from MODEL_CONTRACTS - single source of truth


def _get_model_data_ranks() -> dict[str, int]:
    """Derive MODEL_DATA_RANKS from MODEL_CONTRACTS (lazy import)."""
    from src.core.contracts.model_contract import MODEL_CONTRACTS

    return {name: contract.input_rank.value for name, contract in MODEL_CONTRACTS.items()}


def _get_model_adapter_map() -> dict[str, str]:
    """Derive MODEL_ADAPTER_MAP from MODEL_CONTRACTS (lazy import)."""
    from src.core.contracts.model_contract import MODEL_CONTRACTS

    return {name: contract.adapter_id for name, contract in MODEL_CONTRACTS.items()}


# Lazy-initialized caches
_model_data_ranks_cache: dict[str, int] | None = None
_model_adapter_map_cache: dict[str, str] | None = None


def get_model_data_ranks() -> dict[str, int]:
    """Get model -> data rank mapping (cached)."""
    global _model_data_ranks_cache
    if _model_data_ranks_cache is None:
        _model_data_ranks_cache = _get_model_data_ranks()
    return _model_data_ranks_cache


def get_model_adapter_map() -> dict[str, str]:
    """Get model -> adapter mapping (cached)."""
    global _model_adapter_map_cache
    if _model_adapter_map_cache is None:
        _model_adapter_map_cache = _get_model_adapter_map()
    return _model_adapter_map_cache


# Backward compatibility: MODULE-LEVEL access via __getattr__
# This allows existing code like `from src.core.constants import MODEL_DATA_RANKS` to work
def __getattr__(name: str) -> dict[str, int] | dict[str, str]:
    """Lazy initialization for derived constants."""
    if name == "MODEL_DATA_RANKS":
        return get_model_data_ranks()
    elif name == "MODEL_ADAPTER_MAP":
        return get_model_adapter_map()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# NOTE: MODEL_DATA_RANKS and MODEL_ADAPTER_MAP are now derived from MODEL_CONTRACTS.
# Use get_model_data_ranks() and get_model_adapter_map() for direct access.
# Module-level MODEL_DATA_RANKS and MODEL_ADAPTER_MAP are available for backward
# compatibility via __getattr__ lazy initialization.


# =============================================================================
# FEATURE FAMILIES - 12 families, 162 base features
# =============================================================================

FEATURE_FAMILY_COUNTS: dict[str, int] = {
    "raw": 5,
    "momentum": 23,
    "moving_average": 16,
    "volatility": 25,
    "volume": 15,
    "trend": 6,
    "price": 12,
    "microstructure": 15,
    "entropy": 12,
    "wavelets": 15,
    "temporal": 9,
    "regime": 9,
}

TOTAL_BASE_FEATURES: int = sum(FEATURE_FAMILY_COUNTS.values())
assert TOTAL_BASE_FEATURES == 162, f"Expected 162 base features, got {TOTAL_BASE_FEATURES}"

# MTF adds ~30 features per higher timeframe (8 higher TFs)
MTF_FEATURES_PER_TF: int = 30
MTF_TOTAL_FEATURES: int = MTF_FEATURES_PER_TF * (len(CANONICAL_TIMEFRAMES) - 1)  # ~240


# =============================================================================
# SEQUENCE MODEL DEFAULTS
# =============================================================================

DEFAULT_SEQUENCE_LENGTH: int = 60
DEFAULT_HIDDEN_SIZE: int = 128
DEFAULT_NUM_LAYERS: int = 2
DEFAULT_DROPOUT: float = 0.1


# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

DEFAULT_BATCH_SIZE: int = 512  # Optimized for GPU (H100: 80GB VRAM handles 512+ easily)
DEFAULT_MAX_EPOCHS: int = 100
DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_EARLY_STOPPING_PATIENCE: int = 10
DEFAULT_N_SPLITS: int = 5


# =============================================================================
# OPTUNA OPTIMIZATION DEFAULTS
# =============================================================================

DEFAULT_LABEL_OPTIMIZATION_TRIALS: int = 100
DEFAULT_FEATURE_SELECTION_TRIALS: int = 100
DEFAULT_FEATURE_PRUNING_TRIALS: int = 50
DEFAULT_HYPERPARAM_TRIALS: int = 100
DEFAULT_OPTUNA_RANDOM_STATE: int = 42
DEFAULT_MIN_FEATURES: int = 20
DEFAULT_MAX_FEATURES_TO_SEARCH: int = 40  # Max features in 5D search space

# Trade rate thresholds for 5D objective's Sharpe-like metric.
# Below MIN_TRADE_RATE the model is too selective (returns 0.0).
# Between MIN and PREFERRED, the Sharpe is linearly scaled down.
DEFAULT_MIN_TRADE_RATE: float = 0.10
DEFAULT_PREFERRED_TRADE_RATE: float = 0.20


# =============================================================================
# OHLCV COLUMN NAMES
# =============================================================================

OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]
REQUIRED_COLUMNS: list[str] = ["datetime"] + OHLCV_COLUMNS


# =============================================================================
# LABEL CLASSES
# =============================================================================

LABEL_CLASSES: dict[int, str] = {
    -1: "short",
    0: "neutral",
    1: "long",
}
N_CLASSES: int = 3


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_models_for_family(family: str) -> list[str]:
    """Get all models belonging to a family."""
    return MODEL_FAMILIES.get(family, [])


def get_models_for_rank(rank: int) -> list[str]:
    """Get all models requiring a specific data rank."""
    return [model for model, r in get_model_data_ranks().items() if r == rank]


def get_adapter_for_model(model_name: str) -> str:
    """Get the adapter type for a model."""
    adapter = get_model_adapter_map().get(model_name.lower())
    if adapter is None:
        raise ValueError(f"Unknown model: {model_name}")
    return adapter


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Timeframes
    "CANONICAL_TIMEFRAMES",
    "BASE_TIMEFRAME",
    "DEFAULT_MTF_TIMEFRAMES",
    # Horizons
    "DEFAULT_HORIZONS",
    "DEFAULT_HORIZON",
    # Splits
    "DEFAULT_SPLIT_RATIOS",
    # Purge/Embargo
    "DEFAULT_PURGE_BARS",
    "DEFAULT_EMBARGO_BARS",
    "MINUTES_PER_DAY",
    # Financial constants
    "TRADING_DAYS_PER_YEAR",
    "DEFAULT_ANNUALIZATION_FACTOR",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    # Models
    "MODEL_FAMILIES",
    "ALL_MODELS",
    "MODEL_TO_FAMILY",
    "MODEL_DATA_RANKS",  # noqa: F822 - lazy-loaded via __getattr__
    "MODEL_ADAPTER_MAP",  # noqa: F822 - lazy-loaded via __getattr__
    # Features
    "FEATURE_FAMILY_COUNTS",
    "TOTAL_BASE_FEATURES",
    "MTF_FEATURES_PER_TF",
    "MTF_TOTAL_FEATURES",
    # Sequence defaults
    "DEFAULT_SEQUENCE_LENGTH",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_DROPOUT",
    # Training defaults
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    "DEFAULT_N_SPLITS",
    # Optuna defaults
    "DEFAULT_LABEL_OPTIMIZATION_TRIALS",
    "DEFAULT_FEATURE_SELECTION_TRIALS",
    "DEFAULT_FEATURE_PRUNING_TRIALS",
    "DEFAULT_HYPERPARAM_TRIALS",
    "DEFAULT_OPTUNA_RANDOM_STATE",
    "DEFAULT_MIN_FEATURES",
    "DEFAULT_MAX_FEATURES_TO_SEARCH",
    "DEFAULT_MIN_TRADE_RATE",
    "DEFAULT_PREFERRED_TRADE_RATE",
    # OHLCV
    "OHLCV_COLUMNS",
    "REQUIRED_COLUMNS",
    # Labels
    "LABEL_CLASSES",
    "N_CLASSES",
    # Helper functions
    "get_models_for_family",
    "get_models_for_rank",
    "get_adapter_for_model",
]
