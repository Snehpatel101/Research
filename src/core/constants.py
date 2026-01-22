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
DEFAULT_MTF_TIMEFRAMES: list[str] = ["5min", "15min", "60min"]


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
        "transformer",
        "patchtst",
        "itransformer",
        "tft",
        "nbeats",
        "inceptiontime",
        "resnet1d",
    ],
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

MODEL_DATA_RANKS: dict[str, int] = {
    # Boosting - 2D (tabular)
    "xgboost": 2,
    "lightgbm": 2,
    "catboost": 2,
    # Classical - 2D (tabular)
    "random_forest": 2,
    "logistic": 2,
    "svm": 2,
    # Neural RNN/CNN - 3D (sequence)
    "lstm": 3,
    "gru": 3,
    "tcn": 3,
    "transformer": 3,
    "nbeats": 3,
    "inceptiontime": 3,
    "resnet1d": 3,
    "tft": 3,  # TFT is 3D sequence (see PHASE_0 notes)
    # Advanced Neural - 4D (multi-stream/multi-timeframe)
    "patchtst": 4,
    "itransformer": 4,
    # Ensemble/Meta - 2D (OOF probabilities)
    "voting": 2,
    "stacking": 2,
    "blending": 2,
    "ridge_meta": 2,
    "mlp_meta": 2,
    "xgboost_meta": 2,
    "calibrated_meta": 2,
}

# NOTE: TFT (Temporal Fusion Transformer) is classified as 3D sequence model.
# While TFT can handle multi-timeframe data, the standard implementation
# uses single-timeframe sequence input with engineered features.
# PatchTST and iTransformer are the designated 4D multi-stream models.


# =============================================================================
# MODEL -> ADAPTER MAPPING
# =============================================================================

MODEL_ADAPTER_MAP: dict[str, str] = {
    # Boosting -> Tabular (2D)
    "xgboost": "tabular",
    "lightgbm": "tabular",
    "catboost": "tabular",
    # Classical -> Tabular (2D)
    "random_forest": "tabular",
    "logistic": "tabular",
    "svm": "tabular",
    # Neural -> Sequence (3D)
    "lstm": "sequence",
    "gru": "sequence",
    "tcn": "sequence",
    "transformer": "sequence",
    "nbeats": "sequence",
    "inceptiontime": "sequence",
    "resnet1d": "sequence",
    "tft": "sequence",  # TFT uses sequence adapter (3D)
    # Advanced Neural -> Multi-Stream (4D)
    "patchtst": "multi_stream",
    "itransformer": "multi_stream",
    # Ensemble/Meta -> Tabular (2D on OOF)
    "voting": "tabular",
    "stacking": "tabular",
    "blending": "tabular",
    "ridge_meta": "tabular",
    "mlp_meta": "tabular",
    "xgboost_meta": "tabular",
    "calibrated_meta": "tabular",
}


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

DEFAULT_BATCH_SIZE: int = 256
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
    return [model for model, r in MODEL_DATA_RANKS.items() if r == rank]


def get_adapter_for_model(model_name: str) -> str:
    """Get the adapter type for a model."""
    adapter = MODEL_ADAPTER_MAP.get(model_name.lower())
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
    # Models
    "MODEL_FAMILIES",
    "ALL_MODELS",
    "MODEL_TO_FAMILY",
    "MODEL_DATA_RANKS",
    "MODEL_ADAPTER_MAP",
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
