# SNwH Implementation: Phase 6 - ML for Dummies Config

**Philosophy:** Dead simple for beginners, full power for experts.

Users make **HIGH-LEVEL choices**. The system figures out the details.

---

## The Simple API

```python
from src.config import train

# Absolute beginner - just works!
train("xgboost")

# Pick symbol and horizon
train("xgboost", symbol="MGC", horizon=20)

# Want optimization?
train("xgboost", optimize="hyperparameters")

# Multiple models
train(["xgboost", "lstm"])

# Build an ensemble
train(["xgboost", "lstm", "patchtst"], ensemble=True)

# Override any smart default
train("lstm", sequence_length=120)
```

---

## User Choices (The Only Questions)

Users only need to decide these things:

| Question | Options | Default |
|----------|---------|---------|
| **Which model(s)?** | Any of 23 models | Required |
| **Which symbol?** | "MES", "MGC", etc. | "MES" |
| **Which horizon(s)?** | 5, 10, 15, 20, etc. | [20] |
| **Optimize?** | "none", "features", "hyperparameters", "both" | "none" |
| **Build ensemble?** | True/False | False |

That's it. Everything else is automatic.

---

## Smart Defaults (System Figures Out)

The system automatically determines the right settings based on the model type:

### Per-Model Timeframe

| Model Type | Timeframe | Why |
|------------|-----------|-----|
| **Boosting** (XGBoost, LightGBM, CatBoost) | 15min | Works best with aggregated data |
| **Classical** (Random Forest, Logistic, SVM) | 15min | Same as boosting |
| **RNNs** (LSTM, GRU) | 5min | Need finer granularity for sequences |
| **CNNs** (TCN, InceptionTime, ResNet1D) | 5min | Multi-scale pattern detection |
| **Transformers** (PatchTST, iTransformer, TFT) | 1min | Learn from raw data |

### Per-Model Features

| Model Type | Features | Count | Why |
|------------|----------|-------|-----|
| **Boosting** | Full indicators + MTF | ~100 | Love engineered features |
| **Classical** | Standard indicators | ~70 | Simpler models, fewer features |
| **RNNs** | Indicators + wavelets | ~80 | Benefit from frequency decomposition |
| **CNNs** | Indicators + wavelets | ~80 | Multi-scale patterns |
| **Transformers** | Raw OHLCV only | 5 | Learn representations from scratch |

### Per-Model Sequence Length

| Model | Sequence Length | Why |
|-------|-----------------|-----|
| LSTM, GRU | 60 | Standard RNN window |
| TCN | 120 | Longer receptive field |
| Transformer | 96 | Patch-based attention |
| N-BEATS | 60 | Forecast horizon × 3 |
| PatchTST, iTransformer | 96 | Optimal for patches |
| TFT | 60 | Attention over history |

### Per-Model Batch Size

| Model Type | Batch Size | Why |
|------------|------------|-----|
| Boosting | N/A | Not batched |
| Classical | N/A | Not batched |
| RNNs | 256 | Standard GPU batch |
| CNNs | 128 | Slightly larger models |
| Transformers | 64 | Memory intensive |

---

## MODEL_DEFAULTS (Single Source of Truth)

All 23 models have their smart defaults in one dictionary:

```python
MODEL_DEFAULTS = {
    # === BOOSTING (3) ===
    "xgboost": {
        "family": "boosting",
        "timeframe": "15min",
        "features": "full",        # ~100 indicators + MTF
        "sequence_length": None,
        "batch_size": None,
        "description": "Gradient boosting. Fast and interpretable.",
    },
    "lightgbm": {...},
    "catboost": {...},

    # === CLASSICAL (3) ===
    "random_forest": {
        "family": "classical",
        "timeframe": "15min",
        "features": "standard",    # ~70 features
        "sequence_length": None,
        "batch_size": None,
    },
    "logistic": {...},
    "svm": {...},

    # === RNNs (2) ===
    "lstm": {
        "family": "neural",
        "timeframe": "5min",
        "features": "sequence",    # ~80 + wavelets
        "sequence_length": 60,
        "batch_size": 256,
    },
    "gru": {...},

    # === CNNs (4) ===
    "tcn": {
        "family": "neural",
        "timeframe": "5min",
        "features": "sequence",
        "sequence_length": 120,
        "batch_size": 128,
    },
    "inceptiontime": {...},
    "resnet1d": {...},
    "nbeats": {...},

    # === TRANSFORMERS (4) ===
    "patchtst": {
        "family": "transformer",
        "timeframe": "1min",
        "features": "raw",         # OHLCV only (5 features)
        "sequence_length": 96,
        "batch_size": 64,
    },
    "itransformer": {...},
    "tft": {...},
    "transformer": {...},

    # === META-LEARNERS (4) ===
    "ridge_meta": {...},
    "mlp_meta": {...},
    "calibrated_meta": {...},
    "xgboost_meta": {...},

    # === ENSEMBLES (3) ===
    "voting": {...},
    "stacking": {...},
    "blending": {...},
}
```

---

## Feature Sets

```python
FEATURE_SETS = {
    "full": {
        "description": "All engineered indicators + MTF features",
        "count": "~100",
        "includes": ["momentum", "volatility", "volume", "trend", "mtf"],
    },
    "standard": {
        "description": "Core technical indicators",
        "count": "~70",
        "includes": ["momentum", "volatility", "volume", "trend"],
    },
    "sequence": {
        "description": "Indicators + wavelets for sequence models",
        "count": "~80",
        "includes": ["momentum", "volatility", "wavelets"],
    },
    "minimal": {
        "description": "Only essential indicators",
        "count": "~30",
        "includes": ["momentum", "volatility"],
    },
    "raw": {
        "description": "Raw OHLCV only (for transformers)",
        "count": "5",
        "includes": ["open", "high", "low", "close", "volume"],
    },
}
```

---

## Optimization Options

```python
OPTIMIZATION_DEFAULTS = {
    "none": {
        "tune_features": False,
        "tune_hyperparameters": False,
        "n_trials": 0,
    },
    "features": {
        "tune_features": True,
        "tune_hyperparameters": False,
        "n_trials": 30,           # Optuna trials for feature selection
    },
    "hyperparameters": {
        "tune_features": False,
        "tune_hyperparameters": True,
        "n_trials": 50,           # Optuna trials for HP tuning
    },
    "both": {
        "tune_features": True,
        "tune_hyperparameters": True,
        "n_trials": 100,          # More trials for joint optimization
    },
}
```

---

## SmartConfig (User-Facing)

Only the fields users actually care about:

```python
@dataclass
class SmartConfig:
    """User-facing configuration. Only high-level choices."""

    # Required
    models: list[str]                    # Which model(s)?

    # Common choices
    symbol: str = "MES"                  # Which symbol?
    horizons: list[int] = [20]           # How far ahead?
    optimize: str = "none"               # Tune things?
    ensemble: bool = False               # Combine models?

    # Data paths (usually defaults are fine)
    data_dir: Path = Path("data")
    output_dir: Path = Path("experiments/runs")

    # Rarely changed
    seed: int = 42
```

---

## ResolvedModelConfig (System-Generated)

The full config after merging user choices + smart defaults:

```python
@dataclass
class ResolvedModelConfig:
    """Full configuration for a model. Generated by the system."""

    # From user
    model_name: str
    symbol: str
    horizons: list[int]

    # From MODEL_DEFAULTS (or user override)
    family: str
    timeframe: str
    features: str
    sequence_length: int | None
    batch_size: int | None

    # From OPTIMIZATION_DEFAULTS
    tune_features: bool
    tune_hyperparameters: bool
    n_trials: int

    # Computed
    feature_count: int              # Based on feature set
    is_sequence_model: bool         # sequence_length is not None
```

---

## Helper Functions

```python
# List available models
list_models()                    # All 23
list_models(family="boosting")   # Just boosting models

# Get model description
describe_model("xgboost")
# "Gradient boosting. Fast and interpretable. Great baseline."

# See a model's smart defaults
show_defaults("lstm")
# {"family": "neural", "timeframe": "5min", "features": "sequence", ...}

# Preview what will happen (without training)
preview_config("xgboost", symbol="MGC", optimize="hyperparameters")
# Shows resolved config, estimated time, memory requirements

# Compare models side-by-side
quick_compare(["xgboost", "lstm", "patchtst"])
# Pretty table comparing defaults
```

---

## Override Anything

Advanced users can override any smart default:

```python
# Global override
train("lstm", sequence_length=120)

# Per-model overrides in multi-model training
train(
    ["xgboost", "lstm", "patchtst"],
    ensemble=True,
    xgboost={"timeframe": "5min"},      # Override XGBoost's timeframe
    lstm={"sequence_length": 120},       # Override LSTM's sequence length
    patchtst={"batch_size": 32},         # Override PatchTST's batch size
)
```

---

## Examples

### Beginner

```python
from src.config import train

# Just train XGBoost with all defaults
results = train("xgboost")
print(results.metrics)
```

### Intermediate

```python
# Train on different symbol, optimize hyperparameters
results = train(
    "xgboost",
    symbol="MGC",
    horizon=20,
    optimize="hyperparameters",
)
```

### Advanced

```python
# Heterogeneous ensemble with custom settings
results = train(
    ["xgboost", "lstm", "patchtst"],
    symbol="MES",
    horizons=[10, 20],
    ensemble=True,
    optimize="both",
    lstm={"sequence_length": 120, "batch_size": 128},
)
```

### Expert (Full Control)

```python
from src.config import SmartConfig, resolve_config

# Build config programmatically
config = SmartConfig(
    models=["xgboost", "lstm", "patchtst"],
    symbol="MES",
    horizons=[10, 20],
    optimize="both",
    ensemble=True,
)

# Resolve to full configs
resolved = resolve_config(config)

# Inspect before training
for model_name, model_config in resolved.items():
    print(f"{model_name}: {model_config.timeframe}, {model_config.features}")

# Train with resolved config
results = train(config)
```

---

## What Gets Deleted

All the complexity from the old system:

| Item | Status |
|------|--------|
| `config/` directory (40+ YAML files) | DELETE |
| `src/config/unified.py` (983 lines) | REPLACE with smart_config.py |
| `src/config/global_config.py` | DELETE |
| `src/config/validators.py` | DELETE (validation in SmartConfig) |
| `src/ml_pipeline/config.py` | DELETE |
| `src/training/config.py` | DELETE |
| 6 overlapping config classes | REPLACE with SmartConfig |

---

## Implementation Status

### Phase 6.1: Smart Config (✅ Complete)

**File created:** `src/config/smart_config.py` (~900 lines)

Contains:
- `MODEL_DEFAULTS` - All 23 models with smart defaults
- `FEATURE_SETS` - Feature set definitions
- `OPTIMIZATION_DEFAULTS` - Optimization options
- `SmartConfig` - User-facing config
- `ResolvedModelConfig` - System-resolved config
- `train()` - Main entry point
- Helper functions: `list_models()`, `describe_model()`, `show_defaults()`, `preview_config()`, `quick_compare()`

### Phase 6.2: Training Pipeline Integration (✅ Complete)

**Files modified:**

1. `src/training/config.py`:
   - Added `features` field to `ModelConfig` (feature set name: "full", "standard", "sequence", "raw", etc.)
   - Added `batch_size` field to `ModelConfig`

2. `src/training/feature_selector.py`:
   - Added smart config feature sets: "standard", "sequence", "raw", "minimal_raw", "tft"
   - Added exact-match group patterns for "raw_ohlcv" and "raw_close_volume"

3. `src/training/orchestrator.py`:
   - Added `_filter_container_by_feature_mode()` method
   - Per-model feature filtering applied before training when `features` is specified
   - Full integration with `ExperimentConfig` and `ModelConfig`

**Integration flow:**
```
train("xgboost")
    ↓
SmartConfig created (smart_config.py)
    ↓
resolve_config() merges user + defaults
    ↓
ModelConfig objects created with:
  - name, timeframe, features, batch_size, sequence_length
  - optimize_features, optimize_hyperparams
    ↓
ExperimentConfig passed to TrainingOrchestrator
    ↓
For each model:
  1. Load data for model's timeframe
  2. Filter features by model's feature mode
  3. Optionally optimize features (Optuna)
  4. Train model
    ↓
Build ensemble if enabled
```

### Phase 6.3: Legacy Config Cleanup (Pending)

Files to delete after validation:
- `config/` directory (40+ YAML files)
- Redundant config classes

---

## Success Criteria

- [x] `train("xgboost")` works with no other setup
- [x] All 23 models have smart defaults
- [x] Per-model timeframe automatic
- [x] Per-model features automatic
- [x] Per-model sequence length automatic
- [x] User can override any default
- [x] Optimization is a simple choice ("none", "features", "hyperparameters", "both")
- [x] Integration with actual training pipeline (Phase 6.2)
- [ ] Delete old config files (Phase 6.3)
