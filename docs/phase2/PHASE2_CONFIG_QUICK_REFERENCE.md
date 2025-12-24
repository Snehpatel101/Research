# Phase 2 Configuration Quick Reference

Quick reference for using Phase 1 configuration in Phase 2 model training.

---

## Label Columns

### Import

```python
from src.config.labels import (
    get_required_label_columns,
    get_optional_label_columns,
    is_label_column,
)
```

### Usage

```python
# Get label columns for a horizon
required = get_required_label_columns(5)  # ['label_h5', 'sample_weight_h5']
optional = get_optional_label_columns(5)  # ['quality_h5', 'mae_h5', ...]

# Check if a column is a label
is_label_column('label_h5')        # True
is_label_column('close')           # False
is_label_column('sample_weight_h20')  # True
```

---

## Feature Sets

### Import

```python
from src.config import (
    FEATURE_SET_DEFINITIONS,
    resolve_feature_set_names,
)
```

### Available Sets

| Name | Description | MTF | Cross-Asset | Features |
|------|-------------|-----|-------------|----------|
| `core_min` | Minimal base features | No | No | ~60 |
| `core_full` | All base features | No | No | ~90 |
| `mtf_plus` | Base + MTF + cross-asset | Yes | Yes | ~130 |

### Model Compatibility

```python
fset = FEATURE_SET_DEFINITIONS['mtf_plus']

# Check if compatible with your model
assert 'sequential' in fset.supported_model_types  # OK for LSTM/GRU/Transformer
assert 'tree' in fset.supported_model_types        # OK for LightGBM/XGBoost
assert 'tabular' in fset.supported_model_types     # OK for MLP/LogReg

# Get recommended sequence length
seq_length = fset.default_sequence_length  # 120 for mtf_plus, 60 for others

# Get recommended scaler
scaler_type = fset.recommended_scaler  # 'robust'
```

---

## Feature Categories

### Import

```python
from src.stages.scaling.core import (
    FeatureCategory,
    FEATURE_PATTERNS,
    DEFAULT_SCALING_STRATEGY,
)
```

### Categories

```python
# Available categories
FeatureCategory.RETURNS      # Normalized returns, ratios, z-scores
FeatureCategory.OSCILLATOR   # RSI, Stochastic (0-100 bounded)
FeatureCategory.PRICE_LEVEL  # Raw prices, SMAs, EMAs
FeatureCategory.VOLATILITY   # ATR, volatility measures
FeatureCategory.VOLUME       # Volume-based features
FeatureCategory.TEMPORAL     # Sin/cos time encodings
FeatureCategory.BINARY       # 0/1 flags, regimes
```

### Categorize a Feature

```python
def categorize_feature(feature_name: str) -> FeatureCategory:
    """Categorize a feature by its name."""
    for category, patterns in FEATURE_PATTERNS.items():
        if any(pat in feature_name for pat in patterns):
            return category
    return FeatureCategory.UNKNOWN

# Example
categorize_feature('hl_ratio')       # FeatureCategory.RETURNS
categorize_feature('rsi_14')         # FeatureCategory.OSCILLATOR
categorize_feature('close_15m')      # FeatureCategory.PRICE_LEVEL
categorize_feature('atr_14')         # FeatureCategory.VOLATILITY
```

### Get Recommended Scaler

```python
category = categorize_feature('atr_14')  # VOLATILITY
scaler_type = DEFAULT_SCALING_STRATEGY[category]  # ScalerType.ROBUST

# All strategies
# RETURNS     -> NONE    (already normalized)
# OSCILLATOR  -> MINMAX  (preserve 0-100 range)
# PRICE_LEVEL -> ROBUST  (handle outliers)
# VOLATILITY  -> ROBUST  (handle outliers)
# VOLUME      -> ROBUST  (often skewed)
# TEMPORAL    -> NONE    (already sin/cos normalized)
# BINARY      -> NONE    (keep 0/1)
```

---

## Load Datasets

### Import

```python
import pandas as pd
from pathlib import Path
```

### Dataset Paths

```python
# Dataset manifest
manifest_path = Path('runs/<run_id>/artifacts/dataset_manifest.json')

# Load manifest
import json
with open(manifest_path) as f:
    manifest = json.load(f)

# Get paths for a specific feature set and horizon
paths = manifest['feature_sets']['core_full']['horizons']['5']
train_path = paths['train_path']
val_path = paths['val_path']
test_path = paths['test_path']

# Load data
train = pd.read_parquet(train_path)
val = pd.read_parquet(val_path)
test = pd.read_parquet(test_path)
```

### Dataset Structure

Each dataset contains:

1. **Metadata columns:** `datetime`, `symbol`
2. **Feature columns:** Selected based on feature set
3. **Label columns:** All labels for the specified horizon

```python
# Example columns for core_full/h5
['datetime', 'symbol',                      # Metadata
 'rsi_14', 'macd_hist', 'atr_14', ...,     # Features (~90)
 'label_h5', 'sample_weight_h5',           # Required labels
 'quality_h5', 'mae_h5', 'mfe_h5', ...]    # Optional labels
```

---

## Example: Train a Model

```python
from pathlib import Path
import pandas as pd
import json
from src.config import FEATURE_SET_DEFINITIONS, get_required_label_columns

# 1. Select feature set
feature_set_name = 'core_full'
horizon = 5
fset = FEATURE_SET_DEFINITIONS[feature_set_name]

# 2. Validate model compatibility
assert 'tree' in fset.supported_model_types

# 3. Load dataset manifest
manifest_path = Path('runs/<run_id>/artifacts/dataset_manifest.json')
with open(manifest_path) as f:
    manifest = json.load(f)

# 4. Get dataset paths
dataset_info = manifest['feature_sets'][feature_set_name]['horizons'][str(horizon)]
train_path = dataset_info['train_path']

# 5. Load data
df = pd.read_parquet(train_path)

# 6. Split into features and labels
metadata = ['datetime', 'symbol']
label_cols = get_required_label_columns(horizon)  # ['label_h5', 'sample_weight_h5']

feature_cols = [
    col for col in df.columns
    if col not in metadata and col not in label_cols
]

X = df[feature_cols]
y = df['label_h5']
sample_weight = df['sample_weight_h5']

# 7. Train model
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100,
    random_state=42,
)
model.fit(X, y, sample_weight=sample_weight)
```

---

## Example: Create Sequences for LSTM

```python
import numpy as np
from src.config import FEATURE_SET_DEFINITIONS

# 1. Select feature set for sequential model
fset = FEATURE_SET_DEFINITIONS['mtf_plus']
assert 'sequential' in fset.supported_model_types

# 2. Get recommended sequence length
seq_length = fset.default_sequence_length  # 120

# 3. Create sequences
def create_sequences(df, seq_length, label_col):
    """Create sequences for LSTM/GRU."""
    metadata = ['datetime', 'symbol']
    label_cols = [col for col in df.columns if col.startswith(('label_', 'sample_weight_', 'quality_'))]

    feature_cols = [
        col for col in df.columns
        if col not in metadata and col not in label_cols
    ]

    X_sequences = []
    y_sequences = []

    for i in range(seq_length, len(df)):
        X_sequences.append(df[feature_cols].iloc[i-seq_length:i].values)
        y_sequences.append(df[label_col].iloc[i])

    return np.array(X_sequences), np.array(y_sequences)

# Usage
X_seq, y_seq = create_sequences(train_df, seq_length, 'label_h5')
print(f"X shape: {X_seq.shape}")  # (n_samples, seq_length, n_features)
print(f"y shape: {y_seq.shape}")  # (n_samples,)
```

---

## Cheat Sheet

```python
# Labels
from src.config.labels import get_required_label_columns
labels = get_required_label_columns(5)  # ['label_h5', 'sample_weight_h5']

# Feature Sets
from src.config import FEATURE_SET_DEFINITIONS
fset = FEATURE_SET_DEFINITIONS['core_full']
seq_len = fset.default_sequence_length  # 60

# Categorization
from src.stages.scaling.core import categorize_feature
cat = categorize_feature('rsi_14')  # FeatureCategory.OSCILLATOR

# Load dataset
df = pd.read_parquet('data/splits/datasets/core_full/h5/train.parquet')
```

---

## Common Patterns

### Pattern 1: Load and Split

```python
from src.config.labels import get_required_label_columns

df = pd.read_parquet(train_path)
label_cols = get_required_label_columns(horizon)

X = df.drop(columns=['datetime', 'symbol'] + label_cols)
y = df['label_h5']
sample_weight = df['sample_weight_h5']
```

### Pattern 2: Feature Selection by Category

```python
from src.stages.scaling.core import FeatureCategory, FEATURE_PATTERNS

def select_features_by_category(df, categories):
    """Select features by category."""
    selected = []
    for col in df.columns:
        for category, patterns in FEATURE_PATTERNS.items():
            if category in categories and any(pat in col for pat in patterns):
                selected.append(col)
                break
    return selected

# Example: Only returns and oscillators for tree models
features = select_features_by_category(
    df,
    [FeatureCategory.RETURNS, FeatureCategory.OSCILLATOR]
)
```

### Pattern 3: Validate Feature Set

```python
def validate_feature_set(feature_set_name, model_type):
    """Validate that a feature set supports a model type."""
    fset = FEATURE_SET_DEFINITIONS[feature_set_name]
    if model_type not in fset.supported_model_types:
        raise ValueError(
            f"Feature set '{feature_set_name}' does not support '{model_type}'. "
            f"Supported: {fset.supported_model_types}"
        )
    return fset

# Usage
fset = validate_feature_set('mtf_plus', 'sequential')  # OK
fset = validate_feature_set('core_min', 'sequential')  # Also OK
```

---

## Troubleshooting

**Q: How do I know which features are in a feature set?**

```python
from src.utils.feature_sets import resolve_feature_set

df = pd.read_parquet(train_path)
fset_def = FEATURE_SET_DEFINITIONS['core_full']
features = resolve_feature_set(df, fset_def)
print(f"Features: {features}")
```

**Q: How do I create a custom feature set?**

```python
from src.config.feature_sets import FeatureSetDefinition

custom = FeatureSetDefinition(
    name="custom_momentum",
    description="Momentum indicators only",
    include_prefixes=['rsi_', 'macd_', 'roc_'],
    include_mtf=False,
    include_cross_asset=False,
    supported_model_types=['tree', 'tabular'],
    default_sequence_length=None,
    recommended_scaler='robust',
)
```

**Q: What scaler should I use for my features?**

```python
from src.stages.scaling.core import categorize_feature, DEFAULT_SCALING_STRATEGY

feature = 'atr_14'
category = categorize_feature(feature)
scaler = DEFAULT_SCALING_STRATEGY[category]
print(f"{feature} -> {category.name} -> {scaler.value}")
# Output: atr_14 -> VOLATILITY -> robust
```

---

## Reference Links

- **Full docs:** `/home/jake/Desktop/Research/docs/phase1/PHASE1_CONFIG_FIXES.md`
- **Labels module:** `/home/jake/Desktop/Research/src/config/labels.py`
- **Feature sets:** `/home/jake/Desktop/Research/src/config/feature_sets.py`
- **Scaling core:** `/home/jake/Desktop/Research/src/stages/scaling/core.py`
