# Phase 1 Dataset Quickstart Guide

Quick reference for using Phase 1 dataset components in model training.

---

## 1. Loading Data

### Basic Loading
```python
from src.stages.datasets import TimeSeriesDataContainer

# Load scaled data for a specific horizon
container = TimeSeriesDataContainer.from_parquet_dir(
    parquet_dir='data/splits/scaled',
    horizon=20
)
```

### What You Get
- **Splits:** train, val, test (70/15/15)
- **Features:** 129 (base + MTF + cross-asset)
- **Labels:** {-1, 0, 1} (short, neutral, long)
- **Weights:** [0.5, 1.5] (quality-based)

---

## 2. Model Interfaces

### Sklearn Models (RandomForest, XGBoost, LightGBM)
```python
X_train, y_train, w_train = container.get_sklearn_arrays('train')
X_val, y_val, w_val = container.get_sklearn_arrays('val')
X_test, y_test, w_test = container.get_sklearn_arrays('test')

# Example: XGBoost
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dval, 'val')],
)
```

### PyTorch Models (LSTM, Transformer, CNN)
```python
from torch.utils.data import DataLoader

# Create sequence dataset
dataset_train = container.get_pytorch_sequences('train', seq_len=60)
dataset_val = container.get_pytorch_sequences('val', seq_len=60)

# Create DataLoader
loader_train = DataLoader(
    dataset_train,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for epoch in range(num_epochs):
    for X_seq, y, w in loader_train:
        # X_seq: (batch_size, seq_len, n_features)
        # y: (batch_size,)
        # w: (batch_size,)

        outputs = model(X_seq)
        loss = criterion(outputs, y, weight=w)
        loss.backward()
        optimizer.step()
```

### NeuralForecast Models
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS

# Get NeuralForecast-compatible DataFrame
nf_df = container.get_neuralforecast_df('train')

# Train model
models = [
    NBEATS(
        h=20,
        input_size=60,
        stack_types=['trend', 'seasonality'],
    ),
]

nf = NeuralForecast(models=models, freq='5min')
nf.fit(df=nf_df)
```

---

## 3. Feature Set Selection

### Available Feature Sets
```python
from src.config.feature_sets import FEATURE_SET_DEFINITIONS
from src.utils.feature_sets import resolve_feature_set

# View available feature sets
for name, definition in FEATURE_SET_DEFINITIONS.items():
    print(f"{name}: {definition.description}")
```

**Output:**
- `core_min`: 72 features (minimal, no MTF, no cross-asset)
- `core_full`: 97 features (all base-timeframe)
- `mtf_plus`: 129 features (base + MTF + cross-asset)

### Apply Feature Set
```python
# Get filtered feature columns
train_df = container.splits['train'].df
feature_columns = resolve_feature_set(train_df, FEATURE_SET_DEFINITIONS['core_min'])

# Create filtered arrays
X_train = train_df[feature_columns].values
y_train = train_df['label_h20'].values
w_train = train_df['sample_weight_h20'].values
```

---

## 4. Data Validation

### Validate Model-Ready Status
```python
from src.stages.datasets import validate_model_ready

result = validate_model_ready(container)

if result.is_valid:
    print("✓ Data is model-ready")
else:
    print("✗ Data has issues:")
    for error in result.errors:
        print(f"  - {error}")

# View warnings
for warning in result.warnings:
    print(f"  ⚠ {warning}")
```

### Manual Quality Checks
```python
import numpy as np

X_train, y_train, w_train = container.get_sklearn_arrays('train')

# Check for NaN/Inf
assert not np.any(np.isnan(X_train)), "NaN values found"
assert not np.any(np.isinf(X_train)), "Inf values found"

# Check label range
assert set(np.unique(y_train)).issubset({-1, 0, 1}), "Invalid labels"

# Check weight range
assert (w_train.min() >= 0.5) and (w_train.max() <= 1.5), "Invalid weights"

print("✓ All quality checks passed")
```

---

## 5. Common Patterns

### Train Multiple Horizons
```python
from src.config import HORIZONS

for horizon in HORIZONS:
    print(f"\nTraining for H{horizon}...")

    container = TimeSeriesDataContainer.from_parquet_dir(
        'data/splits/scaled',
        horizon=horizon
    )

    X_train, y_train, w_train = container.get_sklearn_arrays('train')

    # Train model for this horizon
    model = train_model(X_train, y_train, w_train)

    # Save model
    model.save(f'models/model_h{horizon}.pkl')
```

### Cross-Validation with Time Series
```python
# Use walk-forward validation
train_df = container.splits['train'].df

# Example: 5-fold walk-forward
n_samples = len(train_df)
fold_size = n_samples // 5

for fold in range(5):
    # Train on first (fold+1) folds
    train_end = (fold + 1) * fold_size

    X_train = train_df.iloc[:train_end][feature_columns].values
    y_train = train_df.iloc[:train_end]['label_h20'].values

    # Validate on next fold
    if fold < 4:
        val_start = train_end
        val_end = (fold + 2) * fold_size

        X_val = train_df.iloc[val_start:val_end][feature_columns].values
        y_val = train_df.iloc[val_start:val_end]['label_h20'].values

        # Train and validate
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
```

### Filter Constant Features
```python
from sklearn.feature_selection import VarianceThreshold

# Remove constant features
selector = VarianceThreshold(threshold=0.0)
X_train_filtered = selector.fit_transform(X_train)
X_val_filtered = selector.transform(X_val)

print(f"Features before: {X_train.shape[1]}")
print(f"Features after: {X_train_filtered.shape[1]}")
```

---

## 6. Label Information

### Required Label Columns
```python
from src.config import get_required_label_columns

# Get required labels for a horizon
labels = get_required_label_columns(20)
# ['label_h20', 'sample_weight_h20']
```

### Optional Label Columns
```python
from src.config import get_optional_label_columns

# Get optional labels (MAE, MFE, quality, etc.)
opt_labels = get_optional_label_columns(20)
# ['quality_h20', 'bars_to_hit_h20', 'mae_h20', 'mfe_h20', ...]
```

### Label Metadata
```python
from src.config import get_label_metadata

# Get metadata for a label column
metadata = get_label_metadata('label_h{h}', horizon=20)
# {
#   'description': 'Primary triple-barrier label',
#   'dtype': 'int8',
#   'values': [-1, 0, 1],
#   'meanings': {-1: 'short', 0: 'neutral', 1: 'long'},
#   'column_name': 'label_h20',
#   'horizon': 20
# }
```

---

## 7. Troubleshooting

### Issue: "No data found for horizon X"
**Solution:** Ensure scaled data exists for that horizon
```bash
ls data/splits/scaled/*_scaled.parquet
```

### Issue: "Module not found"
**Solution:** Ensure you're running from project root
```bash
cd /path/to/Research
python your_script.py
```

### Issue: "Constant features warning"
**Solution:** Filter them during training
```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.0)
X_filtered = selector.fit_transform(X)
```

### Issue: "Out of memory"
**Solution:** Use smaller batch sizes or sequence lengths
```python
# Reduce batch size
loader = DataLoader(dataset, batch_size=16, ...)

# Reduce sequence length
dataset = container.get_pytorch_sequences('train', seq_len=30)
```

---

## 8. Best Practices

### Always Validate Before Training
```python
result = validate_model_ready(container)
if not result.is_valid:
    raise ValueError(f"Data not ready: {result.errors}")
```

### Use Appropriate Feature Set
- `core_min`: Fast prototyping, interpretable models
- `core_full`: Standard training, balanced performance
- `mtf_plus`: Maximum information, ensemble models

### Respect Temporal Order
- Never shuffle time series data during validation/testing
- Use walk-forward or time series cross-validation
- Don't look ahead (data before validation shouldn't use data after)

### Use Sample Weights
```python
# Always include sample_weight in model training
model.fit(X_train, y_train, sample_weight=w_train)
```

### Monitor Class Balance
```python
import numpy as np

unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    pct = count / len(y_train) * 100
    print(f"Label {label}: {count} ({pct:.1f}%)")
```

---

## 9. Complete Example

### End-to-End Training Script
```python
#!/usr/bin/env python3
"""
Complete example: Load data → Train model → Evaluate
"""
import numpy as np
import xgboost as xgb
from src.stages.datasets import TimeSeriesDataContainer, validate_model_ready

# 1. Load data
print("Loading data...")
container = TimeSeriesDataContainer.from_parquet_dir(
    'data/splits/scaled',
    horizon=20
)

# 2. Validate
print("Validating...")
result = validate_model_ready(container)
if not result.is_valid:
    raise ValueError(f"Data not ready: {result.errors}")

# 3. Extract arrays
print("Extracting arrays...")
X_train, y_train, w_train = container.get_sklearn_arrays('train')
X_val, y_val, w_val = container.get_sklearn_arrays('val')
X_test, y_test, w_test = container.get_sklearn_arrays('test')

# 4. Train model
print("Training model...")
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    verbose_eval=10,
)

# 5. Evaluate
print("\nEvaluating...")
y_pred_train = model.predict(dtrain)
y_pred_val = model.predict(dval)
y_pred_test = model.predict(dtest)

from sklearn.metrics import accuracy_score, classification_report

print(f"Train accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Val accuracy:   {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Test accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")

print("\nTest set classification report:")
print(classification_report(y_test, y_pred_test, target_names=['Short', 'Neutral', 'Long']))

# 6. Save model
print("\nSaving model...")
model.save_model('models/xgboost_h20.json')
print("✓ Done!")
```

---

**For more details, see:**
- `/home/jake/Desktop/Research/PHASE1_VALIDATION_REPORT.md` - Full validation report
- `/home/jake/Desktop/Research/docs/phase1/` - Phase 1 documentation
