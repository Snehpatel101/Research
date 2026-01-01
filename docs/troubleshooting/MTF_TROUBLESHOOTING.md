# MTF Troubleshooting Guide

**Audience:** Developers and users encountering MTF-related issues
**Last Updated:** 2026-01-01

---

## Table of Contents

1. [Common Error Messages](#common-error-messages)
2. [Feature Count Mismatches](#feature-count-mismatches)
3. [Shape Validation Failures](#shape-validation-failures)
4. [Data Preparation Errors](#data-preparation-errors)
5. [Model-Strategy Incompatibility](#model-strategy-incompatibility)
6. [Configuration Errors](#configuration-errors)
7. [Performance Issues](#performance-issues)

---

## Common Error Messages

### Error: "Feature count mismatch"

**Symptom:**
```
ValueError: Feature count mismatch: expected 180, got 143
```

**Cause:** Pipeline generated different feature count than expected.

**Solutions:**
1. Check `mtf_mode` configuration:
   - `'both'` = ~180 features (MTF bars + MTF indicators)
   - `'indicators'` = ~143 features (MTF indicators only)
   - `'bars'` = ~123 features (MTF bars only)

2. Verify timeframe configuration:
   ```python
   # Check current MTF timeframes
   from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES
   print(MTF_TIMEFRAMES)  # Should list 5 or 9 timeframes
   ```

3. If feature count is unexpectedly low:
   - Ensure raw data covers sufficient history for all MTF timeframes
   - Check for NaN values causing feature drops

---

### Error: "Shape mismatch in model input"

**Symptom:**
```
RuntimeError: shape '[64, 60, 180]' is invalid for input of size 768000
```

**Cause:** Model expects different input shape than data provides.

**Solutions:**
1. For **tabular models** (XGBoost, LightGBM, RF):
   ```python
   # Should receive 2D arrays
   assert X_train.ndim == 2, f"Expected 2D, got {X_train.ndim}D"
   ```

2. For **sequence models** (LSTM, GRU, TCN):
   ```python
   # Should receive 3D sequences
   assert X_train.ndim == 3, f"Expected 3D, got {X_train.ndim}D"
   X_train.shape  # (n_samples, seq_len, n_features)
   ```

3. For **advanced models** (PatchTST, TFT) when Strategy 3 implemented:
   ```python
   # May receive 4D multi-resolution tensors
   assert X_train.ndim in [3, 4]
   ```

---

### Error: "Invalid MTF timeframe"

**Symptom:**
```
ValueError: Timeframe '10min' not in MTF_TIMEFRAMES
```

**Cause:** Requested timeframe not currently implemented.

**Current Timeframes (5 implemented):**
- 15min, 30min, 1h, 4h, daily

**Missing Timeframes (4 not yet implemented):**
- 1min, 10min, 20min, 25min

**Workaround:**
```python
# Use only supported timeframes
MTF_SUPPORTED = ['15min', '30min', '1h', '4h', 'daily']
```

---

## Feature Count Mismatches

### Expected Feature Counts by Configuration

| MTF Mode | MTF Timeframes | Base Features | MTF Features | Total |
|----------|----------------|---------------|--------------|-------|
| `'both'` | 5 TFs (current) | ~150 | ~30 | ~180 |
| `'both'` | 9 TFs (intended) | ~150 | ~60 | ~210 |
| `'indicators'` | 5 TFs | ~150 | ~30 indicators | ~180 |
| `'bars'` | 5 TFs | ~150 | ~25 bars | ~175 |
| Single-TF | None | ~40-50 | 0 | ~40-50 |

### Debugging Feature Count Issues

```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer

# Load container
container = TimeSeriesDataContainer.from_parquet_dir("data/splits/scaled/")

# Check feature columns
print(f"Feature count: {len(container.feature_columns)}")
print(f"Feature names: {container.feature_columns[:10]}...")

# Check for MTF features
mtf_features = [f for f in container.feature_columns if '_15m' in f or '_30m' in f or '_1h' in f]
print(f"MTF features: {len(mtf_features)}")

# Check for specific feature groups
indicator_features = [f for f in container.feature_columns if 'rsi' in f or 'macd' in f or 'atr' in f]
bar_features = [f for f in container.feature_columns if f.startswith(('open_', 'high_', 'low_', 'close_', 'volume_'))]
```

### Feature Count Validation Script

```python
def validate_feature_count(container, expected_range=(150, 220)):
    """Validate feature count is within expected range."""
    actual = len(container.feature_columns)
    if not (expected_range[0] <= actual <= expected_range[1]):
        raise ValueError(
            f"Feature count {actual} outside expected range {expected_range}. "
            f"Check MTF configuration."
        )
    return True
```

---

## Shape Validation Failures

### 2D vs 3D Shape Issues

**Problem:** Model receives wrong dimensionality

**Diagnosis:**
```python
# Check data shape before training
print(f"X_train shape: {X_train.shape}")
print(f"X_train ndim: {X_train.ndim}")

# For tabular models
if model_family in ['boosting', 'classical']:
    assert X_train.ndim == 2, f"Tabular model needs 2D, got {X_train.ndim}D"

# For sequence models
if model_family in ['neural', 'cnn', 'advanced']:
    assert X_train.ndim >= 3, f"Sequence model needs 3D+, got {X_train.ndim}D"
```

**Solution for tabular models getting 3D data:**
```python
# If accidentally got sequences, take last timestep
if X_train.ndim == 3:
    X_train = X_train[:, -1, :]  # (n_samples, n_features)
```

**Solution for sequence models getting 2D data:**
```python
# If need sequences from tabular data
from src.phase1.stages.datasets.sequencer import TimeSeriesSequencer
sequencer = TimeSeriesSequencer(seq_len=60)
X_sequences = sequencer.fit_transform(X_train)  # (n_samples, 60, n_features)
```

### Sequence Length Mismatches

**Problem:**
```
ValueError: Sequence length 60 incompatible with requested 30
```

**Cause:** `seq_len` parameter doesn't match data.

**Solution:**
```python
# Check container's default sequence length
container = TimeSeriesDataContainer.from_parquet_dir(...)
print(f"Container default seq_len: {container.default_seq_len}")

# Request matching sequence length
dataset = container.get_pytorch_sequences("train", seq_len=60)  # Match container
```

---

## Data Preparation Errors

### Error: "Insufficient data for MTF"

**Symptom:**
```
ValueError: Insufficient data for MTF timeframe '4h': need 240 bars, have 100
```

**Cause:** Raw data doesn't cover enough history for higher timeframes.

**Solution:**
```python
# Check data coverage
df = pd.read_parquet("data/raw/MES_1m.parquet")
print(f"Data range: {df.index.min()} to {df.index.max()}")
print(f"Total bars: {len(df)}")

# Minimum requirements per MTF timeframe
# 4h = 240 1min bars minimum (need more for lookback)
# daily = 1440 1min bars minimum (one trading day)
# Recommended: 30+ days of data for proper MTF features
```

### Error: "NaN values in MTF features"

**Symptom:**
```
Warning: 45 NaN values found in MTF features
```

**Cause:** MTF resampling creates NaN at boundaries.

**Solution:**
```python
# Forward-fill MTF features (default behavior)
df[mtf_columns] = df[mtf_columns].ffill()

# Or use the anti-lookahead shift + ffill
df[mtf_columns] = df[mtf_columns].shift(1).ffill()

# Check for remaining NaNs after ffill
nan_counts = df[mtf_columns].isna().sum()
print(f"NaN counts after ffill: {nan_counts.sum()}")
```

### Error: "Label end times don't align"

**Symptom:**
```
ValueError: Label end times (1000) don't match samples (950)
```

**Cause:** MTF feature generation dropped rows that labels didn't.

**Solution:**
```python
# Ensure consistent indexing
common_index = df.index.intersection(labels_df.index)
df = df.loc[common_index]
labels_df = labels_df.loc[common_index]

# Verify alignment
assert len(df) == len(labels_df), "Data and labels must have same length"
```

---

## Model-Strategy Incompatibility

### Error: "Strategy 3 not supported for model"

**Symptom:**
```
NotImplementedError: Strategy 'mtf_ingestion' not yet implemented
```

**Status:** Strategy 3 (MTF Ingestion) is NOT YET IMPLEMENTED.

**Workaround:**
```python
# For sequence models, use Strategy 2 temporarily
config.mtf_strategy = 'mtf_indicators'  # Suboptimal but works

# Note: Sequence models will receive indicator features, not raw bars
# Performance may be lower than intended
```

### Invalid Model-Strategy Combinations

| Model Family | Strategy 1 | Strategy 2 | Strategy 3 |
|--------------|------------|------------|------------|
| Boosting | OK | RECOMMENDED | INVALID |
| Classical | OK | RECOMMENDED | INVALID |
| Neural | OK | Suboptimal* | RECOMMENDED (not implemented) |
| CNN | OK | INVALID | RECOMMENDED (not implemented) |
| Advanced | OK | Suboptimal* | RECOMMENDED (not implemented) |

*Suboptimal: Works but performance is limited

**Validation:**
```python
def validate_model_strategy(model_family, mtf_strategy):
    """Validate model-strategy compatibility."""
    if mtf_strategy == 'mtf_ingestion':
        raise NotImplementedError("Strategy 3 not yet implemented")

    if mtf_strategy == 'mtf_indicators' and model_family == 'cnn':
        raise ValueError("CNN models cannot use MTF indicators strategy")

    if mtf_strategy == 'mtf_indicators' and model_family in ['neural', 'advanced']:
        import warnings
        warnings.warn(
            f"{model_family} models should use mtf_ingestion (Strategy 3) "
            f"for optimal performance, but it's not yet implemented. "
            f"Using mtf_indicators as fallback."
        )
```

---

## Configuration Errors

### Error: "Unknown mtf_strategy"

**Symptom:**
```
ValueError: mtf_strategy 'multi_resolution' not in ['single_tf', 'mtf_indicators', 'mtf_ingestion']
```

**Valid Strategies:**
- `'single_tf'` - Single timeframe, no MTF (NOT YET IMPLEMENTED)
- `'mtf_indicators'` - MTF indicator features (PARTIALLY IMPLEMENTED - 5/9 TFs)
- `'mtf_ingestion'` - Raw MTF bars (NOT YET IMPLEMENTED)

**Current Default (as of 2026-01-01):**
```python
# Current behavior is effectively Strategy 2 (partial)
# All models receive ~180 indicator features from 5 timeframes
mtf_strategy = 'mtf_indicators'  # Default
mtf_timeframes = ['15min', '30min', '1h', '4h', 'daily']  # 5 of 9
```

### Error: "Invalid training_timeframe"

**Symptom:**
```
ValueError: training_timeframe '2min' not supported
```

**Supported Training Timeframes:**
- Current: 5min (hardcoded default)
- Intended: 1min, 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h

**Workaround (until configurable):**
```python
# Currently training timeframe is fixed at 5min
# To change, modify src/phase1/stages/clean/pipeline.py directly
# Not recommended until proper config support added
```

### Config File Validation

```python
def validate_pipeline_config(config):
    """Validate pipeline configuration."""
    errors = []

    # Check MTF strategy
    valid_strategies = ['single_tf', 'mtf_indicators', 'mtf_ingestion']
    if hasattr(config, 'mtf_strategy') and config.mtf_strategy not in valid_strategies:
        errors.append(f"Invalid mtf_strategy: {config.mtf_strategy}")

    # Check timeframes
    valid_tfs = ['1min', '5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h', '4h', 'daily']
    if hasattr(config, 'mtf_timeframes'):
        for tf in config.mtf_timeframes:
            if tf not in valid_tfs:
                errors.append(f"Invalid MTF timeframe: {tf}")

    # Check implemented status
    if hasattr(config, 'mtf_strategy'):
        if config.mtf_strategy == 'single_tf':
            errors.append("Strategy 1 (single_tf) not yet implemented")
        if config.mtf_strategy == 'mtf_ingestion':
            errors.append("Strategy 3 (mtf_ingestion) not yet implemented")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
```

---

## Performance Issues

### Slow MTF Feature Generation

**Symptom:** MTF stage takes >10 minutes

**Causes:**
1. Large dataset (>100k rows)
2. Too many MTF timeframes
3. Inefficient resampling

**Solutions:**
```python
# 1. Use efficient resampling
df_resampled = df.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# 2. Limit MTF timeframes for development
MTF_TIMEFRAMES_DEV = ['15min', '1h']  # Fewer TFs for faster iteration

# 3. Use chunked processing for large datasets
chunk_size = 50000
for chunk in pd.read_parquet("data/raw/MES_1m.parquet", chunksize=chunk_size):
    process_mtf_features(chunk)
```

### Memory Issues with Sequence Data

**Symptom:**
```
MemoryError: Unable to allocate 4.5 GiB for array
```

**Cause:** Creating sequences expands memory ~60x (for seq_len=60)

**Solutions:**
```python
# 1. Use PyTorch DataLoader with lazy loading
from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,  # Parallel loading
    pin_memory=True
)

# 2. Reduce sequence length
seq_len = 30  # Instead of 60

# 3. Use float32 instead of float64
X_train = X_train.astype(np.float32)

# 4. Sample subset for development
X_train_sample = X_train[:10000]  # First 10k samples
```

### Model Training Too Slow

**For Tabular Models:**
```python
# Use GPU acceleration for XGBoost
model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0
)
```

**For Neural Models:**
```python
# Reduce model complexity during development
model = LSTMModel(
    hidden_size=64,   # Smaller hidden size
    num_layers=1,     # Single layer
    dropout=0.2
)

# Use early stopping
early_stop = EarlyStopping(patience=5)
```

---

## Quick Diagnostic Commands

```bash
# Check current MTF configuration
python -c "from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES; print(f'MTF Timeframes: {list(MTF_TIMEFRAMES.keys())}')"

# Check feature count in pipeline output
python -c "import pandas as pd; df = pd.read_parquet('data/splits/scaled/train.parquet'); print(f'Features: {len([c for c in df.columns if c not in [\"label\", \"weight\", \"timestamp\"]])}')"

# Verify model registry
python -c "from src.models import ModelRegistry; print(f'Available models: {ModelRegistry.list_all()}')"

# Check strategy support
python -c "print('Strategy 1 (single_tf): NOT IMPLEMENTED'); print('Strategy 2 (mtf_indicators): PARTIAL (5/9 TFs)'); print('Strategy 3 (mtf_ingestion): NOT IMPLEMENTED')"
```

---

## Getting Help

If issues persist after trying these solutions:

1. **Check documentation:**
   - `docs/CURRENT_LIMITATIONS.md` - Known limitations
   - `docs/INTENDED_ARCHITECTURE.md` - Target state
   - `docs/guides/MTF_STRATEGY_GUIDE.md` - Strategy details

2. **Review implementation status:**
   - `docs/analysis/IMPLEMENTATION_TASKS.md` - Implementation roadmap
   - `docs/MIGRATION_ROADMAP.md` - Migration plan

3. **File an issue:**
   - Include error message
   - Include configuration used
   - Include data shape/size
   - Include model and strategy attempted

---

**Last Updated:** 2026-01-01
**Status:** Reflects current implementation (Strategy 2 partial, Strategies 1 & 3 not implemented)
