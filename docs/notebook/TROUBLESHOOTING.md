# Troubleshooting Guide

Common errors, validation failures, and solutions for the ML Pipeline notebook.

---

## Table of Contents

1. [Data Validation Errors](#data-validation-errors)
2. [Environment Setup Errors](#environment-setup-errors)
3. [Training Errors](#training-errors)
4. [GPU and Memory Errors](#gpu-and-memory-errors)
5. [Cross-Validation Errors](#cross-validation-errors)
6. [Ensemble Errors](#ensemble-errors)
7. [Export Errors](#export-errors)
8. [Validation Checks Reference](#validation-checks-reference)

---

## Data Validation Errors

### Error: No data file found

**Message:** `[ERROR] No data file found for {SYMBOL}!`

**Cause:** File not in expected location or filename doesn't match symbol

**Solutions:**

1. **Check file exists:**
   ```python
   import os
   print(os.listdir('data/raw'))
   # Should show: SI_1m.parquet or similar
   ```

2. **Use custom filename:**
   ```python
   # In Cell 1 configuration
   CUSTOM_DATA_FILE = "SI_1m_historical.parquet"
   ```

3. **Verify path (Colab):**
   ```python
   # Ensure Drive mounted
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   os.chdir('/content/drive/MyDrive/Research')
   ```

4. **Check symbol spelling:**
   ```python
   SYMBOL = "SI"  # Must match filename exactly (case-insensitive)
   ```

---

### Error: Missing OHLCV columns

**Message:** `[ERROR] Missing columns: {columns}`

**Cause:** Data file doesn't have required OHLCV columns

**Required Columns:**
- `datetime` (or `timestamp`, `date`)
- `open`
- `high`
- `low`
- `close`
- `volume`

**Solutions:**

1. **Check column names:**
   ```python
   import pandas as pd
   df = pd.read_parquet('data/raw/SI_1m.parquet')
   print(df.columns.tolist())
   ```

2. **Rename columns:**
   ```python
   df = df.rename(columns={
       'Open': 'open',
       'High': 'high',
       'Low': 'low',
       'Close': 'close',
       'Volume': 'volume',
       'Timestamp': 'datetime'
   })
   df.to_parquet('data/raw/SI_1m.parquet')
   ```

3. **Check data format:**
   - First row should NOT be headers (already parsed)
   - `datetime` should be datetime type, not string

---

### Warning: Date range mismatch

**Message:** `[WARNING] Data range differs from config`

**Cause:** Actual data range doesn't match `DATE_RANGE` configuration

**Impact:** Low (warning only, pipeline continues)

**Solutions:**

1. **Adjust config to match data:**
   ```python
   # If data is 2020-2024 but config says 2019-2024
   DATE_RANGE = "2020-2024"
   ```

2. **Use full dataset:**
   ```python
   DATE_RANGE = "Full Dataset"
   ```

3. **Ignore warning:** If intentional (e.g., testing with subset)

---

### Error: Invalid horizon

**Message:** `[WARNING] TRAINING_HORIZON not in processed horizons`

**Cause:** `TRAINING_HORIZON` not in `HORIZONS` list from Phase 1

**Solutions:**

1. **Check processed horizons:**
   ```python
   # In Cell 3.3 output, look for:
   # Labels: 4 horizons (H5, H10, H15, H20)
   ```

2. **Update training horizon:**
   ```python
   TRAINING_HORIZON = 20  # Must be 5, 10, 15, or 20
   ```

3. **Re-run Phase 1 with different horizons:**
   ```python
   HORIZONS = "5,10,15,20,30"  # Add H30
   RUN_DATA_PIPELINE = True
   ```

---

## Environment Setup Errors

### Error: GPU not available

**Message:** `GPU: Not available (using CPU)`

**Cause:** No CUDA GPU detected or runtime not configured

**Solutions:**

1. **Enable GPU runtime (Colab):**
   - Runtime → Change runtime type → GPU
   - Select T4 or A100
   - Save

2. **Verify CUDA:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))
   ```

3. **Restart runtime:**
   - Runtime → Restart runtime
   - Re-run setup cells

4. **Check quota (Colab free tier):**
   - ~15-20 GPU hours per week
   - Try later if exhausted
   - Upgrade to Colab Pro

5. **Use CPU for non-neural models:**
   ```python
   TRAIN_LSTM = False
   TRAIN_TRANSFORMER = False
   # Boosting models work fine on CPU
   ```

---

### Warning: Neural models without GPU

**Message:** `[WARNING] Neural models selected but no GPU available`

**Cause:** LSTM/GRU/TCN/Transformer enabled but no GPU detected

**Impact:** Training will be VERY slow (10-20x slower)

**Solutions:**

1. **Enable GPU:** See above

2. **Disable neural models:**
   ```python
   TRAIN_LSTM = False
   TRAIN_GRU = False
   TRAIN_TCN = False
   TRAIN_TRANSFORMER = False
   ```

3. **Train on CPU anyway (not recommended):**
   - Expect 2-4 hours per neural model instead of 10-15 min
   - Reduce epochs for testing:
     ```python
     MAX_EPOCHS = 10  # Instead of 50
     ```

---

### Error: Package not installed

**Message:** `ModuleNotFoundError: No module named '{package}'`

**Cause:** Required package not installed

**Solutions:**

1. **Re-run installation cell (Cell 2.2):**
   - Should auto-install all dependencies

2. **Manual install:**
   ```python
   !pip install {package}
   # Example:
   !pip install xgboost
   ```

3. **Install all requirements:**
   ```python
   !pip install -q pandas numpy scikit-learn xgboost lightgbm catboost torch optuna ta pywavelets matplotlib seaborn tqdm pyarrow numba psutil
   ```

4. **Restart runtime after install:**
   - Runtime → Restart runtime
   - Re-run setup cells

---

## Training Errors

### Error: Data not ready

**Message:** `[Error] Data not ready. Run Section 3 first.`

**Cause:** Phase 1 data pipeline not completed

**Solutions:**

1. **Run Phase 1 cells (3.1-3.3):**
   - Cell 3.1: Verify raw data
   - Cell 3.2: Run pipeline
   - Cell 3.3: Verify processed data

2. **Check DATA_READY flag:**
   ```python
   print(DATA_READY)  # Should be True
   ```

3. **Enable pipeline:**
   ```python
   RUN_DATA_PIPELINE = True
   # Re-run Cell 3.2
   ```

---

### Error: No models selected

**Message:** `[Error] No models selected. Enable at least one model toggle.`

**Cause:** All `TRAIN_*` toggles are False

**Solutions:**

1. **Enable at least one model:**
   ```python
   TRAIN_XGBOOST = True
   ```

2. **Check model list:**
   ```python
   # Cell 4.1 shows selected models
   print(MODELS_TO_TRAIN)  # Should not be empty
   ```

---

### Error: Model training failed

**Message:** `[ERROR] {model} training failed: {error}`

**Cause:** Model-specific error during training

**Common Causes:**

1. **Memory error (OOM):**
   - See [GPU and Memory Errors](#gpu-and-memory-errors)

2. **Invalid configuration:**
   ```python
   # Example: Transformer n_heads must divide d_model
   TRANSFORMER_N_HEADS = 8
   TRANSFORMER_D_MODEL = 256  # 256 / 8 = 32 ✓ OK
   # Not OK: d_model=250, n_heads=8 (250 / 8 = 31.25)
   ```

3. **Data shape mismatch:**
   - Neural models require 3D input (samples, sequence, features)
   - Check sequence length compatibility

**Solutions:**

1. **Check full error traceback** (printed in cell output)

2. **Test with default parameters:**
   ```python
   # Use defaults in Cell 1
   ```

3. **Train other models first:**
   - Pipeline continues on failure
   - Other models will still train

4. **Isolate model:**
   ```python
   # Disable all except failing model
   TRAIN_XGBOOST = False
   TRAIN_LIGHTGBM = False
   TRAIN_CATBOOST = False
   TRAIN_LSTM = True  # Test this one
   ```

---

## GPU and Memory Errors

### Error: CUDA out of memory

**Message:** `RuntimeError: CUDA out of memory. Tried to allocate {size} GB`

**Cause:** Model + data don't fit in GPU memory

**Solutions (in order of preference):**

1. **Reduce batch size:**
   ```python
   BATCH_SIZE = 128  # Default: 256
   # Or even lower:
   BATCH_SIZE = 64
   ```

2. **Reduce sequence length:**
   ```python
   SEQUENCE_LENGTH = 30  # Default: 60
   TRANSFORMER_SEQUENCE_LENGTH = 64  # Default: 128
   ```

3. **Enable safe mode:**
   ```python
   SAFE_MODE = True  # Clears memory aggressively
   ```

4. **Clear memory manually:**
   ```python
   clear_memory()  # Function from Cell 2.5
   ```

5. **Use gradient accumulation:**
   ```python
   # Simulates larger batch size without memory cost
   # (requires code modification)
   config = {
       "batch_size": 64,
       "gradient_accumulation_steps": 4  # Effective: 256
   }
   ```

6. **Train on CPU (slow):**
   ```python
   # Disable GPU for specific model
   TRAIN_LSTM = False
   ```

**Memory Requirements:**

| Model | Batch 64 | Batch 128 | Batch 256 | Batch 512 |
|-------|----------|-----------|-----------|-----------|
| LSTM (seq=60) | 4 GB | 6 GB | 10 GB | 18 GB |
| GRU (seq=60) | 3 GB | 5 GB | 8 GB | 14 GB |
| TCN (seq=60) | 5 GB | 7 GB | 12 GB | 20 GB |
| Transformer (seq=128) | 8 GB | 12 GB | 20 GB | 35 GB |

**Available GPU Memory:**
- T4: 15 GB
- A100: 40 GB

---

### Error: Kernel crash / Runtime disconnect

**Cause:** Out of RAM (system memory, not GPU)

**Solutions:**

1. **Enable safe mode:**
   ```python
   SAFE_MODE = True
   ```

2. **Reduce dataset size (testing):**
   ```python
   # In a test cell after loading container
   X_train = X_train[:50000]
   y_train = y_train[:50000]
   ```

3. **Use High-RAM runtime (Colab Pro):**
   - Runtime → Change runtime type → High-RAM
   - 32 GB instead of 12.7 GB

4. **Clear memory between models:**
   ```python
   # Automatically done in training loop
   # Or manually:
   clear_memory()
   ```

---

### Error: Session timeout

**Message:** Runtime disconnected during training

**Cause:** Colab session timeout (12 hours max, 90 min idle)

**Solutions:**

1. **Use Colab Pro:**
   - 24-hour timeout
   - Background execution (runs after browser close)

2. **Add periodic outputs:**
   ```python
   # Automatically handled by tqdm progress bars
   # Prevents idle timeout
   ```

3. **Save checkpoints:**
   ```python
   # Automatically done every epoch by trainer
   # Resume from checkpoint if session ends
   ```

4. **Split work:**
   - Session 1: Phase 1 (data pipeline) → Save to Drive
   - Session 2: Phase 2 (training) → Load from Drive
   - Session 3: Phase 3-5 (CV, ensemble, export)

---

## Cross-Validation Errors

### Warning: No trained models found

**Message:** `[WARNING] No successfully trained models found for CV`

**Cause:** All models failed in Phase 2

**Solutions:**

1. **Check Phase 2 results:**
   ```python
   print(TRAINING_RESULTS)  # Should not be empty
   ```

2. **Train at least one model:**
   ```python
   TRAIN_XGBOOST = True
   # Re-run Cell 4.1
   ```

3. **Fix training errors:** See [Training Errors](#training-errors)

---

### Warning: Tuning failed

**Message:** `[Warning] Hyperparameter tuning failed for {model}`

**Cause:** Optuna optimization error

**Solutions:**

1. **Check error message** (printed with traceback)

2. **Reduce trials:**
   ```python
   CV_N_TRIALS = 10  # Default: 20
   # Test if tuning works with fewer trials
   ```

3. **Disable tuning:**
   ```python
   CV_TUNE_HYPERPARAMS = False
   # Run CV without tuning
   ```

4. **Check parameter space:**
   - Some models may not have tuning defined
   - Check `src/cross_validation/param_spaces.py`

---

### Error: CV failed with NaN

**Cause:** Model produced NaN predictions

**Solutions:**

1. **Check for NaN in data:**
   ```python
   # In Cell 3.3
   print(df.isnull().sum())  # Should be 0 for all columns
   ```

2. **Reduce learning rate:**
   ```python
   # For neural models
   config = {"learning_rate": 0.0001}  # Default: 0.001
   ```

3. **Check for inf/inf:**
   ```python
   import numpy as np
   print(np.isinf(X_train).sum())  # Should be 0
   ```

---

## Ensemble Errors

### Error: Need at least 2 models

**Message:** `[Error] Need at least 2 successfully trained models for ensemble`

**Cause:** Less than 2 models trained successfully

**Solutions:**

1. **Check trained models:**
   ```python
   print(list(TRAINING_RESULTS.keys()))
   # Should have at least 2 models
   ```

2. **Train more models:**
   ```python
   TRAIN_XGBOOST = True
   TRAIN_LIGHTGBM = True
   # Re-run Cell 4.1
   ```

3. **Fix training failures:** See [Training Errors](#training-errors)

---

### Warning: Invalid base model

**Message:** `[!] Skipped (not trained/failed): {models}`

**Cause:** Ensemble configured with models that didn't train

**Solutions:**

1. **Check base model spelling:**
   ```python
   # Correct:
   STACKING_BASE_MODELS = "xgboost,lightgbm,lstm"
   # Incorrect:
   STACKING_BASE_MODELS = "xgb,lgbm,lstm"  # Wrong names
   ```

2. **Verify models trained:**
   ```python
   print(list(TRAINING_RESULTS.keys()))
   # Use only models in this list
   ```

3. **Update ensemble config:**
   ```python
   # If LSTM failed, remove it:
   STACKING_BASE_MODELS = "xgboost,lightgbm"
   ```

---

### Warning: Invalid weights format

**Message:** `[!] Invalid weights format. Using equal weights.`

**Cause:** `VOTING_WEIGHTS` string malformed

**Solutions:**

1. **Use comma-separated floats:**
   ```python
   # Correct:
   VOTING_WEIGHTS = "0.52,0.51,0.54"
   # Incorrect:
   VOTING_WEIGHTS = "[0.52, 0.51, 0.54]"  # No brackets
   VOTING_WEIGHTS = "0.52 0.51 0.54"      # No spaces
   ```

2. **Match model count:**
   ```python
   # 3 models:
   VOTING_BASE_MODELS = "xgboost,lightgbm,catboost"
   # 3 weights:
   VOTING_WEIGHTS = "0.52,0.51,0.54"
   ```

3. **Use equal weights (empty string):**
   ```python
   VOTING_WEIGHTS = ""  # Automatic equal weighting
   ```

---

## Export Errors

### Warning: No models to export

**Message:** `[WARNING] No trained models found to export`

**Cause:** No successful training results

**Solutions:**

1. **Check training results:**
   ```python
   print(TRAINING_RESULTS)
   print(ENSEMBLE_RESULTS)  # If ensembles were trained
   ```

2. **Train models first:**
   - Run Phase 2 (Cell 4.1)

3. **Check export selection:**
   ```python
   EXPORT_MODELS = "all"  # Export everything
   # Or specific:
   EXPORT_MODELS = ["xgboost", "stacking"]
   ```

---

### Error: Model file not found

**Message:** `Model file not found: {path}`

**Cause:** Model checkpoint missing or path incorrect

**Solutions:**

1. **Check run_id:**
   ```python
   # In TRAINING_RESULTS
   print(TRAINING_RESULTS['xgboost']['run_id'])
   ```

2. **Verify file exists:**
   ```python
   import os
   run_id = TRAINING_RESULTS['xgboost']['run_id']
   path = f'experiments/runs/{run_id}/model.pkl'
   print(os.path.exists(path))  # Should be True
   ```

3. **Re-train model:**
   ```python
   # If checkpoint corrupted/deleted
   TRAIN_XGBOOST = True
   # Re-run Cell 4.1
   ```

---

### Error: ONNX export failed

**Message:** `[WARNING] ONNX export failed for {model}: {error}`

**Cause:** Model not compatible with ONNX or conversion error

**Solutions:**

1. **Skip ONNX export:**
   ```python
   EXPORT_ONNX = False
   # PKL export still works
   ```

2. **ONNX compatibility:**
   - ✓ Supported: XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM
   - ✗ Limited support: LSTM, GRU, TCN, Transformer (experimental)

3. **Install ONNX packages:**
   ```python
   !pip install skl2onnx onnx onnxruntime
   ```

---

## Validation Checks Reference

### Automatic Validation Checks

| Cell | Check | Condition | Action |
|------|-------|-----------|--------|
| 3.1 | File exists | `RAW_DATA_FILE` found | Continue |
| 3.1 | OHLCV columns | All 5 present | Continue |
| 3.1 | Date range match | Data covers config | Warn if mismatch |
| 3.3 | Data processed | Parquet files exist | Set `DATA_READY = True` |
| 3.3 | Horizon valid | `TRAINING_HORIZON` in list | Warn if invalid |
| 4.1 | Pipeline enabled | `RUN_MODEL_TRAINING = True` | Skip if False |
| 4.1 | Data ready | `DATA_READY = True` | Error if False |
| 4.1 | Models selected | `len(MODELS_TO_TRAIN) > 0` | Error if empty |
| 4.1 | Horizon valid | Horizon in processed | Error if invalid |
| 4.1 | GPU available | Check CUDA | Warn for neural models |
| 4.5 | Test split exists | `container.has_split('test')` | Error if missing |
| 5.1 | CV enabled | `RUN_CROSS_VALIDATION = True` | Skip if False |
| 5.1 | Models trained | `len(TRAINING_RESULTS) > 0` | Warn if empty |
| 6.1 | Base models count | `>= 2` trained models | Error if < 2 |
| 6.1 | Base model valid | Model in `TRAINING_RESULTS` | Skip invalid |
| 6.1 | Weights count | Matches model count | Warn if mismatch |
| 7.2 | Results exist | Training or ensemble results | Warn if empty |
| 7.2 | Run ID exists | Model has `run_id` | Skip model |
| 7.2 | Model file exists | PKL file found | Skip model |

---

### Manual Validation Commands

```python
# Check data integrity
import pandas as pd
df = pd.read_parquet('data/raw/SI_1m.parquet')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Nulls: {df.isnull().sum().sum()}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Check processed data
train = pd.read_parquet('data/splits/scaled/train_scaled.parquet')
print(f"Features: {len([c for c in train.columns if 'label' not in c])}")
print(f"Labels: {[c for c in train.columns if 'label' in c]}")

# Check training results
print(f"Trained models: {list(TRAINING_RESULTS.keys())}")
for model, result in TRAINING_RESULTS.items():
    print(f"  {model}: F1 = {result['metrics']['val_f1']:.4f}")

# Check memory usage
print_memory_status("Current")

# Check GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## Quick Diagnostic Checklist

Run through this checklist before opening an issue:

- [ ] **Data file exists** and has OHLCV columns
- [ ] **GPU enabled** (if using neural models)
- [ ] **All packages installed** (Cell 2.2 completed)
- [ ] **Phase 1 completed** (`DATA_READY = True`)
- [ ] **At least one model enabled** (`TRAIN_* = True`)
- [ ] **Horizon valid** (`TRAINING_HORIZON` in 5, 10, 15, 20)
- [ ] **No OOM errors** (batch size appropriate for GPU)
- [ ] **Session not timed out** (re-run if disconnected)
- [ ] **Error traceback reviewed** (check cell output)

---

**Last Updated:** 2025-12-28
