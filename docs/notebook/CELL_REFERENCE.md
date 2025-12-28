# Cell-by-Cell Reference

Complete documentation for every cell in the ML Pipeline notebook.

---

## Notebook Structure

| Section | Cells | Purpose |
|---------|-------|---------|
| 1. Configuration | 1 | Master settings panel |
| 2. Environment Setup | 5 | Detect environment, install packages, GPU check |
| 3. Phase 1: Data Pipeline | 3 | Process raw data into scaled datasets |
| 4. Phase 2: Model Training | 5 | Train models, compare, visualize |
| 5. Phase 3: Cross-Validation | 2 | CV and hyperparameter tuning |
| 6. Phase 4: Ensemble | 2 | Train and analyze ensembles |
| 7. Results & Export | 2 | Summary and export package |

**Total Cells:** ~20

---

## Section 1: Master Configuration

### Cell 1.1: Master Configuration Panel

**Purpose:** Single configuration panel for ALL pipeline settings

**Inputs:** None (user edits configuration variables)

**Outputs:**
- `SYMBOL`, `DATE_RANGE`, `DRIVE_DATA_PATH`, `CUSTOM_DATA_FILE`
- `HORIZONS`, `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
- `PURGE_BARS`, `EMBARGO_BARS`, `TRAINING_HORIZON`
- 13 model toggles: `TRAIN_XGBOOST`, `TRAIN_LSTM`, etc.
- Neural settings: `SEQUENCE_LENGTH`, `BATCH_SIZE`, `MAX_EPOCHS`, etc.
- Transformer settings: `TRANSFORMER_*` params
- Boosting settings: `N_ESTIMATORS`, `BOOSTING_EARLY_STOPPING`
- Ensemble settings: base models, weights, meta-learners
- Class balance: `USE_CLASS_WEIGHTS`, `USE_SAMPLE_WEIGHTS`
- CV settings: `RUN_CROSS_VALIDATION`, `CV_N_SPLITS`, etc.
- Execution: `RUN_DATA_PIPELINE`, `RUN_MODEL_TRAINING`, `SAFE_MODE`, `RANDOM_SEED`

**Key Variables:**
```python
# Data
SYMBOL = "SI"
DATE_RANGE = "2019-2024"

# Models to train
TRAIN_XGBOOST = True
TRAIN_LIGHTGBM = True
TRAIN_LSTM = False  # Requires GPU

# Training horizon
TRAINING_HORIZON = 20  # Must be in HORIZONS list
```

**Notes:**
- All configuration happens here - no scattered settings
- See [CONFIGURATION.md](CONFIGURATION.md) for full parameter reference
- Configuration is saved to `experiments/runs/{run_id}/config.json`

---

## Section 2: Environment Setup

### Cell 2.1: Environment Detection & Setup

**Purpose:** Auto-detect Colab vs Local, set up paths, mount Google Drive

**Inputs:** None

**Outputs:**
- `IS_COLAB` (bool): True if running on Google Colab
- `PROJECT_ROOT` (Path): Project root directory
- `RAW_DATA_DIR` (Path): `PROJECT_ROOT/data/raw`
- `SPLITS_DIR` (Path): `PROJECT_ROOT/data/splits/scaled`
- `EXPERIMENTS_DIR` (Path): `PROJECT_ROOT/experiments/runs`

**Actions:**
1. Detect environment via `google.colab` import
2. If Colab:
   - Mount Google Drive at `/content/drive`
   - Set `PROJECT_ROOT` to Drive path
   - Clone/update repository if needed
3. If Local:
   - Set `PROJECT_ROOT` to current directory
4. Create output directories if missing

**Output Example:**
```
Environment: Google Colab
Project root: /content/drive/MyDrive/Research
Directories created: data/raw, data/splits/scaled, experiments/runs
```

---

### Cell 2.2: Install Dependencies

**Purpose:** Install required Python packages (Colab only)

**Inputs:** None

**Outputs:** Verified package imports

**Packages Installed:**
- **Core:** pandas, numpy, scikit-learn
- **ML:** xgboost, lightgbm, catboost
- **Deep Learning:** torch, torchvision, torchaudio
- **Optimization:** optuna
- **Technical Analysis:** ta (ta-lib), pywavelets
- **Visualization:** matplotlib, seaborn
- **Utilities:** tqdm, pyarrow, numba, psutil

**Local Environments:**
- Cell skipped (assumes packages already installed)
- Use `pip install -e .` to install project

**Expected Runtime:** 1-2 minutes (Colab)

---

### Cell 2.3: GPU Detection

**Purpose:** Detect GPU availability and capabilities

**Inputs:** None

**Outputs:**
- `GPU_AVAILABLE` (bool): True if CUDA GPU detected
- `GPU_NAME` (str): GPU model name (e.g., "Tesla T4")
- `GPU_MEMORY` (float): Total GPU memory in GB
- `RECOMMENDED_BATCH` (int): Recommended batch size for GPU

**GPU Detection Logic:**
```python
import torch

GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
    RECOMMENDED_BATCH = 256 if GPU_MEMORY > 12 else 128
else:
    GPU_NAME = "CPU"
    GPU_MEMORY = 0
    RECOMMENDED_BATCH = 64
```

**Warnings:**
- If neural models enabled but no GPU: Prints warning about slow training
- Suggests changing runtime type in Colab (Runtime → Change runtime type → GPU)

**Output Example:**
```
GPU: Available
Device: Tesla T4
Memory: 15.11 GB
Recommended batch size: 256
```

---

### Cell 2.4: Reproducibility Setup

**Purpose:** Set random seeds for reproducible results

**Inputs:** `RANDOM_SEED` from configuration (default: 42)

**Outputs:** Seeds set across all libraries

**Actions:**
```python
import random
import numpy as np
import torch

if RANDOM_SEED != 0:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

**Notes:**
- `RANDOM_SEED = 0`: Disables seeding (non-deterministic, slightly faster)
- `RANDOM_SEED = 42`: Default, ensures reproducibility
- Deterministic mode may reduce performance by ~5-10%

---

### Cell 2.5: Memory Utilities

**Purpose:** Define memory monitoring and cleanup functions

**Inputs:** None

**Outputs:**
- `print_memory_status(label)`: Function to print current memory usage
- `clear_memory()`: Function to clear GPU and system memory

**Functions:**

```python
import gc
import psutil
import torch

def print_memory_status(label=""):
    """Print current memory usage (CPU and GPU)."""
    # CPU memory
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1e9
    print(f"[{label}] CPU Memory: {cpu_mem:.2f} GB")

    # GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_max = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{label}] GPU Memory: {gpu_mem:.2f} GB (peak: {gpu_max:.2f} GB)")

def clear_memory():
    """Clear GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared")
```

**Usage:**
```python
# Monitor memory before/after operations
print_memory_status("Before training")
# ... train model ...
print_memory_status("After training")

# Clear memory when needed
clear_memory()
```

---

## Section 3: Phase 1 - Data Pipeline

### Cell 3.1: Verify Raw Data & Detect Date Range

**Purpose:** Find and validate raw data file, auto-detect date range

**Inputs:**
- `SYMBOL` (from config)
- `RAW_DATA_DIR` (from env setup)
- `CUSTOM_DATA_FILE` (from config, optional)

**Outputs:**
- `RAW_DATA_FILE` (Path): Selected data file
- `DATA_START`, `DATA_END` (datetime): Actual date range in data
- `DATA_START_YEAR`, `DATA_END_YEAR` (int): Start/end years

**File Detection Logic:**
1. If `CUSTOM_DATA_FILE` specified: Use that file
2. Otherwise: Search `RAW_DATA_DIR` for files matching `SYMBOL`
3. Prioritize by scoring:
   - Date range match: +10 points
   - `_1m` pattern: +5 points
   - `historical`: +3 points
   - `.parquet`: +2 points
   - `.csv`: +1 point
4. Select highest-scoring file

**Validation:**
- Check file exists
- Verify OHLCV columns present: `open`, `high`, `low`, `close`, `volume`
- Detect date range from `datetime` column
- Warn if data range differs from `DATE_RANGE` config

**Output Example:**
```
Data File Search
================
Searching in: /content/Research/data/raw
Symbol: SI

Candidates:
  SI_1m_historical.parquet (score: 20)
  SI_5m.csv (score: 4)

Selected: SI_1m_historical.parquet
OHLCV columns: ✓ OK
Date range: 2019-01-02 to 2024-12-27
Years: 2019-2024
Config Match: ✓ Data covers configured range
```

**Error Messages:**
- `[ERROR] No data file found for {SYMBOL}!` → Set `CUSTOM_DATA_FILE` or check path
- `[ERROR] Missing columns: {missing}` → Ensure file has OHLCV columns
- `[WARNING] Data range differs from config` → Adjust `DATE_RANGE` or use full dataset

---

### Cell 3.2: Run Data Pipeline

**Purpose:** Execute Phase 1 data processing pipeline

**Inputs:**
- `RAW_DATA_FILE` (from Cell 3.1)
- Configuration parameters (horizons, splits, purge/embargo)
- `RUN_DATA_PIPELINE` (bool, from config)

**Outputs:**
- `data/splits/scaled/train_h{horizon}.parquet` (for each horizon)
- `data/splits/scaled/val_h{horizon}.parquet`
- `data/splits/scaled/test_h{horizon}.parquet`

**Pipeline Stages:**

1. **Ingest:** Load raw 1-minute data
2. **Clean:** Resample to 5-minute bars, handle gaps
3. **Features:** Generate 150+ technical indicators
4. **MTF:** Add multi-timeframe features (15m, 1h, daily)
5. **Labeling:** Triple-barrier labeling
6. **GA Optimize:** Optuna-based parameter optimization
7. **Final Labels:** Apply optimized parameters
8. **Splits:** Train/val/test with purge/embargo
9. **Scaling:** Robust scaling (train-only fit)
10. **Datasets:** Create TimeSeriesDataContainer
11. **Validation:** Feature correlation checks
12. **Reporting:** Generate completion report

**Expected Runtime:** 20-30 minutes

**Output Example:**
```
Phase 1: Data Pipeline
======================
[✓] Stage 1: Ingest (loaded 1.2M rows)
[✓] Stage 2: Clean (resampled to 240K 5-min bars)
[✓] Stage 3: Features (generated 152 features)
[✓] Stage 4: MTF (added 45 multi-timeframe features)
[✓] Stage 5: Labeling (created initial labels)
[✓] Stage 6: GA Optimize (optimized barrier params)
[✓] Stage 7: Final Labels (applied optimized params)
[✓] Stage 8: Splits (train: 168K, val: 36K, test: 36K)
[✓] Stage 9: Scaling (robust scaler fit on train only)
[✓] Stage 10: Datasets (created containers for H5, H10, H15, H20)
[✓] Stage 11: Validation (removed 12 highly correlated features)
[✓] Stage 12: Reporting (saved completion report)

Pipeline completed in 24m 32s
```

**Skip Condition:**
- If `RUN_DATA_PIPELINE = False`: Prints `[Skipped] Data pipeline disabled` and uses existing files

---

### Cell 3.3: Verify Processed Data

**Purpose:** Validate processed data and extract metadata

**Inputs:** Processed parquet files from Cell 3.2

**Outputs:**
- `FEATURE_COLS` (list): List of feature column names
- `LABEL_COLS` (list): List of label columns (e.g., `label_h5`, `label_h10`, etc.)
- `TRAIN_LEN`, `VAL_LEN`, `TEST_LEN` (int): Number of samples per split
- `label_dists` (dict): Label distribution per horizon
- `DATA_READY` (bool): True if validation passed

**Validation Checks:**

1. **Files Exist:** Check for `train_scaled.parquet`, `val_scaled.parquet`, `test_scaled.parquet`
2. **Columns:** Extract feature and label columns
3. **Label Distribution:** Count Long (-1), Neutral (0), Short (1) per horizon
4. **Horizon Check:** Verify `TRAINING_HORIZON` is in processed horizons

**Output Example:**
```
Processed Data Verification
===========================
✓ Train set: 168,432 samples
✓ Val set: 36,092 samples
✓ Test set: 36,092 samples

Features: 185 columns
  - Technical indicators: 152
  - Multi-timeframe: 45
  - After correlation filter: 185

Labels: 4 horizons (H5, H10, H15, H20)

Label Distribution (H20):
  Long (-1):    32,154 (19.1%)
  Neutral (0): 104,124 (61.8%)
  Short (1):    32,154 (19.1%)

Training Horizon: H20 ✓ Valid

DATA_READY: True
```

**Error Messages:**
- `[ERROR] Processed data not found!` → Run Cell 3.2 first
- `[WARNING] TRAINING_HORIZON not in processed horizons` → Set `TRAINING_HORIZON` to valid value (5, 10, 15, or 20)

---

## Section 4: Phase 2 - Model Training

### Cell 4.1: Train Models

**Purpose:** Train selected models on processed data

**Inputs:**
- `container` (TimeSeriesDataContainer from processed data)
- `MODELS_TO_TRAIN` (derived from `TRAIN_*` toggles)
- Configuration parameters (batch size, epochs, etc.)
- `DATA_READY` (from Cell 3.3)
- `RUN_MODEL_TRAINING` (from config)

**Outputs:**
- `TRAINING_RESULTS` (dict): Results for each trained model
  ```python
  {
      "xgboost": {
          "model": <XGBoostModel>,
          "metrics": {...},
          "run_id": "20251228_143052",
          "training_time": 72.5
      },
      # ... other models
  }
  ```
- Model artifacts in `experiments/runs/{run_id}/`
- `experiments/runs/training_results.json` (summary)

**Training Flow:**

1. **Validation:** Check `DATA_READY`, `RUN_MODEL_TRAINING`, models selected
2. **Load Container:** Load data for `TRAINING_HORIZON`
3. **Class Weights:** Compute if `USE_CLASS_WEIGHTS = True`
4. **For each model:**
   - Create model instance from `ModelRegistry`
   - Configure device (CPU/GPU)
   - Train model with validation set
   - Save model to `experiments/runs/{run_id}/`
   - Collect metrics (accuracy, F1, precision, recall)
   - Handle errors (continue on failure, log error)
5. **Save Summary:** Write `training_results.json`

**Output Example:**
```
Phase 2: Model Training
=======================
Horizon: H20
Train: 168,432 samples | Val: 36,092 samples
Features: 185 | Classes: 3 (Long, Neutral, Short)

Class Weights: {-1: 1.62, 0: 0.54, 1: 1.62}

[1/3] Training XGBoost...
  Device: CPU
  Train time: 72.5s
  Val Accuracy: 0.5234
  Val F1 (macro): 0.5187
  ✓ Saved to: experiments/runs/20251228_143052

[2/3] Training LightGBM...
  Device: CPU
  Train time: 48.2s
  Val Accuracy: 0.5189
  Val F1 (macro): 0.5142
  ✓ Saved to: experiments/runs/20251228_143128

[3/3] Training CatBoost...
  Device: CPU
  Train time: 124.8s
  Val Accuracy: 0.5267
  Val F1 (macro): 0.5221
  ✓ Saved to: experiments/runs/20251228_143310

Training Summary
================
Successfully trained: 3 models
Failed: 0 models
Total time: 4m 5s
```

**Error Handling:**
- Per-model try/except blocks
- Failed models excluded from `TRAINING_RESULTS`
- Training continues on failure

**Skip Conditions:**
- `RUN_MODEL_TRAINING = False`: Skips entire cell
- `DATA_READY = False`: Shows error, requires Cell 3.3
- No models selected: Shows error, requires at least one `TRAIN_*` toggle

---

### Cell 4.2: Compare Models

**Purpose:** Display model comparison table and visualizations

**Inputs:** `TRAINING_RESULTS` (from Cell 4.1)

**Outputs:**
- Comparison DataFrame (printed)
- Bar charts (accuracy, F1, training time)

**Comparison Table:**

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Precision | Recall | Train Time (s) |
|-------|----------|------------|---------------|-----------|--------|----------------|
| CatBoost | 0.5267 | 0.5221 | 0.5245 | 0.5234 | 0.5267 | 124.8 |
| XGBoost | 0.5234 | 0.5187 | 0.5211 | 0.5201 | 0.5234 | 72.5 |
| LightGBM | 0.5189 | 0.5142 | 0.5166 | 0.5156 | 0.5189 | 48.2 |

**Best Model:** CatBoost (highest macro F1: 0.5221)

**Visualizations:**
1. **Accuracy Comparison:** Horizontal bar chart
2. **F1 Score Comparison:** Horizontal bar chart
3. **Training Time:** Horizontal bar chart

**Notes:**
- Best model highlighted in bold
- Sorted by macro F1 (descending)
- If no models trained: Prints `[WARNING] No models to compare`

---

### Cell 4.3: Visualize Training Results

**Purpose:** Generate 5 types of visualizations

**Inputs:**
- `TRAINING_RESULTS` (from Cell 4.1)
- Prediction files from `experiments/runs/{run_id}/predictions.json`

**Outputs:** 5 visualizations (toggleable)

**Visualization 1: Confusion Matrices**
- One matrix per model (3x3 heatmap)
- Rows: True labels (Long, Neutral, Short)
- Columns: Predicted labels
- Values: Counts, normalized by row

**Visualization 2: Feature Importance (Boosting/RF Only)**
- Top 20 features by importance
- Horizontal bar chart per model
- Not available for neural/classical models

**Visualization 3: Learning Curves (Neural Only)**
- Training and validation loss per epoch
- Training and validation accuracy per epoch
- Detects early stopping point
- Not available for boosting/classical models

**Visualization 4: Prediction Distribution**
- Stacked bar chart per model
- Shows % of Long/Neutral/Short predictions
- Compares to true distribution

**Visualization 5: Per-Class Metrics**
- Grouped bar chart: Precision, Recall, F1
- Separate bars for Long, Neutral, Short
- One group per model

**Toggle Flags:**
```python
SHOW_CONFUSION_MATRIX = True
SHOW_FEATURE_IMPORTANCE = True
SHOW_LEARNING_CURVES = True
SHOW_PREDICTION_DISTRIBUTION = True
SHOW_PER_CLASS_METRICS = True
```

---

### Cell 4.4: Transformer Attention Visualization

**Purpose:** Visualize transformer self-attention patterns (transformers only)

**Inputs:**
- Trained transformer model from `TRAINING_RESULTS`
- Validation data
- `sample_index` (which validation sample to visualize)
- `layer_to_visualize` (which layer, -1 for last)
- `head_to_visualize` (which attention head)

**Outputs:**
- Attention heatmaps per head (8 heatmaps for 8-head transformer)
- Attention analysis (most attended positions, entropy)
- Interpretability insights

**Visualizations:**

1. **All Heads Heatmap:** 8 subplots showing attention weights
   - X-axis: Key positions (0-127 for seq_len=128)
   - Y-axis: Query positions (0-127)
   - Color: Attention weight (0-1)

2. **Single Head Analysis:**
   - Larger heatmap for selected head
   - Most attended positions
   - Attention entropy (measure of focus vs diffuse)

3. **Interpretability Insights:**
   - **Recency bias:** Does model attend more to recent timesteps?
   - **Long-range dependencies:** Are distant positions attended?
   - **Position patterns:** Are there structured attention patterns?

**Example Output:**
```
Transformer Attention Visualization
===================================
Sample: Validation sample #100
Layer: 2 (last layer)
Sequence length: 128 timesteps

Attention Analysis (Head 0):
  Most attended positions: [127, 126, 125] (recent timesteps)
  Attention entropy: 2.34 bits (moderately focused)
  Recency bias: Strong (70% attention on last 20 timesteps)

Insights:
  ✓ Model exhibits strong recency bias
  ✓ Some long-range dependencies to position 15-20
  - Limited attention to middle sequence (40-80)
```

**Skip Condition:**
- If transformer not trained: Prints `[Skipped] Transformer not trained`

---

### Cell 4.5: Test Set Performance

**Purpose:** Evaluate models on held-out test set

**Inputs:**
- `TRAINING_RESULTS` (from Cell 4.1)
- Test data from `container`

**Outputs:**
- `TEST_RESULTS` (dict): Test metrics per model
- Per-model test metrics (printed)
- Generalization gap analysis
- Sample predictions (first 20 rows)

**Evaluation Flow:**

1. Load test split from container
2. For each trained model:
   - Load model from checkpoint
   - Generate predictions on test set
   - Compute metrics (accuracy, F1, precision, recall)
   - Save predictions to `experiments/runs/{run_id}/test_predictions.json`
3. Compare to validation metrics (generalization gap)
4. Display sample predictions

**Output Example:**
```
Test Set Evaluation
===================
Test set: 36,092 samples

XGBoost:
  Test Accuracy: 0.5198
  Test F1 (macro): 0.5152
  Generalization Gap: 0.0035 (0.67% drop from validation)

  Per-class metrics:
    Long (-1):    Precision: 0.48, Recall: 0.52, F1: 0.50
    Neutral (0):  Precision: 0.64, Recall: 0.61, F1: 0.62
    Short (1):    Precision: 0.47, Recall: 0.51, F1: 0.49

LightGBM:
  Test Accuracy: 0.5154
  Test F1 (macro): 0.5108
  Generalization Gap: 0.0034 (0.66% drop from validation)

CatBoost:
  Test Accuracy: 0.5231
  Test F1 (macro): 0.5186
  Generalization Gap: 0.0035 (0.67% drop from validation)

Sample Predictions (first 20 rows):
     true_label  xgboost_pred  lightgbm_pred  catboost_pred
0            0             0              0              0
1            1             1              0              1
2           -1            -1             -1             -1
...
```

**Metrics Computed:**
- Accuracy
- Macro F1 (average across classes)
- Weighted F1 (weighted by class support)
- Per-class precision, recall, F1
- Confusion matrix

**Generalization Gap:**
- `gap = val_metric - test_metric`
- Small gap (< 0.01): Good generalization
- Large gap (> 0.05): Potential overfitting

---

## Section 5: Phase 3 - Cross-Validation

### Cell 5.1: Run Cross-Validation

**Purpose:** Run purged K-fold CV with optional hyperparameter tuning

**Inputs:**
- Trained models from `TRAINING_RESULTS`
- `RUN_CROSS_VALIDATION` (bool, from config)
- `CV_N_SPLITS`, `CV_TUNE_HYPERPARAMS`, `CV_N_TRIALS` (from config)

**Outputs:**
- `CV_RESULTS` (dict): CV results per model
  ```python
  {
      "xgboost": {
          "mean_f1": 0.5142,
          "std_f1": 0.0087,
          "fold_scores": [0.51, 0.52, 0.50, 0.52, 0.51],
          "stability": "Good"
      }
  }
  ```
- `TUNING_RESULTS` (dict): Tuning results if `CV_TUNE_HYPERPARAMS = True`

**CV Process:**

1. **Setup:** Create PurgedKFold splitter (5 folds default)
2. **For each model:**
   - Load unscaled data (or prescaled if `CV_USE_PRESCALED = True`)
   - For each fold:
     - Split data with purge/embargo
     - Train model on fold training data
     - Evaluate on fold validation data
     - Collect fold metrics
   - Compute mean and std across folds
   - Assign stability grade (Excellent/Good/Fair/Poor)
3. **If tuning enabled:**
   - Run Optuna hyperparameter search
   - Track best parameters and best score
   - Save to `TUNING_RESULTS`

**Expected Runtime:**
- **CV only (5 folds):** 5-10 min per model
- **CV + Tuning (20 trials):** 30-60 min per model (boosting), 2-3 hours (neural)

**Output Example:**
```
Cross-Validation
================
Configuration:
  N Splits: 5
  Purge: 60 bars
  Embargo: 1440 bars
  Tuning: Enabled (20 trials)

XGBoost:
  Fold 1: F1 = 0.5124
  Fold 2: F1 = 0.5198
  Fold 3: F1 = 0.5087
  Fold 4: F1 = 0.5201
  Fold 5: F1 = 0.5142

  Mean F1: 0.5150 ± 0.0049
  Stability: Excellent (CV < 1%)

  Hyperparameter Tuning:
    Trial 1/20: F1 = 0.5098 (max_depth=4, learning_rate=0.05)
    Trial 2/20: F1 = 0.5187 (max_depth=6, learning_rate=0.1)
    ...
    Trial 20/20: F1 = 0.5142 (max_depth=5, learning_rate=0.08)

    Best params: {max_depth: 6, learning_rate: 0.1, subsample: 0.9}
    Best F1: 0.5187
    Improvement over default: +0.0037 (+0.71%)

LightGBM:
  Mean F1: 0.5108 ± 0.0062
  Stability: Good (CV = 1.2%)

CatBoost:
  Mean F1: 0.5176 ± 0.0041
  Stability: Excellent (CV < 1%)
```

**Stability Grading:**
- **Excellent:** CV < 1%
- **Good:** 1% ≤ CV < 2%
- **Fair:** 2% ≤ CV < 5%
- **Poor:** CV ≥ 5%

**Skip Condition:**
- `RUN_CROSS_VALIDATION = False`: Prints `[Skipped] Cross-validation disabled`

---

### Cell 5.2: Hyperparameter Tuning Results

**Purpose:** Display tuning results and recommendations

**Inputs:**
- `TUNING_RESULTS` (from Cell 5.1)
- `TRAINING_RESULTS` (from Cell 4.1)

**Outputs:**
- Parameter comparison table (default vs tuned)
- Improvement analysis
- Retrain recommendations

**Output Example:**
```
Hyperparameter Tuning Results
==============================

XGBoost:
  Default params:
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8

  Tuned params:
    max_depth: 6 (unchanged)
    learning_rate: 0.1 (unchanged)
    subsample: 0.9 (+0.1)
    colsample_bytree: 0.75 (-0.05)

  Performance:
    Default F1: 0.5150
    Tuned F1: 0.5187
    Improvement: +0.0037 (+0.71%)

  Recommendation: RETRAIN with tuned params (improvement > 0.5%)

LightGBM:
  Improvement: +0.0012 (+0.23%)
  Recommendation: Keep default params (improvement < 0.5%)

CatBoost:
  Improvement: +0.0045 (+0.87%)
  Recommendation: RETRAIN with tuned params (improvement > 0.5%)
```

**Retrain Recommendations:**
- **Retrain:** If improvement > 0.5%
- **Keep default:** If improvement < 0.5%

**Notes:**
- Tuning results saved to `experiments/runs/tuning_results.json`
- Can manually retrain with tuned params by updating config

---

## Section 6: Phase 4 - Ensemble

### Cell 6.1: Train Ensemble

**Purpose:** Train voting, stacking, and/or blending ensembles

**Inputs:**
- Trained base models from `TRAINING_RESULTS`
- `TRAIN_VOTING`, `TRAIN_STACKING`, `TRAIN_BLENDING` (from config)
- Ensemble configuration (base models, weights, meta-learners)

**Outputs:**
- `ENSEMBLE_RESULTS` (dict): Results for each ensemble
  ```python
  {
      "voting": {
          "model": <VotingEnsemble>,
          "metrics": {...},
          "base_models": ["xgboost", "lightgbm", "catboost"]
      }
  }
  ```
- Ensemble comparison table

**Training Flow:**

1. **Voting Ensemble:**
   - Combine base model predictions via weighted average
   - No training required (uses pre-trained models)
   - Weights from config or equal weighting

2. **Stacking Ensemble:**
   - Generate out-of-fold predictions from base models
   - Train meta-learner on OOF predictions
   - Meta-learner: Logistic/XGBoost/RandomForest

3. **Blending Ensemble:**
   - Split training data: 80% for base models, 20% holdout
   - Train base models on 80% split
   - Train meta-learner on 20% holdout predictions

**Output Example:**
```
Ensemble Training
=================

Voting Ensemble:
  Base models: XGBoost, LightGBM, CatBoost
  Weights: [0.52, 0.51, 0.54] (proportional to val F1)
  Val Accuracy: 0.5298
  Val F1 (macro): 0.5254
  Improvement over best base model: +0.0033 (+0.63%)

Stacking Ensemble:
  Base models: XGBoost, LightGBM, LSTM
  Meta-learner: Logistic Regression
  Generating OOF predictions (5 folds)...
    Fold 1/5: Done
    Fold 2/5: Done
    Fold 3/5: Done
    Fold 4/5: Done
    Fold 5/5: Done
  Training meta-learner...
  Val Accuracy: 0.5342
  Val F1 (macro): 0.5287
  Improvement over best base model: +0.0066 (+1.27%)

Comparison:
| Ensemble | Accuracy | F1 (Macro) | Improvement |
|----------|----------|------------|-------------|
| Stacking | 0.5342   | 0.5287     | +1.27%      |
| Voting   | 0.5298   | 0.5254     | +0.63%      |

Best Ensemble: Stacking
```

**Validation:**
- Requires at least 2 successfully trained base models
- Invalid base model names skipped with warning
- Weight count must match model count (voting only)

---

### Cell 6.2: Ensemble Analysis & Diversity

**Purpose:** Analyze ensemble diversity and base model contributions

**Inputs:**
- `ENSEMBLE_RESULTS` (from Cell 6.1)
- `TRAINING_RESULTS` (from Cell 4.1)

**Outputs:**
- Diversity metrics (disagreement rate, Q-statistic)
- Base model contributions
- Contribution charts
- Best ensemble recommendation

**Diversity Metrics:**

1. **Disagreement Rate:** % of samples where models disagree
2. **Q-Statistic:** Pairwise correlation (-1 to 1)
   - Q < 0: Negatively correlated (good diversity)
   - Q ≈ 0: Independent
   - Q > 0.5: Highly correlated (poor diversity)

**Output Example:**
```
Ensemble Diversity Analysis
===========================

Base Model Contributions (Stacking):
  XGBoost:  34.2% (weight: 0.342 in meta-learner)
  LightGBM: 31.8% (weight: 0.318)
  LSTM:     34.0% (weight: 0.340)

Diversity Metrics:
  Overall Disagreement: 42.3%

  Pairwise Q-Statistics:
    XGBoost ↔ LightGBM:  Q = 0.62 (correlated)
    XGBoost ↔ LSTM:      Q = 0.18 (diverse)
    LightGBM ↔ LSTM:     Q = 0.21 (diverse)

  Average Q: 0.34 (moderate diversity)

Interpretation:
  ✓ Good diversity between boosting and neural models
  - Boosting models are correlated (expected)
  - Adding neural model improves diversity

Recommendation:
  Best ensemble: Stacking
  Reason: Highest F1 (0.5287) + good diversity (Q = 0.34)
```

**Contribution Chart:**
- Horizontal bar chart showing meta-learner weights
- Indicates which base models are most valuable

---

## Section 7: Results & Export

### Cell 7.1: Final Summary

**Purpose:** Display complete pipeline summary

**Inputs:**
- `TRAINING_RESULTS`
- `ENSEMBLE_RESULTS` (if any)
- `CV_RESULTS` (if any)
- `TEST_RESULTS` (if any)

**Outputs:** Formatted summary report

**Summary Sections:**

1. **Configuration:** Symbol, horizon, date range
2. **Data:** Train/val/test sizes, features, labels
3. **Models Trained:** List with training time
4. **Best Model:** Highest validation F1
5. **Ensemble Results:** If any ensembles trained
6. **CV Results:** If CV was run
7. **Test Performance:** If test evaluation done
8. **Recommendations:** Next steps

**Example Output:**
```
═══════════════════════════════════════════════════════════
                    FINAL SUMMARY
═══════════════════════════════════════════════════════════

Configuration:
  Symbol: SI
  Horizon: H20 (100 minutes forward)
  Date Range: 2019-2024
  Random Seed: 42

Data:
  Train: 168,432 samples
  Val: 36,092 samples
  Test: 36,092 samples
  Features: 185
  Classes: 3 (Long, Neutral, Short)

Models Trained: 3
  1. XGBoost (72.5s)
  2. LightGBM (48.2s)
  3. CatBoost (124.8s)

Best Single Model: CatBoost
  Val Accuracy: 0.5267
  Val F1 (macro): 0.5221
  Test F1 (macro): 0.5186 (gap: 0.0035)

Ensemble Results:
  Voting: Val F1 = 0.5254
  Stacking: Val F1 = 0.5287 ⭐ BEST

Cross-Validation:
  CatBoost: 0.5176 ± 0.0041 (Excellent stability)
  XGBoost: 0.5150 ± 0.0049 (Excellent stability)

Recommendations:
  ✓ Deploy: Stacking ensemble (best performance)
  ✓ Alternative: CatBoost (simpler, nearly as good)
  - Consider retraining XGBoost with tuned params (+0.71% improvement)
  - Test performance shows good generalization (gap < 1%)
```

---

### Cell 7.2: Export Model Package

**Purpose:** Export models and artifacts as professional package

**Inputs:**
- All results dicts
- Export configuration:
  - `EXPORT_MODELS` (list or "all" or "best")
  - `EXPORT_ONNX` (bool, default False)
  - `CREATE_ZIP` (bool, default False)

**Outputs:** Complete export directory

**Export Structure:**
```
experiments/exports/{timestamp}_{symbol}_H{horizon}/
├── models/{model}/
│   ├── model.pkl              # Pickle (sklearn compatible)
│   ├── model.onnx             # ONNX (optional, cross-platform)
│   └── config.json            # Training configuration
├── predictions/{model}/
│   ├── val_predictions.csv    # Validation predictions
│   ├── test_predictions.csv   # Test predictions
│   └── predictions_summary.json
├── metrics/
│   ├── training_metrics.json  # All training metrics
│   ├── test_metrics.json      # Test set metrics
│   └── cv_results.json        # CV results (if available)
├── visualizations/{model}/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── learning_curves.png
│   └── attention_heatmap.png  # Transformers only
├── model_cards/
│   └── {model}_card.md        # Model card (usage, performance, notes)
├── data/
│   ├── feature_names.txt
│   ├── label_mapping.json
│   └── data_stats.json
├── manifest.json              # Complete package manifest
└── README.md                  # Package overview
```

**Expected Runtime:** 2-5 minutes

**Output Example:**
```
Export Model Package
====================
Export directory: experiments/exports/20251228_SI_H20/

Exporting 4 models: xgboost, lightgbm, catboost, stacking

[1/4] Exporting XGBoost...
  ✓ Model saved: models/xgboost/model.pkl
  ✓ Config saved: models/xgboost/config.json
  ✓ Val predictions: predictions/xgboost/val_predictions.csv
  ✓ Test predictions: predictions/xgboost/test_predictions.csv
  ✓ Visualizations: 3 files
  ✓ Model card: model_cards/xgboost_card.md

[2/4] Exporting LightGBM...
  ✓ Completed

[3/4] Exporting CatBoost...
  ✓ Completed

[4/4] Exporting Stacking...
  ✓ Completed

Metrics:
  ✓ training_metrics.json
  ✓ test_metrics.json
  ✓ cv_results.json

Data:
  ✓ feature_names.txt (185 features)
  ✓ label_mapping.json
  ✓ data_stats.json

Manifest:
  ✓ manifest.json
  ✓ README.md

ZIP Archive:
  ✓ 20251228_SI_H20.zip (24.3 MB)

Export completed!
Total size: 24.3 MB
Location: experiments/exports/20251228_SI_H20/
```

**ONNX Export:**
- Only for compatible models (boosting, classical)
- Neural models: Requires `torch.onnx.export` (experimental)
- Enables cross-platform deployment (C++, Java, etc.)

---

**Last Updated:** 2025-12-28
