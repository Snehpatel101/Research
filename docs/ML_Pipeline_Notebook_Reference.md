# ML Pipeline Notebook - Complete Reference

**File:** `notebooks/ML_Pipeline.ipynb`
**Purpose:** Complete ML training pipeline for OHLCV time series with 13 models
**Last Updated:** 2025-12-27

---

## Table of Contents

1. [Overview](#overview)
2. [Cell-by-Cell Documentation](#cell-by-cell-documentation)
3. [Configuration Parameters](#configuration-parameters)
4. [Features Generated](#features-generated)
5. [Model Parameters](#model-parameters)
6. [Output Files](#output-files)
7. [Validation Checks](#validation-checks)
8. [Error Messages](#error-messages)
9. [Dependencies](#dependencies)

---

## Overview

### Model Support (13 Models)

| Family | Models | Count |
|--------|--------|-------|
| Boosting | XGBoost, LightGBM, CatBoost | 3 |
| Neural | LSTM, GRU, TCN, Transformer | 4 |
| Classical | Random Forest, Logistic Regression, SVM | 3 |
| Ensemble | Voting, Stacking, Blending | 3 |

### Pipeline Phases

1. **Configuration** - Master settings panel
2. **Environment Setup** - Auto-detects Colab vs Local
3. **Phase 1: Data Pipeline** - Clean -> Features -> Labels -> Splits -> Scale
4. **Phase 2: Model Training** - Train any of 13 model types
5. **Phase 3: Cross-Validation** - Purged K-fold with optional tuning
6. **Phase 4: Ensemble** - Combine models intelligently
7. **Results & Export** - Professional packages with ONNX

---

## Cell-by-Cell Documentation

### Section 1: Master Configuration

#### Cell 1.1: Master Configuration Panel
- **Purpose:** Configure ALL pipeline settings in one place
- **Inputs:** None (user configuration)
- **Outputs:** Global configuration variables
- **Key Variables Created:**
  - `SYMBOL`, `DATE_RANGE`, `HORIZONS`, `MODELS_TO_TRAIN`
  - Split ratios, purge/embargo bars
  - Model-specific settings

### Section 2: Environment Setup

#### Cell 2.1: Environment Detection & Setup
- **Purpose:** Detect Colab vs Local environment, set up paths
- **Inputs:** None
- **Outputs:**
  - `IS_COLAB` (bool)
  - `PROJECT_ROOT`, `RAW_DATA_DIR`, `SPLITS_DIR`, `EXPERIMENTS_DIR` (Path objects)
- **Actions:**
  - Mounts Google Drive (Colab only)
  - Clones/updates repository (Colab only)
  - Creates output directories

#### Cell 2.2: Install Dependencies
- **Purpose:** Install required packages (Colab only)
- **Inputs:** None
- **Outputs:** Verified imports
- **Packages Installed (Colab):**
  - torch, xgboost, lightgbm, catboost, optuna
  - ta, pywavelets, scikit-learn, pandas, numpy
  - matplotlib, tqdm, pyarrow, numba, psutil

#### Cell 2.3: GPU Detection
- **Purpose:** Detect GPU availability and capabilities
- **Inputs:** None
- **Outputs:**
  - `GPU_AVAILABLE` (bool)
  - `GPU_NAME` (str)
  - `GPU_MEMORY` (float, GB)
  - `RECOMMENDED_BATCH` (int)
- **Warnings:** Alerts if neural models selected without GPU

#### Cell 2.4: Reproducibility Setup
- **Purpose:** Set random seeds for reproducibility
- **Inputs:** `RANDOM_SEED` from config
- **Outputs:** Seeds set for Python, NumPy, PyTorch
- **Note:** Sets `torch.backends.cudnn.deterministic = True`

#### Cell 2.5: Memory Utilities
- **Purpose:** Define memory monitoring and cleanup functions
- **Inputs:** None
- **Outputs:**
  - `print_memory_status(label)` function
  - `clear_memory()` function

### Section 3: Phase 1 - Data Pipeline

#### Cell 3.1: Verify Raw Data & Detect Date Range
- **Purpose:** Find and validate raw data file, auto-detect date range
- **Inputs:** `SYMBOL`, `RAW_DATA_DIR`, `CUSTOM_DATA_FILE`
- **Outputs:**
  - `RAW_DATA_FILE` (Path)
  - `DATA_START`, `DATA_END` (datetime)
  - `DATA_START_YEAR`, `DATA_END_YEAR` (int)
- **File Detection Logic:**
  - Searches for files matching symbol (case-insensitive)
  - Priority scoring: date range match (+10), `_1m` pattern (+5), `historical` (+3), `.parquet` (+2)
- **Validation:**
  - Checks for OHLCV columns (open, high, low, close, volume)
  - Validates date range against configuration

#### Cell 3.2: Run Data Pipeline
- **Purpose:** Execute Phase 1 data processing pipeline
- **Inputs:** `RAW_DATA_FILE`, config parameters
- **Outputs:**
  - `data/splits/scaled/train_scaled.parquet`
  - `data/splits/scaled/val_scaled.parquet`
  - `data/splits/scaled/test_scaled.parquet`
- **Pipeline Stages:**
  1. Load raw 1-minute data
  2. Clean and resample to 5-minute bars
  3. Generate 150+ technical features
  4. Apply triple-barrier labeling
  5. Create train/val/test splits with purge/embargo
  6. Scale features (train-only fit)

#### Cell 3.3: Verify Processed Data
- **Purpose:** Validate processed data and extract metadata
- **Inputs:** Processed parquet files
- **Outputs:**
  - `FEATURE_COLS` (list)
  - `LABEL_COLS` (list)
  - `TRAIN_LEN`, `VAL_LEN`, `TEST_LEN` (int)
  - `label_dists` (dict)
  - `DATA_READY` (bool)

### Section 4: Phase 2 - Model Training

#### Cell 4.1: Train Models
- **Purpose:** Train selected models on processed data
- **Inputs:** `container`, `MODELS_TO_TRAIN`, config parameters
- **Outputs:**
  - `TRAINING_RESULTS` (dict)
  - Model artifacts in `experiments/runs/{run_id}/`
  - `experiments/runs/training_results.json`
- **Features:**
  - Per-model error handling (continues on failure)
  - Class weight computation if enabled
  - Sample weight handling
  - GPU/CPU device selection for neural models

#### Cell 4.2: Compare Models
- **Purpose:** Display model comparison table and visualizations
- **Inputs:** `TRAINING_RESULTS`
- **Outputs:** Comparison DataFrame, bar charts

#### Cell 4.3: Visualize Training Results
- **Purpose:** Generate 5 types of visualizations
- **Inputs:** `TRAINING_RESULTS`, prediction files
- **Visualizations:**
  1. Confusion matrices (all models)
  2. Feature importance (boosting/RF models, top 20)
  3. Learning curves (neural models, loss + accuracy)
  4. Prediction distribution (stacked bar chart)
  5. Per-class metrics (precision, recall, F1)
- **Toggles:** Each visualization can be enabled/disabled

#### Cell 4.4: Transformer Attention Visualization
- **Purpose:** Visualize transformer self-attention patterns
- **Inputs:** Trained transformer model, validation data
- **Outputs:**
  - Attention heatmaps per head
  - Attention analysis (most attended positions, entropy)
  - Interpretability insights (recency bias, long-range dependencies)
- **Parameters:**
  - `sample_index`: Which validation sample to visualize
  - `layer_to_visualize`: Layer index (-1 for last)
  - `head_to_visualize`: Head index for detailed analysis

#### Cell 4.5: Test Set Performance
- **Purpose:** Evaluate models on held-out test set
- **Inputs:** `TRAINING_RESULTS`, test data
- **Outputs:**
  - `TEST_RESULTS` (dict)
  - Per-model test metrics
  - Generalization gap analysis
  - Sample predictions
- **Metrics Calculated:**
  - Accuracy, Macro F1, Weighted F1
  - Per-class precision, recall, F1
  - Confusion matrix

### Section 5: Phase 3 - Cross-Validation

#### Cell 5.1: Run Cross-Validation
- **Purpose:** Run purged K-fold CV with optional hyperparameter tuning
- **Inputs:** Trained models, CV configuration
- **Outputs:**
  - `CV_RESULTS` (dict)
  - `TUNING_RESULTS` (dict, if tuning enabled)
- **CV Features:**
  - Purged K-fold (time-series aware)
  - Per-fold or pre-scaled data options
  - Optuna-based hyperparameter tuning
  - Stability grading (Excellent/Good/Fair/Poor)

#### Cell 5.2: Hyperparameter Tuning Results
- **Purpose:** Display tuning results and recommendations
- **Inputs:** `TUNING_RESULTS`, `TRAINING_RESULTS`
- **Outputs:**
  - Parameter comparison tables
  - Improvement analysis
  - Retrain recommendations

### Section 6: Phase 4 - Ensemble

#### Cell 6.1: Train Ensemble
- **Purpose:** Train voting, stacking, and/or blending ensembles
- **Inputs:** Trained base models, ensemble configuration
- **Outputs:**
  - `ENSEMBLE_RESULTS` (dict)
  - Ensemble comparison table
- **Ensemble Types:**
  - Voting: Weighted/unweighted prediction averaging
  - Stacking: Meta-learner on OOF predictions
  - Blending: Meta-learner on holdout predictions

#### Cell 6.2: Ensemble Analysis & Diversity
- **Purpose:** Analyze ensemble diversity and base model contributions
- **Inputs:** `ENSEMBLE_RESULTS`, `TRAINING_RESULTS`
- **Outputs:**
  - Diversity metrics (disagreement, Q-statistic)
  - Base model contributions
  - Contribution charts
  - Best ensemble recommendation

### Section 7: Results & Export

#### Cell 7.1: Final Summary
- **Purpose:** Display pipeline summary
- **Inputs:** All results dicts
- **Outputs:** Formatted summary with best model

#### Cell 7.2: Export Model Package
- **Purpose:** Export models and artifacts as professional package
- **Inputs:** All results, export configuration
- **Outputs:**
  - `experiments/exports/{timestamp}_{symbol}_H{horizon}/`
  - Models (PKL, optional ONNX)
  - Predictions (CSV)
  - Metrics (JSON)
  - Visualizations (PNG)
  - Model cards (MD)
  - manifest.json, README.md
  - Optional ZIP archive

---

## Configuration Parameters

### Data Configuration

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `SYMBOL` | string | "SI" | "SI", "MES", "MGC", "ES", "GC", "NQ", "CL", "HG", "ZB", "ZN" | Futures contract symbol |
| `DATE_RANGE` | string | "2019-2024" | "2019-2024", "2020-2024", "2021-2024", "2022-2024", "2023-2024", "Full Dataset" | Date range filter |
| `DRIVE_DATA_PATH` | string | "research/data/raw" | Any valid path | Google Drive path (Colab) |
| `CUSTOM_DATA_FILE` | string | "" | Any filename | Override auto-detection |

### Pipeline Configuration

| Parameter | Type | Default | Range/Values | Description |
|-----------|------|---------|--------------|-------------|
| `HORIZONS` | string | "5,10,15,20" | Comma-separated ints | Prediction horizons (bars) |
| `TRAIN_RATIO` | float | 0.70 | 0.0-1.0 | Training set ratio |
| `VAL_RATIO` | float | 0.15 | 0.0-1.0 | Validation set ratio |
| `TEST_RATIO` | float | 0.15 | 0.0-1.0 | Test set ratio |
| `PURGE_BARS` | int | 60 | >= 0 | Bars to purge around boundaries |
| `EMBARGO_BARS` | int | 1440 | >= 0 | Embargo period (~5 days at 5-min) |

### Model Training Configuration

| Parameter | Type | Default | Range/Values | Description |
|-----------|------|---------|--------------|-------------|
| `TRAINING_HORIZON` | int | 20 | 5, 10, 15, 20 | Which horizon to train on |

#### Model Selection (Boolean Toggles)

| Parameter | Default | Model |
|-----------|---------|-------|
| `TRAIN_XGBOOST` | True | XGBoost |
| `TRAIN_LIGHTGBM` | True | LightGBM |
| `TRAIN_CATBOOST` | True | CatBoost |
| `TRAIN_RANDOM_FOREST` | False | Random Forest |
| `TRAIN_LOGISTIC` | False | Logistic Regression |
| `TRAIN_SVM` | False | Support Vector Machine |
| `TRAIN_LSTM` | False | LSTM |
| `TRAIN_GRU` | False | GRU |
| `TRAIN_TCN` | False | TCN |
| `TRAIN_TRANSFORMER` | False | Transformer |
| `TRAIN_VOTING` | False | Voting Ensemble |
| `TRAIN_STACKING` | False | Stacking Ensemble |
| `TRAIN_BLENDING` | False | Blending Ensemble |

### Neural Network Settings

| Parameter | Type | Default | Range/Values | Description |
|-----------|------|---------|--------------|-------------|
| `SEQUENCE_LENGTH` | int | 60 | 30-120 (step 10) | Input sequence length |
| `BATCH_SIZE` | int | 256 | 64, 128, 256, 512, 1024 | Training batch size |
| `MAX_EPOCHS` | int | 50 | >= 1 | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | int | 10 | >= 1 | Epochs before early stopping |

### Transformer Settings

| Parameter | Type | Default | Range/Values | Description |
|-----------|------|---------|--------------|-------------|
| `TRANSFORMER_SEQUENCE_LENGTH` | int | 128 | >= 32 | Transformer input length |
| `TRANSFORMER_N_HEADS` | int | 8 | 4, 8, 16 | Number of attention heads |
| `TRANSFORMER_N_LAYERS` | int | 3 | 2, 3, 4, 6 | Number of transformer layers |
| `TRANSFORMER_D_MODEL` | int | 256 | 128, 256, 512 | Model dimension |

### Boosting Settings

| Parameter | Type | Default | Range/Values | Description |
|-----------|------|---------|--------------|-------------|
| `N_ESTIMATORS` | int | 500 | >= 1 | Number of boosting rounds |
| `BOOSTING_EARLY_STOPPING` | int | 50 | >= 1 | Early stopping rounds |

### Ensemble Configuration

#### Voting Ensemble

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `VOTING_BASE_MODELS` | string | "xgboost,lightgbm,catboost" | Comma-separated base models |
| `VOTING_WEIGHTS` | string | "" | Comma-separated weights (empty = equal) |

#### Stacking Ensemble

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `STACKING_BASE_MODELS` | string | "xgboost,lightgbm,lstm" | - | Comma-separated base models |
| `STACKING_META_LEARNER` | string | "logistic" | "logistic", "xgboost", "random_forest" | Meta-learner type |
| `STACKING_N_FOLDS` | int | 5 | >= 2 | CV folds for OOF predictions |

#### Blending Ensemble

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `BLENDING_BASE_MODELS` | string | "xgboost,lightgbm,random_forest" | - | Comma-separated base models |
| `BLENDING_META_LEARNER` | string | "logistic" | "logistic", "xgboost", "random_forest" | Meta-learner type |
| `BLENDING_HOLDOUT_RATIO` | float | 0.2 | 0.0-1.0 | Holdout ratio for blending |

### Class Balancing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_CLASS_WEIGHTS` | bool | True | Use balanced class weights |
| `USE_SAMPLE_WEIGHTS` | bool | True | Use quality-based sample weights |

### Cross-Validation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUN_CROSS_VALIDATION` | bool | False | Enable CV phase |
| `CV_N_SPLITS` | int | 5 | Number of CV folds |
| `CV_TUNE_HYPERPARAMS` | bool | False | Enable Optuna tuning |
| `CV_N_TRIALS` | int | 20 | Optuna trial count |
| `CV_USE_PRESCALED` | bool | True | Use pre-scaled data (faster) |

### Execution Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUN_DATA_PIPELINE` | bool | True | Run Phase 1 |
| `RUN_MODEL_TRAINING` | bool | True | Run Phase 2 |
| `SAFE_MODE` | bool | False | Low-memory mode |
| `RANDOM_SEED` | int | 42 | Seed for reproducibility (0 = random) |

---

## Features Generated

The data pipeline generates **150+ features** organized into categories:

### Technical Indicators (via `ta` library)

#### Trend Indicators
- SMA (5, 10, 20, 50)
- EMA (5, 10, 20, 50)
- MACD (12, 26, 9)
- ADX
- Ichimoku Cloud components

#### Momentum Indicators
- RSI (14)
- Stochastic (%K, %D)
- Williams %R
- ROC (Rate of Change)
- CCI (Commodity Channel Index)

#### Volatility Indicators
- Bollinger Bands (upper, middle, lower, %B)
- ATR (Average True Range)
- Keltner Channels
- Donchian Channels

#### Volume Indicators
- OBV (On-Balance Volume)
- VWAP
- CMF (Chaikin Money Flow)
- MFI (Money Flow Index)
- ADI (Accumulation/Distribution Index)

### Custom Features

#### Price-Based
- Log returns
- Price momentum (various periods)
- Range (high - low)
- Gap (open - prev_close)
- Body ratio (close - open) / (high - low)

#### Volatility
- Rolling std of returns
- Parkinson volatility
- Garman-Klass volatility

#### Microstructure
- Bid-ask spread proxies
- Order flow imbalance
- Volume ratios

#### Wavelet Features
- Wavelet decomposition (pywavelets)
- Approximation and detail coefficients
- Multi-scale analysis

#### Multi-Timeframe Features
- 15-min, 30-min, 1-hour, 4-hour, daily aggregations
- Cross-timeframe momentum
- Higher timeframe trend direction

---

## Model Parameters

### XGBoost

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 500 | Number of boosting rounds |
| `early_stopping_rounds` | int | 50 | Early stopping patience |
| `learning_rate` | float | 0.1 | Step size shrinkage |
| `max_depth` | int | 6 | Maximum tree depth |
| `subsample` | float | 0.8 | Row subsampling ratio |
| `colsample_bytree` | float | 0.8 | Column subsampling ratio |
| `objective` | string | "multi:softprob" | Loss function |
| `eval_metric` | string | "mlogloss" | Evaluation metric |

### LightGBM

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 500 | Number of boosting rounds |
| `early_stopping_rounds` | int | 50 | Early stopping patience |
| `learning_rate` | float | 0.1 | Step size shrinkage |
| `num_leaves` | int | 31 | Max leaves per tree |
| `max_depth` | int | -1 | Maximum tree depth |
| `feature_fraction` | float | 0.8 | Column subsampling ratio |
| `bagging_fraction` | float | 0.8 | Row subsampling ratio |

### CatBoost

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iterations` | int | 500 | Number of boosting rounds |
| `early_stopping_rounds` | int | 50 | Early stopping patience |
| `learning_rate` | float | 0.1 | Step size shrinkage |
| `depth` | int | 6 | Maximum tree depth |
| `task_type` | string | "CPU" | Device type |
| `use_gpu` | bool | False | Use GPU |
| `verbose` | bool | False | Print training progress |

### Random Forest

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of trees |
| `max_depth` | int | None | Maximum tree depth |
| `min_samples_split` | int | 2 | Min samples to split |
| `min_samples_leaf` | int | 1 | Min samples per leaf |
| `class_weight` | string | "balanced" | Class weight strategy |

### Logistic Regression

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C` | float | 1.0 | Regularization strength |
| `solver` | string | "lbfgs" | Optimization algorithm |
| `max_iter` | int | 1000 | Maximum iterations |
| `class_weight` | string | "balanced" | Class weight strategy |

### SVM

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C` | float | 1.0 | Regularization parameter |
| `kernel` | string | "rbf" | Kernel type |
| `gamma` | string | "scale" | Kernel coefficient |
| `class_weight` | string | "balanced" | Class weight strategy |
| `probability` | bool | True | Enable probability estimates |

### LSTM / GRU

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | int | 128 | Hidden layer size |
| `num_layers` | int | 2 | Number of RNN layers |
| `dropout` | float | 0.2 | Dropout rate |
| `bidirectional` | bool | False | Bidirectional RNN |
| `batch_size` | int | 256 | Training batch size |
| `learning_rate` | float | 0.001 | Optimizer learning rate |
| `max_epochs` | int | 50 | Maximum epochs |
| `early_stopping_patience` | int | 10 | Early stopping patience |

### TCN (Temporal Convolutional Network)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_channels` | list | [64, 128, 256] | Channel sizes per layer |
| `kernel_size` | int | 3 | Convolution kernel size |
| `dropout` | float | 0.2 | Dropout rate |
| `batch_size` | int | 256 | Training batch size |
| `learning_rate` | float | 0.001 | Optimizer learning rate |

### Transformer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 256 | Model dimension |
| `n_heads` | int | 8 | Number of attention heads |
| `n_layers` | int | 3 | Number of encoder layers |
| `d_ff` | int | 1024 | Feedforward dimension |
| `dropout` | float | 0.1 | Dropout rate |
| `max_seq_len` | int | 128 | Maximum sequence length |
| `batch_size` | int | 256 | Training batch size |
| `learning_rate` | float | 0.0001 | Optimizer learning rate |

### Voting Ensemble

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model_names` | list | ["xgboost", "lightgbm", "catboost"] | Base models |
| `voting_type` | string | "soft" | "hard" or "soft" voting |
| `weights` | list | None | Per-model weights |

### Stacking Ensemble

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model_names` | list | ["xgboost", "lightgbm", "lstm"] | Base models |
| `meta_learner` | string | "logistic" | Meta-learner type |
| `use_proba` | bool | True | Use probabilities as features |

### Blending Ensemble

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model_names` | list | ["xgboost", "lightgbm", "random_forest"] | Base models |
| `meta_learner` | string | "logistic" | Meta-learner type |
| `holdout_ratio` | float | 0.2 | Holdout set ratio |

---

## Output Files

### Data Pipeline Outputs

| Path | Format | Description |
|------|--------|-------------|
| `data/splits/scaled/train_scaled.parquet` | Parquet | Scaled training data |
| `data/splits/scaled/val_scaled.parquet` | Parquet | Scaled validation data |
| `data/splits/scaled/test_scaled.parquet` | Parquet | Scaled test data |

### Model Training Outputs

| Path | Format | Description |
|------|--------|-------------|
| `experiments/runs/{run_id}/model.pkl` | Pickle | Trained model |
| `experiments/runs/{run_id}/config.json` | JSON | Training configuration |
| `experiments/runs/{run_id}/metrics.json` | JSON | Performance metrics |
| `experiments/runs/{run_id}/predictions.json` | JSON | Validation predictions |
| `experiments/runs/{run_id}/feature_importance.json` | JSON | Feature importance (boosting) |
| `experiments/runs/{run_id}/training_history.json` | JSON | Learning curves (neural) |
| `experiments/runs/{run_id}/test_predictions.json` | JSON | Test set predictions |
| `experiments/runs/training_results.json` | JSON | Summary of all training |

### Cross-Validation Outputs

| Path | Format | Description |
|------|--------|-------------|
| `experiments/runs/tuning_results.json` | JSON | Hyperparameter tuning results |

### Export Package Structure

```
experiments/exports/{timestamp}_{symbol}_H{horizon}/
  models/
    {model_name}/
      model.pkl
      model.onnx (optional)
      config.json
  predictions/
    {model_name}/
      val_predictions.csv
      test_predictions.csv
      predictions_summary.json
  metrics/
    training_metrics.json
    test_metrics.json
    cv_results.json
  visualizations/
    {model_name}/
      *.png
  model_cards/
    {model_name}_card.md
  data/
    feature_names.txt
    label_mapping.json
    data_stats.json
  manifest.json
  README.md
```

---

## Validation Checks

### Data Validation (Cell 3.1)

| Check | Condition | Message |
|-------|-----------|---------|
| File exists | `RAW_DATA_FILE` found | "Selected: {filename}" |
| File not found | No matching file | "[ERROR] No data file found for {SYMBOL}!" |
| OHLCV columns | All 5 present | "OHLCV columns: OK" |
| Missing columns | Any missing | "[ERROR] Missing columns: {missing}" |
| Date range match | Data covers config | "Config Match: Data covers {range}" |
| Date range mismatch | Data differs | "[WARNING] Data range differs from config" |

### Processed Data Validation (Cell 3.3)

| Check | Condition | Message |
|-------|-----------|---------|
| Data exists | Parquet files present | Shows sample counts |
| Data missing | No parquet files | "[ERROR] Processed data not found!" |
| Horizon valid | `TRAINING_HORIZON` in `HORIZON_LIST` | OK (implicit) |
| Horizon invalid | Horizon not in list | "[WARNING] TRAINING_HORIZON not in HORIZON_LIST" |

### Training Validation (Cell 4.1)

| Check | Condition | Message |
|-------|-----------|---------|
| Pipeline disabled | `RUN_MODEL_TRAINING = False` | "[Skipped] Model training disabled" |
| Data not ready | `DATA_READY = False` | "[Error] Data not ready" |
| No models selected | `MODELS_TO_TRAIN` empty | "[Error] No models selected" |
| Invalid horizon | Horizon not in processed | "[ERROR] TRAINING_HORIZON not in processed horizons" |

### Ensemble Validation (Cell 6.1)

| Check | Condition | Message |
|-------|-----------|---------|
| Not enough models | < 2 successful base models | "[Error] Need at least 2 successfully trained models" |
| Invalid base model | Model not trained/failed | "[!] Skipped (not trained/failed): {models}" |
| Weight mismatch | Weights count != models | "[!] Weights count != models count" |

### Export Validation (Cell 7.2)

| Check | Condition | Message |
|-------|-----------|---------|
| No results | No training/ensemble results | "[WARNING] No trained models found" |
| No run_id | Model has no run_id | "No run_id found for {model}" |
| Model file missing | PKL not found | "Model file not found: {path}" |
| Custom selection empty | No models specified | "Custom selection requires model names" |

---

## Error Messages

### Environment Setup Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| "PyTorch: not installed" | torch missing | Install via pip |
| "GPU: Not available (using CPU)" | No CUDA GPU | Use Runtime > Change runtime type > GPU |
| "[WARNING] Neural models selected but no GPU" | Neural models + no GPU | Training will be slow, consider GPU |

### Data Pipeline Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| "[ERROR] No data file found for {SYMBOL}!" | File not in expected location | Set CUSTOM_DATA_FILE or check path |
| "[ERROR] Missing columns: {columns}" | OHLCV columns missing | Check data file format |
| "[ERROR] Pipeline failed" | Pipeline execution error | Check detailed traceback |

### Training Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| "[Error] Data not ready. Run Section 3 first." | Data pipeline not run | Execute cells 3.1-3.3 |
| "[Error] No models selected" | All model toggles False | Enable at least one model |
| "[ERROR] TRAINING_HORIZON not in processed horizons" | Horizon mismatch | Set TRAINING_HORIZON to valid value |
| "[ERROR] {model} training failed: {error}" | Model-specific failure | Check model config and data |

### Test Evaluation Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| "[ERROR] Test split not found in container!" | Test data missing | Re-run data pipeline |
| "[WARNING] Model file not found" | Model checkpoint missing | Re-train the model |
| "[WARNING] Model has no predict method" | Incompatible model | Check model implementation |

### Cross-Validation Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| "[WARNING] No successfully trained models found" | All training failed | Re-run training |
| "[Warning] Tuning failed" | Optuna error | Check param space definition |
| "[Skipped] No param space defined" | Model lacks tuning config | Define param space or skip |

### Ensemble Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| "[X] Need at least 2 valid base models" | Insufficient trained models | Train more base models |
| "[!] Invalid weights format" | Malformed weight string | Use comma-separated floats |

### Export Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| "No training results found" | Model has no entry | Check training completed |
| "No run_id found" | Missing run identifier | Re-train the model |
| "Model file not found" | PKL missing | Check experiments directory |
| "ONNX export failed" | Conversion error | Check model compatibility |

---

## Dependencies

### Core Dependencies

```python
# Data manipulation
pandas
numpy
pyarrow  # For parquet I/O

# Machine learning
scikit-learn
xgboost
lightgbm
catboost

# Deep learning
torch

# Optimization
optuna

# Technical analysis
ta  # Technical analysis library
pywavelets  # Wavelet transforms

# Visualization
matplotlib
seaborn

# Utilities
tqdm  # Progress bars
numba  # JIT compilation
psutil  # Memory monitoring
```

### Optional Dependencies

```python
# ONNX export
skl2onnx
onnx
onnxruntime

# Colab-specific
google.colab  # Drive mounting, file download
```

### Import Statement (Cell 2.2)

```python
import os
import sys
import gc
import json
import time
import pickle
import joblib
import shutil
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import warnings
warnings.filterwarnings('ignore')
```

### Project Imports

```python
from src.phase1.pipeline_config import PipelineConfig
from src.pipeline.runner import PipelineRunner
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
from src.models import ModelRegistry, Trainer, TrainerConfig
from src.cross_validation import PurgedKFold, PurgedKFoldConfig
from src.cross_validation.cv_runner import TimeSeriesOptunaTuner
from src.cross_validation.param_spaces import get_param_space

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    precision_recall_fscore_support, ConfusionMatrixDisplay
)
from sklearn.preprocessing import RobustScaler
```

---

## Quick Reference

### Command Line Equivalents

```bash
# Train single model
python scripts/train_model.py --model xgboost --horizon 20

# Train neural model
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60

# Run cross-validation
python scripts/run_cv.py --models xgboost,lightgbm --horizons 20 --n-splits 5

# Train ensemble
python scripts/train_model.py --model voting --horizon 20

# List all available models
python scripts/train_model.py --list-models
```

### Model Family Summary

| Family | Models | Best For |
|--------|--------|----------|
| Boosting | XGBoost, LightGBM, CatBoost | Fast, accurate, tabular data |
| Classical | Random Forest, Logistic, SVM | Baselines, interpretability |
| Neural | LSTM, GRU, TCN, Transformer | Sequential patterns |
| Ensemble | Voting, Stacking, Blending | Combined predictions |

---

**Document generated:** 2025-12-27
