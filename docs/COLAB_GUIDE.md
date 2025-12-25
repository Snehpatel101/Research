# Google Colab Setup Guide

Complete guide for running the ML Model Factory on Google Colab with GPU acceleration.

---

## Quick Start

1. Open one of the notebooks in Google Colab:
   - `notebooks/01_quickstart.ipynb` - Complete pipeline walkthrough
   - `notebooks/02_train_all_models.ipynb` - Train all 12 models
   - `notebooks/03_cross_validation.ipynb` - CV and hyperparameter tuning
   - `notebooks/Phase1_Pipeline_Colab.ipynb` - Phase 1 data pipeline

2. Enable GPU runtime (see [GPU Configuration](#gpu-configuration))

3. Mount Google Drive (see [Data Setup](#data-setup))

4. Run cells sequentially

---

## GPU Configuration

### Enable GPU Runtime

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** from "Hardware accelerator" dropdown
3. Choose **T4** (free tier) or **A100** (Colab Pro)
4. Click **Save**

### Verify GPU Access

Run this cell to verify GPU is available:

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU detected. Some models will be slow.")
```

Expected output:
```
CUDA available: True
Device count: 1
Device name: Tesla T4
Memory: 15.11 GB
```

### GPU Recommendations by Model

| Model | Minimum GPU | Recommended | Training Time (H20) |
|-------|-------------|-------------|---------------------|
| XGBoost | None (CPU) | T4 | 5-10 min |
| LightGBM | None (CPU) | T4 | 3-8 min |
| CatBoost | None (CPU) | T4 | 8-15 min |
| LSTM | T4 | A100 | 30-60 min |
| GRU | T4 | A100 | 25-50 min |
| TCN | T4 | A100 | 20-40 min |
| Random Forest | None (CPU) | None | 10-20 min |
| Logistic | None (CPU) | None | 2-5 min |
| SVM | None (CPU) | None | 15-30 min |
| Voting | Inherited | Inherited | Sum of base models |
| Stacking | Inherited | Inherited | Sum + meta-learner |
| Blending | Inherited | Inherited | Sum + meta-learner |

---

## Data Setup

### Option 1: Mount Google Drive (Recommended)

If your data is in Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Set data directory
import os
os.chdir('/content/drive/MyDrive/Research')

# Verify data exists
assert os.path.exists('data/raw/MGC_1m.parquet'), "Data not found!"
print("Data directory mounted successfully")
```

### Option 2: Upload Data Directly

For small datasets (< 100MB):

```python
from google.colab import files
import shutil

# Upload files
uploaded = files.upload()

# Move to data directory
os.makedirs('data/raw', exist_ok=True)
for filename in uploaded.keys():
    shutil.move(filename, f'data/raw/{filename}')
    print(f'Moved {filename} to data/raw/')
```

### Option 3: Download from Cloud Storage

From Google Cloud Storage or S3:

```python
# From Google Cloud Storage
!gsutil -m cp -r gs://your-bucket/data ./data

# From S3 (requires credentials)
!aws s3 sync s3://your-bucket/data ./data
```

---

## Installation

### Install Required Packages

The notebooks include installation cells. Run them in this order:

```python
# Core dependencies
!pip install -q pandas numpy scikit-learn

# ML libraries
!pip install -q xgboost lightgbm catboost

# PyTorch (for neural models)
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optimization
!pip install -q optuna

# Visualization
!pip install -q matplotlib seaborn plotly

# Project-specific
!pip install -q pyarrow fastparquet ta-lib
```

### Clone Repository

If running outside the repository:

```python
# Clone repo
!git clone https://github.com/your-username/Research.git
%cd Research

# Install in development mode
!pip install -e .
```

---

## Running Notebooks

### Notebook 1: Quickstart (`01_quickstart.ipynb`)

**Purpose:** End-to-end pipeline walkthrough
**Time:** 15-20 minutes
**GPU:** Optional (recommended for neural models)

**What it does:**
1. Loads preprocessed data from Phase 1
2. Trains a single model (XGBoost by default)
3. Evaluates on validation set
4. Visualizes results

**Key cells:**
```python
# Load data
from src.phase1.stages.datasets import TimeSeriesDataContainer
container = TimeSeriesDataContainer.from_parquet_dir("data/splits/scaled", horizon=20)

# Train model
from src.models.trainer import train_model
results = train_model(model="xgboost", horizon=20, device="cuda")

# Evaluate
print(f"Val Accuracy: {results['val_accuracy']:.4f}")
print(f"Val F1: {results['val_f1']:.4f}")
```

---

### Notebook 2: Train All Models (`02_train_all_models.ipynb`)

**Purpose:** Train all 12 models across all horizons
**Time:** 2-4 hours (GPU required for neural models)
**GPU:** Required (T4 minimum, A100 recommended)

**What it does:**
1. Trains all 12 models for each horizon (H5, H10, H15, H20)
2. Saves results to `experiments/runs/`
3. Generates comparison plots
4. Creates performance summary table

**Key configuration:**
```python
# Configure which models to train
MODELS = [
    "xgboost", "lightgbm", "catboost",      # Boosting
    "lstm", "gru", "tcn",                   # Neural
    "random_forest", "logistic", "svm",     # Classical
]

HORIZONS = [5, 10, 15, 20]

# Run training loop
for model in MODELS:
    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"Training {model.upper()} for H{horizon}")
        print(f"{'='*60}")

        results = train_model(
            model=model,
            horizon=horizon,
            device="cuda" if model in ["lstm", "gru", "tcn"] else "cpu"
        )
```

**Output:**
- Trained models in `experiments/runs/{run_id}/`
- Performance CSV: `experiments/all_models_summary.csv`
- Comparison plots: `experiments/plots/`

---

### Notebook 3: Cross-Validation (`03_cross_validation.ipynb`)

**Purpose:** Run purged k-fold CV and hyperparameter tuning
**Time:** 1-3 hours depending on trials
**GPU:** Recommended

**What it does:**
1. Runs 5-fold purged cross-validation
2. Performs walk-forward feature selection
3. Tunes hyperparameters with Optuna
4. Generates out-of-fold predictions for stacking

**Key cells:**
```python
from src.cross_validation.cv_runner import run_cross_validation

# Run CV
cv_results = run_cross_validation(
    models=["xgboost", "lstm"],
    horizons=[20],
    n_splits=5,
    tune=True,
    n_trials=50,
    save_oof=True
)

# View results
print(f"Mean F1: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
print(f"Best hyperparameters: {cv_results['best_params']}")
```

---

### Notebook 4: Phase 1 Pipeline (`Phase1_Pipeline_Colab.ipynb`)

**Purpose:** Run Phase 1 data pipeline from scratch
**Time:** 30-60 minutes
**GPU:** Optional (minor speedup)

**What it does:**
1. Ingests raw 1-minute OHLCV data
2. Resamples to 5-minute bars
3. Engineers 150+ features
4. Creates triple-barrier labels
5. Splits data with purge/embargo
6. Scales features

**Prerequisites:**
- Raw data: `data/raw/{SYMBOL}_1m.parquet`
- Columns: `datetime`, `open`, `high`, `low`, `close`, `volume`

**Run the pipeline:**
```python
# Run Phase 1 pipeline
!./pipeline run --symbols MGC --horizons 5,10,15,20

# Verify outputs
!ls -lh data/splits/scaled/
```

---

## Common Issues and Solutions

### Issue 1: GPU Not Detected

**Symptom:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Verify runtime type is set to GPU (Runtime → Change runtime type)
2. Restart runtime (Runtime → Restart runtime)
3. Check quota limits (Colab Pro users have higher limits)
4. Try later if free tier GPU quota is exhausted

### Issue 2: Out of Memory (OOM)

**Symptom:** `CUDA out of memory` error during training

**Solutions:**
1. Reduce batch size:
   ```python
   # In training config
   config = {"batch_size": 128}  # Default is 256
   ```

2. Use gradient accumulation:
   ```python
   config = {
       "batch_size": 64,
       "gradient_accumulation_steps": 4  # Effective batch size: 256
   }
   ```

3. Reduce sequence length (neural models):
   ```python
   config = {"seq_len": 30}  # Default is 60
   ```

4. Use CPU for that model:
   ```python
   results = train_model(model="lstm", device="cpu")
   ```

### Issue 3: Slow Training on CPU

**Symptom:** Training takes hours instead of minutes

**Solutions:**
1. Enable GPU runtime (see [GPU Configuration](#gpu-configuration))
2. Use boosting models (faster on CPU than neural models)
3. Reduce dataset size for testing:
   ```python
   # Subsample data
   X_train = X_train[:10000]
   y_train = y_train[:10000]
   ```

### Issue 4: Package Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**
1. Install project in development mode:
   ```python
   !pip install -e .
   ```

2. Add src to Python path:
   ```python
   import sys
   sys.path.append('/content/Research/src')
   ```

3. Verify current directory:
   ```python
   import os
   print(f"Current directory: {os.getcwd()}")
   # Should be: /content/Research or /content/drive/MyDrive/Research
   ```

### Issue 5: Session Timeout

**Symptom:** Runtime disconnects during long training

**Solutions:**
1. Use Colab Pro (longer timeout limits)
2. Add periodic outputs to keep session alive:
   ```python
   import time
   for epoch in range(epochs):
       # Training code
       if epoch % 10 == 0:
           print(f"Epoch {epoch}/{epochs}")  # Prevents timeout
   ```

3. Save checkpoints frequently:
   ```python
   # In training loop
   if epoch % 10 == 0:
       model.save(f"checkpoint_epoch_{epoch}.pkl")
   ```

4. Use background execution (Colab Pro):
   ```python
   # Enable background execution
   # Runtime → Run after disconnection
   ```

---

## Performance Optimization

### 1. Use Mixed Precision Training

For neural models on A100/T4:

```python
# Enable automatic mixed precision
config = {
    "mixed_precision": True,  # Faster training, lower memory
    "dtype": "bfloat16"        # A100 supports bfloat16
}
```

### 2. Optimize Data Loading

```python
# Use smaller batches of data
container = TimeSeriesDataContainer.from_parquet_dir(
    "data/splits/scaled",
    horizon=20,
    in_memory=False  # Stream from disk instead of loading all
)
```

### 3. Parallel Training

Train multiple models in parallel using separate notebooks:

1. Open multiple Colab tabs
2. Assign different models to each tab:
   - Tab 1: `train_model(model="xgboost", ...)`
   - Tab 2: `train_model(model="lstm", ...)`
   - Tab 3: `train_model(model="random_forest", ...)`

### 4. Use Precomputed Features

If Phase 1 is already complete, skip feature engineering:

```python
# Load precomputed features
container = TimeSeriesDataContainer.from_parquet_dir("data/splits/scaled", horizon=20)
```

---

## Saving Results

### Save to Google Drive

```python
# Save trained model
output_dir = "/content/drive/MyDrive/Research/experiments"
os.makedirs(output_dir, exist_ok=True)

# Save model
model.save(f"{output_dir}/xgboost_h20.pkl")

# Save metrics
import json
with open(f"{output_dir}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Download Results Locally

```python
# Zip results
!zip -r results.zip experiments/

# Download
from google.colab import files
files.download('results.zip')
```

### Export to Notebook

```python
# Save key metrics to notebook cells
print(f"Final Val Accuracy: {results['val_accuracy']:.4f}")
print(f"Final Val F1: {results['val_f1']:.4f}")
print(f"Training Time: {results['training_time_seconds']:.2f}s")

# This output is saved with the notebook
```

---

## Cost Optimization

### Free Tier Limits

- **GPU hours:** ~15-20 hours per week
- **RAM:** 12.7 GB
- **Disk:** 107 GB
- **Session timeout:** 12 hours (90 min idle)

### Colab Pro ($9.99/month)

- **GPU hours:** ~50-100 hours per month
- **RAM:** Up to 32 GB (High-RAM runtime)
- **Disk:** Up to 200 GB
- **Session timeout:** 24 hours
- **Background execution:** Yes
- **Priority GPU access:** Yes (A100 available)

### Cost-Saving Tips

1. **Train boosting models first** (fast, CPU-only)
2. **Use CPU for classical models** (Random Forest, Logistic, SVM)
3. **Reserve GPU for neural models** (LSTM, GRU, TCN)
4. **Run CV overnight** (Colab Pro background execution)
5. **Subsample data for testing** before full runs

---

## Example Workflows

### Workflow 1: Quick Model Comparison

**Goal:** Compare 3 models in 30 minutes
**Notebook:** `01_quickstart.ipynb`

```python
models = ["xgboost", "lightgbm", "random_forest"]
results = {}

for model in models:
    print(f"Training {model}...")
    results[model] = train_model(model=model, horizon=20, device="cpu")

# Compare
for model, res in results.items():
    print(f"{model}: F1={res['val_f1']:.4f}")
```

### Workflow 2: Neural Model Training

**Goal:** Train all 3 neural models with GPU
**Notebook:** `02_train_all_models.ipynb`

```python
neural_models = ["lstm", "gru", "tcn"]
results = {}

for model in neural_models:
    print(f"Training {model}...")
    results[model] = train_model(
        model=model,
        horizon=20,
        device="cuda",
        config={
            "batch_size": 256,
            "max_epochs": 100,
            "seq_len": 60,
            "hidden_size": 128
        }
    )
```

### Workflow 3: Hyperparameter Tuning

**Goal:** Find best XGBoost hyperparameters
**Notebook:** `03_cross_validation.ipynb`

```python
from src.cross_validation.cv_runner import run_cross_validation

cv_results = run_cross_validation(
    models=["xgboost"],
    horizons=[20],
    n_splits=5,
    tune=True,
    n_trials=100,  # 100 Optuna trials
    save_oof=True
)

print(f"Best params: {cv_results['best_params']}")
print(f"Best F1: {cv_results['best_f1']:.4f}")
```

### Workflow 4: Ensemble Training

**Goal:** Train voting ensemble of top 3 models
**Notebook:** `02_train_all_models.ipynb`

```python
# Train base models
base_models = ["xgboost", "lightgbm", "lstm"]
for model in base_models:
    train_model(model=model, horizon=20)

# Train voting ensemble
ensemble_results = train_model(
    model="voting",
    base_models=base_models,
    horizon=20,
    device="cuda"  # Inherited from base models
)

print(f"Ensemble F1: {ensemble_results['val_f1']:.4f}")
```

---

## Advanced: TPU Support

Colab also offers TPUs. To use TPUs with PyTorch:

### Enable TPU Runtime

1. Runtime → Change runtime type → TPU
2. Install XLA:
   ```python
   !pip install cloud-tpu-client
   !pip install torch_xla
   ```

3. Configure PyTorch for TPU:
   ```python
   import torch_xla.core.xla_model as xm

   device = xm.xla_device()
   model = model.to(device)
   ```

**Note:** TPUs are optimized for large models (transformers). For our use case, GPUs are more suitable.

---

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] GPU runtime is enabled
- [ ] CUDA is available (`torch.cuda.is_available()`)
- [ ] Data is mounted/uploaded correctly
- [ ] All packages are installed
- [ ] Current directory is project root
- [ ] No OOM errors (reduce batch size if needed)
- [ ] Session hasn't timed out

---

## Additional Resources

- **Colab Documentation:** https://colab.research.google.com/notebooks/intro.ipynb
- **PyTorch GPU Guide:** https://pytorch.org/docs/stable/notes/cuda.html
- **Optuna Documentation:** https://optuna.readthedocs.io/
- **Project README:** `../README.md`
- **Architecture Map:** `../ARCHITECTURE_MAP.md`

---

## Support

If you encounter issues:

1. Check [Common Issues](#common-issues-and-solutions)
2. Search GitHub issues
3. Create new issue with:
   - Notebook name
   - Error message
   - GPU/runtime info
   - Steps to reproduce

---

*Last updated: 2025-12-25*
