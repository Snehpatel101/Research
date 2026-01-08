# Notebook Setup Guide

Complete guide for running the ML Model Factory notebook in Jupyter and Google Colab.

**Last Updated:** 2026-01-01

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Jupyter Setup](#local-jupyter-setup)
3. [Google Colab Setup](#google-colab-setup)
4. [Data Configuration](#data-configuration)
5. [GPU Configuration](#gpu-configuration)
6. [Common Issues](#common-issues)
7. [Performance Optimization](#performance-optimization)

---

## Quick Start

**Notebook Location:** `notebooks/ML_Pipeline.ipynb`

### Minimum Configuration

```python
SYMBOL = "SI"           # Your contract symbol
TRAIN_XGBOOST = True    # Enable at least one model
TRAINING_HORIZON = 20   # Prediction horizon (bars forward)
```

### Basic Workflow

```
1. Open notebooks/ML_Pipeline.ipynb in Colab or Jupyter
2. Configure Section 1 (symbol, models, horizons)
3. Run All Cells (Ctrl+F9 / Cmd+F9)
4. Export trained models from Section 7
```

---

## Local Jupyter Setup

### Prerequisites

```bash
# Python 3.10+
python --version  # Should be 3.10+

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Installation

```bash
# Core dependencies
pip install pandas numpy scikit-learn pyarrow

# ML libraries
pip install xgboost lightgbm catboost

# PyTorch (with CUDA for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Jupyter
pip install jupyterlab jupyter

# Additional utilities
pip install optuna ta pywavelets matplotlib seaborn tqdm
```

### Starting Jupyter

```bash
# From project root
cd /path/to/Research

# Start JupyterLab (recommended)
jupyter lab

# Or classic Jupyter
jupyter notebook
```

### Opening the Notebook

1. Navigate to `notebooks/` in the Jupyter file browser
2. Open `ML_Pipeline.ipynb`
3. Verify kernel is set to your virtual environment

### Directory Structure

```
Research/
  data/
    raw/                # Place your data here
      SI_1m.parquet     # Required: 1-minute OHLCV data
  notebooks/
    ML_Pipeline.ipynb   # Main notebook
  experiments/
    runs/               # Training outputs
```

---

## Google Colab Setup

### Quick Start

1. **Open notebook in Colab:**
   - Click "Open in Colab" badge in notebook, OR
   - Upload `.ipynb` file to Colab

2. **Enable GPU runtime:**
   - Runtime -> Change runtime type -> GPU (T4 or A100)

3. **Mount Google Drive:**
   - Run Cell 2.1 to mount Drive and set up paths

4. **Run All Cells:**
   - Runtime -> Run all (Ctrl+F9 / Cmd+F9)

### Enable GPU Runtime

1. Click **Runtime** -> **Change runtime type**
2. Select **GPU** from "Hardware accelerator" dropdown
3. Choose **T4** (free tier) or **A100** (Colab Pro)
4. Click **Save**

**Verification:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Mount Google Drive

**Upload data to Google Drive first:**
1. Create folder: `MyDrive/Research/data/raw/`
2. Upload data file: `{SYMBOL}_1m.parquet` or `.csv`

**Mount Drive in notebook:**
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/Research')

# Verify data exists
assert os.path.exists('data/raw/SI_1m.parquet'), "Data not found!"
print("Data directory mounted successfully")
```

### Alternative: Direct Upload

For small datasets (< 100 MB):

```python
from google.colab import files
import shutil
import os

# Upload files
print("Select your data file:")
uploaded = files.upload()

# Move to data directory
os.makedirs('data/raw', exist_ok=True)
for filename in uploaded.keys():
    shutil.move(filename, f'data/raw/{filename}')
    print(f'Moved {filename} to data/raw/')
```

### Colab Pricing

| Tier | GPU Hours | RAM | Session Timeout | Background Exec |
|------|-----------|-----|-----------------|-----------------|
| **Free** | ~15-20/week | 12.7 GB | 12 hours | No |
| **Pro** ($9.99/mo) | ~50-100/month | 32 GB | 24 hours | Yes |

---

## Data Configuration

### Data Requirements

**File Format:** Parquet (recommended) or CSV

**Required Columns:**
```
timestamp (datetime)  # Bar timestamp
open (float)          # Open price
high (float)          # High price
low (float)           # Low price
close (float)         # Close price
volume (int/float)    # Bar volume
```

**File Location:**
- Local: `data/raw/{SYMBOL}_1m.parquet`
- Colab: `MyDrive/Research/data/raw/{SYMBOL}_1m.parquet`

### Notebook Configuration

```python
# Cell 1: Configuration
SYMBOL = "SI"                    # Contract symbol
TRAIN_XGBOOST = True            # Enable models
TRAIN_LIGHTGBM = True
TRAIN_CATBOOST = True
TRAIN_LSTM = False              # Enable if GPU available
RUN_CROSS_VALIDATION = False
TRAINING_HORIZON = 20           # Prediction horizon (bars)
```

### Custom Data File

```python
# Override default data file
CUSTOM_DATA_FILE = "SI_1m_historical.parquet"
```

---

## GPU Configuration

### GPU Recommendations by Model

| Model | Minimum GPU | Recommended | Training Time (T4) |
|-------|-------------|-------------|---------------------|
| XGBoost | None (CPU) | T4 | 1-2 min |
| LightGBM | None (CPU) | T4 | 30-60 sec |
| CatBoost | None (CPU) | T4 | 2-3 min |
| Random Forest | None (CPU) | None | 1-2 min |
| Logistic | None (CPU) | None | 10-20 sec |
| SVM | None (CPU) | None | 5-10 min |
| LSTM | T4 | A100 | 10-15 min (T4), 5-8 min (A100) |
| GRU | T4 | A100 | 8-12 min (T4), 4-6 min (A100) |
| TCN | T4 | A100 | 12-18 min (T4), 6-10 min (A100) |
| Transformer | T4 | A100 | 15-25 min (T4), 8-12 min (A100) |

### Memory Management

**If you get OOM (Out of Memory) errors:**

```python
# Reduce batch size
BATCH_SIZE = 128  # Default is 256

# Reduce sequence length (neural models)
SEQUENCE_LENGTH = 30  # Default is 60

# Use gradient accumulation
config = {
    "batch_size": 64,
    "gradient_accumulation_steps": 4  # Effective batch size: 256
}
```

---

## Common Issues

### Issue: GPU Not Detected

**Symptom:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Verify runtime type: Runtime -> Change runtime type -> GPU
2. Restart runtime: Runtime -> Restart runtime
3. Check quota (free tier: ~15-20 GPU hours/week)
4. Force reconnect: Runtime -> Disconnect and delete runtime

### Issue: Out of Memory (OOM)

**Symptom:** `CUDA out of memory` error

**Solutions:**
1. Reduce batch size: `BATCH_SIZE = 128` or `64`
2. Reduce sequence length: `SEQUENCE_LENGTH = 30`
3. Enable safe mode: `SAFE_MODE = True`
4. Disable GPU-heavy models: `TRAIN_LSTM = False`

### Issue: Session Timeout

**Symptom:** Runtime disconnects during long training

**Solutions:**
1. Use Colab Pro (24-hour timeout)
2. Add periodic outputs to show activity
3. Save checkpoints frequently
4. Split work across sessions

### Issue: Data File Not Found

**Symptom:** `[ERROR] No data file found for {SYMBOL}!`

**Solutions:**
1. Verify file exists: `print(os.listdir('data/raw'))`
2. Re-run Cell 2.1 to mount Drive
3. Upload data directly (see Direct Upload section)

### Issue: Package Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**
1. Install project in dev mode: `pip install -e .`
2. Add src to path: `sys.path.append('/content/Research/src')`
3. Verify current directory: `os.getcwd()`

---

## Performance Optimization

### Pre-process Data Once

Don't re-run Phase 1 unnecessarily:

```python
# First run: Process data
RUN_DATA_PIPELINE = True

# Subsequent runs: Skip Phase 1
RUN_DATA_PIPELINE = False
```

**Time saved:** 20-30 minutes per run

### Use Mixed Precision Training

Automatically enabled on A100/T4 for neural models:
- A100: bfloat16 (faster, better)
- T4: float16 (faster)

**Expected speedup:** 30-50% faster training

### Reduce CV Trials for Testing

```python
# Testing configuration (fast)
CV_N_SPLITS = 3  # Default: 5
CV_N_TRIALS = 10  # Default: 20

# Production configuration (thorough)
CV_N_SPLITS = 5
CV_N_TRIALS = 50
```

### Train Boosting Models First

Boosting models are fast and don't require GPU:

```python
# Quick iteration (5-10 min)
TRAIN_XGBOOST = True
TRAIN_LIGHTGBM = True
TRAIN_CATBOOST = True
TRAIN_LSTM = False  # Skip neural models
```

### Use Prescaled Data for CV

```python
CV_USE_PRESCALED = True  # Default, faster
```

**Time saved:** 20-30% faster CV

---

## Workflow Examples

### Workflow 1: Quick Boosting Comparison (30 min)

```python
# Cell 1: Configuration
SYMBOL = "SI"
TRAIN_XGBOOST = True
TRAIN_LIGHTGBM = True
TRAIN_CATBOOST = True
TRAIN_LSTM = False
RUN_CROSS_VALIDATION = False
TRAINING_HORIZON = 20

# Run All Cells -> Check results in Cell 4.2
```

### Workflow 2: Neural Model Training (60 min)

```python
# Enable GPU: Runtime -> Change runtime type -> GPU

# Cell 1: Configuration
TRAIN_LSTM = True
SEQUENCE_LENGTH = 60
BATCH_SIZE = 256
MAX_EPOCHS = 50

# Run Cells 2.1-2.3 (setup)
# Run Cells 3.1-3.3 (data pipeline)
# Run Cell 4.1 (train LSTM)
# Run Cell 4.3 (visualize learning curves)
```

### Workflow 3: Full Pipeline with Ensemble (2 hours)

```python
# Cell 1: Configuration
TRAIN_XGBOOST = True
TRAIN_LIGHTGBM = True
TRAIN_LSTM = True
TRAIN_STACKING = True
STACKING_BASE_MODELS = "xgboost,lightgbm,lstm"
TRAINING_HORIZON = 20

# Run All Cells
# Check ensemble results in Cell 6.1
# Export in Cell 7.2
```

### Workflow 4: Hyperparameter Tuning (3 hours)

```python
# Colab Pro recommended (background execution)

# Cell 1: Configuration
TRAIN_XGBOOST = True
RUN_CROSS_VALIDATION = True
CV_TUNE_HYPERPARAMS = True
CV_N_TRIALS = 100
CV_N_SPLITS = 5

# Run All Cells, close browser
# Check results in Cell 5.2 next morning
```

---

## Saving and Exporting Results

### Automatic Saving (Google Drive)

When running from Drive, outputs save automatically:
- Models: `experiments/runs/{run_id}/`
- Exports: `experiments/exports/{timestamp}_{symbol}_H{horizon}/`

### Download Results Locally

```python
from google.colab import files

# Download single file
files.download('experiments/runs/20251228_143052/model.pkl')

# Download entire export package
!zip -r results.zip experiments/exports/20251228_SI_H20/
files.download('results.zip')
```

---

## References

- **Architecture:** `docs/ARCHITECTURE.md`
- **Quick Reference:** `docs/QUICK_REFERENCE.md`
- **MTF Troubleshooting:** `docs/troubleshooting/MTF_TROUBLESHOOTING.md`
