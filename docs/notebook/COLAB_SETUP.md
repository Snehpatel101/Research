# Google Colab Setup Guide

Complete guide for running the ML Model Factory on Google Colab with GPU acceleration.

---

## Quick Start

1. **Open notebook in Colab:**
   - Click "Open in Colab" badge in notebook
   - Or upload `.ipynb` file to Colab

2. **Enable GPU runtime:** Runtime → Change runtime type → GPU (T4 or A100)

3. **Mount Google Drive:** Run Cell 2.1 to mount Drive and set up paths

4. **Run All Cells:** Runtime → Run all (Ctrl+F9 / Cmd+F9)

---

## GPU Configuration

### Enable GPU Runtime

**Steps:**
1. Click **Runtime** → **Change runtime type**
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

Expected output:
```
CUDA available: True
Device name: Tesla T4
Memory: 15.11 GB
```

### GPU Recommendations by Model

| Model | Minimum GPU | Recommended | Training Time (H20) |
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
| Voting | Inherited | Inherited | Sum of base models |
| Stacking | Inherited | Inherited | Sum + meta-learner (~5 min) |
| Blending | Inherited | Inherited | Sum + meta-learner (~3 min) |

**Notes:**
- Free tier: T4 GPU, 15 GB RAM, ~15-20 GPU hours per week
- Colab Pro: A100 GPU (priority), 32 GB RAM, ~50-100 GPU hours per month

---

## Data Setup

### Option 1: Mount Google Drive (Recommended)

**Upload data to Google Drive:**
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
print("✓ Data directory mounted successfully")
```

**Advantages:**
- Persistent storage (data survives session end)
- No re-upload needed
- Faster for large files

---

### Option 2: Upload Data Directly

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
    print(f'✓ Moved {filename} to data/raw/')
```

**Disadvantages:**
- Must re-upload after session timeout
- Slow for large files (> 100 MB)
- Not persistent

---

### Option 3: Download from Cloud Storage

**From Google Cloud Storage:**
```python
# Install gsutil
!pip install -q gsutil

# Download data
!gsutil -m cp -r gs://your-bucket/data ./data
```

**From AWS S3:**
```python
# Configure AWS credentials
!pip install -q awscli
!aws configure set aws_access_key_id YOUR_KEY
!aws configure set aws_secret_access_key YOUR_SECRET

# Download data
!aws s3 sync s3://your-bucket/data ./data
```

**From URL:**
```python
import urllib.request
import os

os.makedirs('data/raw', exist_ok=True)

url = "https://example.com/SI_1m.parquet"
output_path = "data/raw/SI_1m.parquet"

urllib.request.urlretrieve(url, output_path)
print(f"✓ Downloaded to {output_path}")
```

---

## Installation

### Automatic Installation (Recommended)

The notebook automatically installs dependencies in Cell 2.2:

```python
# Core dependencies
!pip install -q pandas numpy scikit-learn pyarrow

# ML libraries
!pip install -q xgboost lightgbm catboost

# PyTorch (for neural models)
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optimization & utilities
!pip install -q optuna ta pywavelets matplotlib seaborn tqdm numba psutil
```

**Expected runtime:** 1-2 minutes

---

### Manual Installation

If automatic installation fails:

```bash
# Create requirements.txt
cat > requirements.txt << EOF
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
torch>=2.0.0
optuna>=3.3.0
ta>=0.11.0
pywavelets>=1.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
pyarrow>=13.0.0
numba>=0.57.0
psutil>=5.9.0
EOF

# Install
!pip install -r requirements.txt
```

---

## Running the Notebook

### Full Pipeline (All Phases)

**Configuration:**
```python
# Cell 1: Configuration
SYMBOL = "SI"
TRAIN_XGBOOST = True
TRAIN_LIGHTGBM = True
TRAIN_LSTM = False  # Enable if GPU available
RUN_CROSS_VALIDATION = False
TRAINING_HORIZON = 20
```

**Execution:**
```
Runtime → Run all (Ctrl+F9)
```

**Expected runtime:**
- Phase 1 (data pipeline): 20-30 min
- Phase 2 (3 boosting models): 5-10 min
- Phase 2 (with neural models): +30-60 min
- Phase 3 (CV, optional): +30-90 min
- Phase 4 (ensemble, optional): +5-10 min
- **Total:** 25-40 min (boosting only), 90-150 min (all models)

---

### Quick Model Comparison (30 min)

**Goal:** Compare 3 boosting models

**Configuration:**
```python
TRAIN_XGBOOST = True
TRAIN_LIGHTGBM = True
TRAIN_CATBOOST = True
TRAIN_LSTM = False  # Disable neural models
RUN_CROSS_VALIDATION = False
```

**Run:**
1. Cell 1: Configuration
2. Cells 2.1-2.5: Environment setup
3. Cells 3.1-3.3: Data pipeline
4. Cells 4.1-4.2: Train and compare models

**Output:** Comparison table in Cell 4.2

---

### Neural Model Training (60 min)

**Goal:** Train LSTM with GPU

**Prerequisites:**
- GPU runtime enabled (T4 or A100)
- Data pipeline already complete

**Configuration:**
```python
TRAIN_LSTM = True
SEQUENCE_LENGTH = 60
BATCH_SIZE = 256
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
```

**Run:**
1. Skip Phase 1 if already done: `RUN_DATA_PIPELINE = False`
2. Run Cells 4.1-4.3

**Expected output:**
- Training progress with loss/accuracy per epoch
- Learning curves in Cell 4.3
- Final metrics in Cell 4.1

---

## Common Issues and Solutions

### Issue 1: GPU Not Detected

**Symptom:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. **Verify runtime type:**
   - Runtime → Change runtime type → GPU
2. **Restart runtime:**
   - Runtime → Restart runtime
3. **Check quota:**
   - Free tier: ~15-20 GPU hours per week
   - If exhausted, try later or upgrade to Colab Pro
4. **Force reconnect:**
   - Runtime → Disconnect and delete runtime
   - Runtime → Connect

---

### Issue 2: Out of Memory (OOM)

**Symptom:** `CUDA out of memory` error during training

**Solutions:**

**1. Reduce batch size:**
```python
# In Cell 1 configuration
BATCH_SIZE = 128  # Default is 256
# Or even lower:
BATCH_SIZE = 64
```

**2. Reduce sequence length (neural models):**
```python
SEQUENCE_LENGTH = 30  # Default is 60
TRANSFORMER_SEQUENCE_LENGTH = 64  # Default is 128
```

**3. Use gradient accumulation:**
```python
# Simulates larger batch size without memory overhead
config = {
    "batch_size": 64,
    "gradient_accumulation_steps": 4  # Effective batch size: 256
}
```

**4. Enable safe mode:**
```python
SAFE_MODE = True  # Clears memory aggressively
```

**5. Use CPU for that model:**
```python
TRAIN_LSTM = False  # Disable GPU-heavy model
```

---

### Issue 3: Session Timeout

**Symptom:** Runtime disconnects during long training

**Solutions:**

**1. Use Colab Pro:**
- Longer timeout limits (24 hours vs 12 hours)
- Background execution (runs after browser close)

**2. Add periodic outputs:**
```python
# Prevents timeout by showing activity
for epoch in range(epochs):
    # Training code
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
```

**3. Save checkpoints frequently:**
```python
# In training loop (auto-handled by trainer)
if epoch % 10 == 0:
    model.save(f"checkpoint_epoch_{epoch}.pkl")
```

**4. Split work across sessions:**
- Session 1: Run Phase 1 (data pipeline), save to Drive
- Session 2: Load processed data, train models
- Session 3: Run CV and ensembles

---

### Issue 4: Package Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**

**1. Install project in development mode:**
```python
!pip install -e .
```

**2. Add src to path:**
```python
import sys
sys.path.append('/content/Research/src')
```

**3. Verify current directory:**
```python
import os
print(f"Current directory: {os.getcwd()}")
# Should be: /content/Research or /content/drive/MyDrive/Research
```

**4. Re-run environment setup:**
- Run Cell 2.1 again

---

### Issue 5: Data File Not Found

**Symptom:** `[ERROR] No data file found for {SYMBOL}!`

**Solutions:**

**1. Verify file exists:**
```python
import os
print(os.listdir('data/raw'))
# Should show: SI_1m.parquet or similar
```

**2. Use custom data file:**
```python
# In Cell 1 configuration
CUSTOM_DATA_FILE = "SI_1m_historical.parquet"
```

**3. Check Drive mount:**
```python
# Re-run Cell 2.1 to mount Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

**4. Upload data directly:**
- See [Option 2: Upload Data Directly](#option-2-upload-data-directly)

---

## Performance Optimization

### 1. Use Mixed Precision Training

For neural models on A100/T4:

```python
# Automatically handled by trainer if GPU supports it
# A100: Supports bfloat16 (faster, better)
# T4: Supports float16 (faster, slightly less stable)
```

**Expected speedup:** 30-50% faster training

---

### 2. Pre-process Data Once

Don't re-run Phase 1 unnecessarily:

```python
# First run: Process data
RUN_DATA_PIPELINE = True
# Run all cells

# Subsequent runs: Skip Phase 1
RUN_DATA_PIPELINE = False
# Only run Phase 2+ cells
```

**Time saved:** 20-30 minutes per run

---

### 3. Use Prescaled Data for CV

```python
# In Cell 1 configuration
CV_USE_PRESCALED = True  # Default, faster
# vs
CV_USE_PRESCALED = False  # Per-fold scaling, more accurate but slower
```

**Time saved:** 20-30% faster CV

---

### 4. Reduce CV Trials for Testing

```python
# Testing configuration
CV_N_SPLITS = 3  # Default: 5
CV_N_TRIALS = 10  # Default: 20

# Production configuration
CV_N_SPLITS = 5
CV_N_TRIALS = 50
```

---

## Saving Results

### Save to Google Drive (Automatic)

When running from Drive, all outputs save automatically:
- Models: `experiments/runs/{run_id}/`
- Exports: `experiments/exports/{timestamp}_{symbol}_H{horizon}/`

**Persistence:** Survives session end

---

### Download Results Locally

**Download single file:**
```python
from google.colab import files
files.download('experiments/runs/20251228_143052/model.pkl')
```

**Download entire export package:**
```python
# Create ZIP
!zip -r results.zip experiments/exports/20251228_SI_H20/

# Download
from google.colab import files
files.download('results.zip')
```

---

### Export to Google Sheets (Metrics)

```python
import pandas as pd
from google.colab import auth
from gspread_dataframe import set_with_dataframe
import gspread

# Authenticate
auth.authenticate_user()
gc = gspread.oauth()

# Create comparison DataFrame
results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'Accuracy': [0.5234, 0.5189, 0.5267],
    'F1 (Macro)': [0.5187, 0.5142, 0.5221]
})

# Upload to Google Sheets
spreadsheet = gc.create('ML Pipeline Results')
worksheet = spreadsheet.sheet1
set_with_dataframe(worksheet, results_df)

print(f"Results: {spreadsheet.url}")
```

---

## Cost Optimization

### Free Tier Limits

- **GPU hours:** ~15-20 hours per week
- **RAM:** 12.7 GB
- **Disk:** 107 GB temporary (session-based)
- **Session timeout:** 12 hours max, 90 min idle
- **Background execution:** No

**Recommendations:**
1. Train boosting models first (fast, CPU-only)
2. Use GPU only for neural models
3. Skip CV for initial experiments
4. Use T4 GPU (A100 not available on free tier)

---

### Colab Pro ($9.99/month)

- **GPU hours:** ~50-100 hours per month
- **RAM:** Up to 32 GB (High-RAM runtime)
- **Disk:** Up to 200 GB
- **Session timeout:** 24 hours
- **Background execution:** Yes (runs after browser close)
- **Priority GPU access:** Yes (A100 available)

**Worth it if:**
- Training neural models frequently
- Running CV regularly (2-3 hours per run)
- Need longer session times
- Want faster GPUs (A100 vs T4)

---

### Cost-Saving Tips

1. **Train boosting models first (5-10 min, CPU):**
   ```python
   TRAIN_XGBOOST = True
   TRAIN_LIGHTGBM = True
   TRAIN_CATBOOST = True
   TRAIN_LSTM = False  # Skip GPU models for now
   ```

2. **Subsample data for testing:**
   ```python
   # In a test cell
   X_train = X_train[:10000]
   y_train = y_train[:10000]
   # Quick test run: 1-2 min instead of 10 min
   ```

3. **Run CV overnight (Colab Pro only):**
   - Enable background execution
   - Close browser
   - Check results in morning

4. **Use CPU for classical models:**
   - Random Forest, Logistic, SVM don't benefit from GPU

5. **Skip ONNX export (optional):**
   ```python
   EXPORT_ONNX = False  # Faster export
   ```

---

## Example Workflows

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

# Run All Cells
# Runtime → Run all

# Check results in Cell 4.2
```

---

### Workflow 2: Neural Model Training (60 min, GPU required)

```python
# Enable GPU: Runtime → Change runtime type → GPU

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

---

### Workflow 3: Full Pipeline with Ensemble (2 hours, GPU required)

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

---

### Workflow 4: Hyperparameter Tuning (3 hours, overnight)

**Colab Pro only (background execution):**

```python
# Cell 1: Configuration
TRAIN_XGBOOST = True
RUN_CROSS_VALIDATION = True
CV_TUNE_HYPERPARAMS = True
CV_N_TRIALS = 100  # Thorough search
CV_N_SPLITS = 5

# Run All Cells
# Close browser (Colab Pro keeps running)
# Check results in Cell 5.2 next morning
```

---

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] **GPU runtime enabled:** Runtime → Change runtime type → GPU
- [ ] **CUDA available:** `torch.cuda.is_available()` returns `True`
- [ ] **Data mounted/uploaded:** File exists in `data/raw/`
- [ ] **Packages installed:** Cell 2.2 ran without errors
- [ ] **Current directory correct:** `os.getcwd()` shows project root
- [ ] **No OOM errors:** Reduce batch size if needed
- [ ] **Session not timed out:** Re-run if disconnected

---

## Additional Resources

- **Colab Documentation:** https://colab.research.google.com/notebooks/intro.ipynb
- **PyTorch GPU Guide:** https://pytorch.org/docs/stable/notes/cuda.html
- **Optuna Documentation:** https://optuna.readthedocs.io/
- **Main README:** `../../README.md`
- **Configuration Guide:** [CONFIGURATION.md](CONFIGURATION.md)
- **Cell Reference:** [CELL_REFERENCE.md](CELL_REFERENCE.md)

---

**Last Updated:** 2025-12-28
