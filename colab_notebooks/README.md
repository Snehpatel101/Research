# Google Colab Training Guide

## Overview

This directory contains Jupyter notebooks for training the ML factory on Google Colab. Colab provides free GPU access but has ephemeral runtimes (12-hour limit, frequent disconnections).

**Key Features:**
- Auto-checkpoint to Google Drive every 30 minutes
- W&B experiment tracking (cloud-based, permanent)
- Resume from checkpoint after disconnections
- Phase-based workflow (run data pipeline once, train multiple models)

---

## Quick Start

### 1. Upload Data to Google Drive

Create folder structure in Google Drive:

```
MyDrive/
└── ml_factory/
    ├── data/
    │   └── raw/
    │       ├── MES_1m.parquet  # Upload your OHLCV data here
    │       └── MGC_1m.parquet
    ├── models/          # Trained models will be saved here
    ├── checkpoints/     # Auto-save checkpoints
    └── results/         # Experiment results
```

### 2. Open Notebooks in Colab

**Option A: Upload notebooks to Google Drive**
1. Upload all notebooks from `colab_notebooks/` to Drive
2. Right-click notebook → Open with → Google Colab

**Option B: Open from GitHub**
1. Go to https://colab.research.google.com/
2. Select "GitHub" tab
3. Enter: `https://github.com/yourusername/ml-factory`
4. Select notebook

### 3. Run Notebooks in Sequence

#### Step 1: Setup Environment (5 min)

Open `00_setup.ipynb` and run all cells. This will:
- Mount Google Drive
- Clone repository
- Install dependencies
- Authenticate W&B
- Detect GPU

#### Step 2: Run Data Pipeline (30-60 min, run once)

Open `01_data_pipeline.ipynb` and run all cells. This will:
- Load raw OHLCV from Drive
- Run phases 1-5 (MTF, features, labeling)
- Save processed datasets to Drive
- **Important:** Only run this once per symbol

#### Step 3: Train Models (20-120 min each)

**Tabular models (fastest):**
- Open `02_train_tabular.ipynb`
- Set `MODEL = "xgboost"` (or `"lightgbm"`, `"catboost"`)
- Run all cells
- Repeat for each model

**Sequence models (medium):**
- Open `03_train_sequence.ipynb`
- Set `MODEL = "lstm"` (or `"gru"`, `"tcn"`)
- Run all cells

**Advanced models (slowest):**
- Open `04_train_advanced.ipynb`
- Set `MODEL = "patchtst"` (or `"tft"`, `"itransformer"`)
- Run all cells

#### Step 4: Train Ensemble (30-60 min)

Open `05_train_ensemble.ipynb` and run all cells. This will:
- Load base models from W&B
- Generate OOF predictions
- Train stacking meta-learner
- Save ensemble to Drive + W&B

---

## Notebook Reference

| Notebook | Purpose | Runtime | GPU Required |
|----------|---------|---------|--------------|
| `00_setup.ipynb` | Environment setup | 5 min | No |
| `01_data_pipeline.ipynb` | Phases 1-5 (run once) | 30-60 min | No |
| `02_train_tabular.ipynb` | Train XGBoost/LightGBM/CatBoost | 20-40 min | Recommended |
| `03_train_sequence.ipynb` | Train LSTM/GRU/TCN | 40-90 min | Yes |
| `04_train_advanced.ipynb` | Train PatchTST/TFT/iTransformer | 60-120 min | Yes |
| `05_train_ensemble.ipynb` | Heterogeneous stacking | 30-60 min | No |
| `06_cross_validation.ipynb` | 5-fold CV evaluation | 2-4 hours | Yes |
| `07_inference.ipynb` | Batch predictions | 10-20 min | No |

---

## Handling Disconnections

### What Happens on Disconnect?

**Lost:**
- Local Colab runtime (/content/)
- Running processes
- Installed packages

**Preserved:**
- Google Drive files
- W&B experiments
- Checkpoints (auto-saved every 30 min)

### How to Resume After Disconnect

1. **Reconnect to runtime**
   - Click "Reconnect" in Colab

2. **Re-run setup cell**
   ```python
   from utils.colab_setup import setup_colab_environment
   env_info = setup_colab_environment(...)
   ```

3. **Load latest checkpoint**
   ```python
   checkpoint = ckpt_mgr.load_latest_checkpoint(phase="train_xgboost")
   if checkpoint:
       model.load_state(checkpoint['state']['model_state'])
       start_epoch = checkpoint['state']['epoch']
   ```

4. **Resume training**
   - Training will continue from last saved epoch
   - No progress lost (thanks to checkpointing!)

---

## Best Practices

### 1. Enable GPU Before Training

1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. GPU type → T4 (free) or V100 (Colab Pro)

### 2. Monitor Session Time

```python
from utils.colab_setup import estimate_training_time_remaining

remaining = estimate_training_time_remaining()
print(f"Time remaining: {remaining:.1f} hours")

if remaining < 2.0:
    print("⚠️ Less than 2 hours remaining - save checkpoint now!")
```

### 3. Use Local Disk for Fast I/O

**Bad (slow):** Read data from Drive every epoch
```python
# ❌ Slow - reads from Drive every time
df = pd.read_parquet("/content/drive/MyDrive/data/MES_1m.parquet")
```

**Good (fast):** Copy to local disk once
```python
# ✅ Fast - copy once, read from local disk
import shutil
shutil.copy(
    "/content/drive/MyDrive/data/MES_1m.parquet",
    "/content/data/MES_1m.parquet"
)
df = pd.read_parquet("/content/data/MES_1m.parquet")
```

### 4. Clean Up Memory

```python
import gc
import torch

# Delete unused variables
del large_dataframe
del old_model

# Clear GPU cache (PyTorch)
torch.cuda.empty_cache()

# Force garbage collection
gc.collect()
```

### 5. Use Mixed Precision Training

```python
# PyTorch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Troubleshooting

### Problem: "Runtime disconnected"

**Cause:** Idle for too long, or 12-hour limit reached

**Solution:**
1. Reconnect to runtime
2. Re-run setup cell
3. Load checkpoint and resume

### Problem: "Out of memory (OOM)"

**Cause:** Model too large for GPU

**Solutions:**
1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision training
4. Use gradient checkpointing

### Problem: "Checkpoint not found"

**Cause:** Drive not mounted or checkpoint path incorrect

**Solutions:**
1. Check Drive is mounted: `ls /content/drive/MyDrive`
2. Verify checkpoint path in `CheckpointManager`
3. Check W&B for artifacts: `wandb.use_artifact(...)`

### Problem: "Training too slow"

**Cause:** Reading from Drive, no GPU, inefficient code

**Solutions:**
1. Copy data to local disk first
2. Enable GPU (Runtime → Change runtime type)
3. Use batch processing
4. Profile code with `cProfile`

---

## Resource Limits

### Colab Free Tier

- **Session limit:** ~12 hours (can disconnect earlier if idle)
- **GPU:** T4/P100/V100 (inconsistent, not guaranteed)
- **RAM:** ~12-13 GB
- **Disk:** ~78 GB (ephemeral)
- **Compute units:** Limited per day (no hard quota published)

### Colab Pro ($10/month)

- **Session limit:** ~24 hours
- **GPU:** V100/A100 (higher priority)
- **RAM:** ~26 GB (High-RAM option: 52 GB)
- **Disk:** ~166 GB
- **Compute units:** Higher quota

### Google Drive Storage

- **Free:** 15 GB (shared with Gmail, Photos)
- **Paid (Google One):**
  - 100 GB: $2/month
  - 200 GB: $3/month
  - 2 TB: $10/month

---

## Advanced Usage

### Multi-Session Parallel Training

Train multiple models in parallel using separate Colab sessions:

**Session 1:** Train XGBoost
```python
MODEL = "xgboost"
# ... train ...
```

**Session 2:** Train LSTM
```python
MODEL = "lstm"
# ... train ...
```

**Session 3:** Train PatchTST
```python
MODEL = "patchtst"
# ... train ...
```

All sessions upload to same W&B project → Combine later for ensemble

### Programmatic Notebook Execution

Use `papermill` to run notebooks from CLI:

```bash
# Install papermill
pip install papermill

# Execute notebook with parameters
papermill 02_train_tabular.ipynb output.ipynb \
  -p SYMBOL "MES" \
  -p HORIZON 20 \
  -p MODEL "xgboost"
```

### Custom Checkpoint Logic

```python
class CustomCheckpoint:
    def __call__(self, epoch, metrics):
        # Save only if val_loss improved
        if metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            ckpt_mgr.save_checkpoint(
                phase="train_xgboost",
                state={"epoch": epoch, "model": model.get_state()},
                force=True,
            )
```

---

## FAQ

**Q: Do I need to re-run the data pipeline for every model?**

A: No! Run `01_data_pipeline.ipynb` once per symbol, then use the processed datasets for all models.

**Q: How do I switch between symbols (MES vs MGC)?**

A: Change `SYMBOL = "MES"` to `SYMBOL = "MGC"` in each notebook. Processed datasets are stored separately per symbol.

**Q: Can I use Colab for production inference?**

A: No, Colab is for training only. For production, export models to FastAPI/Cloud Run/AWS Lambda.

**Q: What if I run out of Drive storage?**

A: Use Google Cloud Storage (GCS) instead - faster and cheaper for large datasets. Update DVC config to use GCS remote.

**Q: How do I delete old checkpoints?**

A: Checkpoints auto-cleanup (keeps last 3). To manually delete: `rm -rf /content/drive/MyDrive/ml_factory/checkpoints/*`

---

## Next Steps

1. Run `01_data_pipeline.ipynb` to create processed datasets
2. Train baseline models (XGBoost, LSTM)
3. Compare results in W&B dashboard
4. Train heterogeneous ensemble
5. Export best model for production

**W&B Dashboard:** https://wandb.ai/yourusername/ohlcv-ml-factory

**GitHub Repository:** https://github.com/yourusername/ml-factory
