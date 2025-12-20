# Phase 1 Pipeline Stages

Production-ready splitting, validation, backtesting, and reporting modules for the ensemble trading system.

## Modules

### Stage 7: Time-Based Splitting (`stage7_splits.py`)

Creates chronological train/val/test splits with purging and embargo to prevent label leakage.

**Features:**
- Chronological splitting (default 70/15/15)
- Purging: removes N bars at split boundaries
- Embargo: adds N bars buffer between splits
- Validates no overlap between splits
- Saves indices as .npy files
- Generates split metadata with date ranges

**Usage:**
```python
from stages.stage7_splits import create_splits

metadata = create_splits(
    data_path=Path("data/final/combined_final_labeled.parquet"),
    output_dir=Path("data/splits"),
    run_id="20240101_120000",  # Optional, auto-generated if None
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    purge_bars=60,  # = max_bars for H20 (prevents label leakage)
    embargo_bars=288
)
```

**Outputs:**
- `splits/{run_id}/train.npy` - Training set indices
- `splits/{run_id}/val.npy` - Validation set indices
- `splits/{run_id}/test.npy` - Test set indices
- `splits/{run_id}/split_config.json` - Metadata

---

### Stage 8: Comprehensive Validation (`stage8_validate.py`)

**Checks:** Data integrity, label sanity, feature quality

**Usage:**
```python
from stages.stage8_validate import validate_data

summary = validate_data(
    data_path=Path("data/final/combined_final_labeled.parquet"),
    output_path=Path("results/validation_report.json"),
    horizons=[1, 5, 20]
)
```

---

### Baseline Backtest (`baseline_backtest.py`)

Simple label-following strategy to verify labels have predictive signal.

**Usage:**
```python
from stages.baseline_backtest import run_baseline_backtest

results = run_baseline_backtest(
    data_path=Path("data/final/combined_final_labeled.parquet"),
    split_indices_path=Path("data/splits/test.npy"),
    output_dir=Path("results/baseline_backtest"),
    horizon=5
)
```

---

### Report Generation (`generate_report.py`)

Generates comprehensive Phase 1 summary with charts.

**Usage:**
```python
from stages.generate_report import generate_phase1_report

output_files = generate_phase1_report(
    data_path=Path("data/final/combined_final_labeled.parquet"),
    output_dir=Path("reports")
)
```

---

## Complete Pipeline

```bash
python src/run_phase1_complete.py
```
