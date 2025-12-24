# Phase 1: Data Preparation Pipeline

**Status:** COMPLETE (9.5/10)

Phase 1 transforms raw OHLCV data into model-ready datasets for any ML model type.

## What It Does

```
Raw OHLCV (1-min bars)
    ↓ Stage 1: Ingest
    ↓ Stage 2: Clean & Resample (5-min)
    ↓ Stage 3: Feature Engineering (107 features)
    ↓ Stage 4-6: Triple-Barrier Labeling + GA Optimization
    ↓ Stage 7: Train/Val/Test Splits (70/15/15)
    ↓ Stage 7.5: Feature Scaling (train-only fit)
    ↓ Stage 8: Validation
Model-Ready Parquet Files
```

## Quick Start

```bash
# Run full pipeline
./pipeline run --symbols MES --stages all

# Output location
ls data/splits/scaled/
# train_scaled.parquet  (41K rows × 129 cols)
# val_scaled.parquet    (7K rows × 129 cols)
# test_scaled.parquet   (7K rows × 129 cols)
```

## Phase 2 Integration

```python
from src.phase1.stages.datasets import TimeSeriesDataContainer

# Load Phase 1 outputs
container = TimeSeriesDataContainer.from_parquet_dir(
    'data/splits/scaled', horizon=20
)

# Get data in any format
X, y, w = container.get_sklearn_arrays('train')      # XGBoost/LightGBM
dataset = container.get_pytorch_sequences('train')   # LSTM/TCN
nf_df = container.get_neuralforecast_df('train')     # N-HiTS/TFT
```

## Directory Structure

```
src/phase1/
├── stages/           # 15 pipeline stages
│   ├── ingest/       # Data ingestion
│   ├── clean/        # Cleaning & resampling
│   ├── features/     # Feature engineering
│   ├── labeling/     # Triple-barrier labels
│   ├── ga_optimize/  # GA barrier optimization
│   ├── final_labels/ # Apply optimized labels
│   ├── splits/       # Train/val/test splits
│   ├── scaling/      # Feature scaling
│   ├── datasets/     # Dataset builder
│   ├── validation/   # Data validation
│   └── reporting/    # Completion reports
├── config/           # Phase 1 config
│   ├── barriers_config.py
│   ├── labeling_config.py
│   └── feature_sets.py
├── utils/            # Phase 1 utilities
├── pipeline_config.py
└── presets.py
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| PURGE_BARS | 60 | Prevent label leakage |
| EMBARGO_BARS | 288 | ~1 day buffer |
| HORIZONS | [5, 10, 15, 20] | Multi-horizon labels |
| TRAIN/VAL/TEST | 70/15/15 | Split ratios |

## Documentation

- [PHASE_1.md](../docs/phases/PHASE_1.md) - Full specification
- [QUICKSTART.md](../docs/getting-started/QUICKSTART.md) - 15-min guide
- [ARCHITECTURE.md](../docs/reference/ARCHITECTURE.md) - System design
