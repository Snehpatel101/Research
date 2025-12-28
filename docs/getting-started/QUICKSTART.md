# Quickstart: Phase 1 Data Pipeline

## What this project does
Phase 1 turns raw OHLCV bars into model-ready datasets with features, triple-barrier labels, leakage-safe splits, and train-only scaling.

## Prerequisites

```bash
python --version
# Phase 1 (data pipeline) dependencies only
pip install -r requirements.txt

# If you also want Phase 2+ (training/CV/ensembles), install the full package deps
pip install -e .
```

## Run your first pipeline

Requires real OHLCV data in `data/raw/` (e.g., `MES_1m.parquet`).

```bash
# Defaults (MES, MGC, 5min, horizons 5/10/15/20)
./pipeline run

# Custom symbols and date range
./pipeline run --symbols MES,MGC --start 2020-01-01 --end 2024-12-31

# Custom run ID for tracking
./pipeline run --run-id my_experiment_v1
```

## Check progress

```bash
./pipeline status <run_id>

# Logs
 tail -f runs/<run_id>/logs/pipeline.log
```

## Expected outputs

```
data/final/
├── MES_labeled.parquet
└── MGC_labeled.parquet

data/splits/scaled/
├── train_scaled.parquet
├── val_scaled.parquet
└── test_scaled.parquet

results/
└── PHASE1_COMPLETION_REPORT_<run_id>.md

runs/<run_id>/
├── config/config.json
├── logs/pipeline.log
└── artifacts/manifest.json
```

## Use the outputs in Python

```python
from src.phase1.stages.datasets import TimeSeriesDataContainer

container = TimeSeriesDataContainer.from_parquet_dir(
    "data/splits/scaled",
    horizon=20,
)
X_train, y_train, w_train = container.get_sklearn_arrays("train")
```

## Next

- CLI reference: `docs/getting-started/PIPELINE_CLI.md`
- Architecture: `docs/reference/ARCHITECTURE.md`
- Feature catalog: `docs/reference/FEATURES.md`
