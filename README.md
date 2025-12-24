# ML Pipeline for OHLCV Time Series

Modular Phase 1 data pipeline that turns raw OHLCV bars into model-ready datasets with leakage-safe splits, scaling, and labeling. Phase 2+ (model factory, CV, ensembles) is planned but not implemented here.

```
[ Phase 1: Data ] → [ Phase 2: Models ] → [ Phase 3: CV ] → [ Phase 4: Ensemble ] → [ Phase 5: Prod ]
    IMPLEMENTED          PLANNED            PLANNED           PLANNED             FUTURE
```

## Quick Start

```bash
# Run Phase 1 pipeline with defaults (requires real data in data/raw/)
./pipeline run --symbols MES,MGC

# Check status
./pipeline status <run_id>
```

```python
from src.phase1.stages.datasets import TimeSeriesDataContainer

container = TimeSeriesDataContainer.from_parquet_dir(
    "data/splits/scaled",
    horizon=20,
)
X_train, y_train, w_train = container.get_sklearn_arrays("train")
```

## Pipeline Stages (Phase 1)

1) data_ingestion
2) data_cleaning
3) feature_engineering
4) initial_labeling
5) ga_optimize
6) final_labels
7) create_splits
7.5) feature_scaling
7.6) build_datasets
7.7) validate_scaled
8) validate
9) generate_report

## Key Outputs

- Labeled data: `data/final/{SYMBOL}_labeled.parquet`
- Combined labeled data: `data/final/combined_final_labeled.parquet`
- Scaled splits: `data/splits/scaled/train_scaled.parquet`, `val_scaled.parquet`, `test_scaled.parquet`
- Dataset manifests: `runs/<run_id>/artifacts/feature_set_manifest.json`, `dataset_manifest.json`
- Completion report: `results/PHASE1_COMPLETION_REPORT_<run_id>.md`

## Configuration Defaults (Phase 1)

- Horizons: `[5, 10, 15, 20]`
- Timeframe: `5min` (resampled from 1-min bars)
- Splits: `70/15/15` train/val/test
- Purge/embargo: auto-scaled from horizons (embargo defaults to 1440 bars unless overridden)

## Project Structure

```
.
├── src/
│   ├── cli/                 # Typer CLI entrypoints
│   ├── pipeline/            # Runner + stage registry
│   ├── phase1/              # Phase 1 logic, configs, and stages
│   └── common/              # Shared utilities (manifest, horizon config)
├── data/                    # Raw/clean/features/final/splits
├── runs/                    # Per-run configs/logs/artifacts
├── results/                 # Reports and plots
├── docs/                    # Docs (non-phase + phase specs)
└── tests/                   # Phase 1 tests and fixtures
```

## Documentation

- `docs/README.md`
- `docs/getting-started/QUICKSTART.md`
- `docs/getting-started/PIPELINE_CLI.md`
- `docs/reference/ARCHITECTURE.md`
- `docs/reference/FEATURES.md`

## Engineering Principles

1) Modularity: small, composable modules
2) Fail fast: validate at boundaries
3) No leakage: purge/embargo + train-only scaling
4) Keep it simple: remove unused code

---

Phase 1 is implemented; Phase 2+ is planned.
