# Phase 1 Architecture (Data Pipeline)

## Overview
Phase 1 is a modular, stage-based pipeline that converts raw OHLCV bars into model-ready datasets with triple-barrier labels, leakage-safe splits, and train-only scaling.

Key goals:
- Clear stage boundaries with explicit artifacts
- No lookahead leakage (purge, embargo, train-only scaling)
- Configurable horizons and feature sets

## Code Layout

```
src/
├── cli/                 # Typer CLI
├── pipeline/            # Runner + stage registry + wrappers
├── phase1/
│   ├── config/          # Feature sets, labeling config, barriers
│   ├── stages/          # Ingest/clean/features/labeling/splits/scaling/validation
│   └── utils/           # Feature set resolution, selection helpers
└── common/              # Artifact manifest + horizon utilities
```

## Stage Flow

1) data_ingestion - Load and validate raw OHLCV data
2) data_cleaning - Resample 1min to 5min, handle gaps
3) feature_engineering - Generate 150+ features (momentum, wavelets, microstructure)
4) initial_labeling - Triple-barrier labels with default parameters
5) ga_optimize - Optuna-based parameter optimization
6) final_labels - Apply optimized barrier parameters
7) create_splits - Train/val/test with purge (60) and embargo (1440)
7.5) feature_scaling - Robust scaling fitted on train only
7.6) build_datasets - Build TimeSeriesDataContainer
7.7) validate_scaled - Verify scaled data quality
8) validate - Feature correlation and quality checks
9) generate_report - Produce completion report

The stage list is defined in `src/pipeline/stage_registry.py` and executed by `src/pipeline/runner.py`.

## Artifacts and Outputs

```
data/
├── raw/                 # Input 1-min OHLCV
├── clean/               # Resampled/cleaned data
├── features/            # Feature-enriched data
├── final/               # Labeled data
└── splits/
    ├── train_indices.npy
    ├── val_indices.npy
    ├── test_indices.npy
    └── scaled/
        ├── train_scaled.parquet
        ├── val_scaled.parquet
        └── test_scaled.parquet

runs/<run_id>/
├── config/config.json
├── logs/pipeline.log
└── artifacts/
    ├── manifest.json
    ├── pipeline_state.json
    ├── feature_set_manifest.json
    └── dataset_manifest.json

results/
└── PHASE1_COMPLETION_REPORT_<run_id>.md
```

## Leakage Prevention

- Purge/embargo at split boundaries
- Invalid labels for last `max_bars` samples
- Train-only scaling for all features
- MTF features shift by one higher-timeframe bar

## Configuration

- Primary config: `src/phase1/pipeline_config.py`
- Barrier params: `src/phase1/config/barriers_config.py`
- Feature sets: `src/phase1/config/feature_sets.py`

## Notes

- Horizons are configurable (default `[5, 10, 15, 20]`).
- Cross-asset features are implemented but disabled by default.
- Phase 2+ (models, CV, ensembles) is planned, not implemented here.
