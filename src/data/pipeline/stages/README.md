# Pipeline Stages

Production-ready data preparation pipeline for the ML Factory ensemble trading system.

## Stage Architecture

The pipeline is organized into subdirectories, each containing a `run.py` entry point:

```
stages/
├── ingest/          Stage 1: Data Ingestion (run_data_generation)
├── clean/           Stage 2: Data Cleaning (run_data_cleaning)
├── features/        Stage 3: Feature Engineering (run_feature_engineering)
├── labeling/        Stage 4: Initial Labeling (run_initial_labeling)
├── ga_optimize/     Stage 5: GA Optimization (run_ga_optimization)
├── final_labels/    Stage 6: Final Labels (run_final_labels)
├── splits/          Stage 7: Create Splits (run_create_splits)
├── scaling/         Stage 7.5: Feature Scaling (run_feature_scaling)
├── datasets/        Stage 7.6: Build Datasets (run_build_datasets)
├── scaled_validation/ Stage 7.7: Post-Scale Validation (run_scaled_validation)
├── validation/      Stage 8: Comprehensive Validation (run_validation)
├── reporting/       Stage 9: Generate Report (run_generate_report)
└── evaluation/      Stage 10: Post-Training Evaluation (run_evaluation)
```

## Stage Details

### Stage 1: Data Ingestion (`ingest/run.py`)
Generates or validates raw OHLCV data files.

### Stage 2: Data Cleaning (`clean/run.py`)
Cleans and resamples OHLCV data, handling gaps and anomalies.

### Stage 3: Feature Engineering (`features/run.py`)
Generates ~180 technical indicators and derived features.

**Features:**
- Momentum indicators (RSI, MACD, etc.)
- Volatility features (ATR, Bollinger Bands)
- Volume features (OBV, VWAP)
- Multi-timeframe (MTF) features
- Wavelet and entropy features

### Stage 4: Initial Labeling (`labeling/run.py`)
Applies triple-barrier labeling for multi-class targets.

### Stage 5: GA Optimization (`ga_optimize/run.py`)
Optimizes barrier parameters using genetic algorithms.

### Stage 6: Final Labels (`final_labels/run.py`)
Applies optimized labels with quality scores.

### Stage 7: Create Splits (`splits/run.py`)
Creates chronological train/val/test splits with purging and embargo.

**Features:**
- Chronological splitting (default 70/15/15)
- Purging: removes N bars at split boundaries
- Embargo: adds N bars buffer between splits
- Validates no overlap between splits

### Stage 7.5: Feature Scaling (`scaling/run.py`)
Train-only feature scaling to prevent leakage.

### Stage 7.6: Build Datasets (`datasets/run.py`)
Builds dataset splits and manifests.

### Stage 7.7: Post-Scale Validation (`scaled_validation/run.py`)
Validates scaled data for drift and distribution issues.

### Stage 8: Comprehensive Validation (`validation/run.py`)
Final data integrity, label sanity, and feature quality checks.

### Stage 9: Generate Report (`reporting/run.py`)
Generates comprehensive Phase 1 summary with charts.

### Stage 10: Post-Training Evaluation (`evaluation/run.py`)
Post-training model evaluation (optional, runs after model training).

## Usage

The pipeline is orchestrated by `PipelineRunner`:

```python
from src.data.pipeline.runner import PipelineRunner
from src.data.pipeline.data_config import DataConfig

config = DataConfig(
    symbols=["MES"],
    target_timeframe="5min",
    project_root=Path("/path/to/project"),
)

runner = PipelineRunner(config)
success = runner.run()
```

## Stage Registration

Stages are registered in `stage_registry.py`. Each stage definition includes:
- Name (from `StageName` enum)
- Dependencies
- Description
- Required flag
- Stage number (for ordering)

## Configuration

Stage behavior is controlled via `DataConfig`:
- `stage_timeout_seconds`: Maximum execution time per stage
- `enable_stage_timeouts`: Enable/disable timeout enforcement
- `stage3_fail_on_partial`: Fail Stage 3 if tasks fail
- `stage3_min_success_rate`: Minimum success rate for Stage 3
- `enable_transition_validation`: Validate data between stages
