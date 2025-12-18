# Phase 1 Pipeline CLI Guide

## Overview

This comprehensive configuration system and CLI provides complete control over the Phase 1 data preparation pipeline for the Ensemble Price Prediction System.

## Architecture

The system consists of four main components:

### 1. **pipeline_config.py** - Configuration Management
- `PipelineConfig` dataclass with all pipeline settings
- Automatic run_id generation (YYYYMMDD_HHMMSS format)
- Configuration validation and persistence
- Support for symbols, date ranges, feature sets, label horizons
- GA settings for future Phase 2
- Split ratios, purge/embargo bars
- `save_config()` and `load_config()` methods

### 2. **pipeline_runner.py** - Pipeline Orchestrator
- Executes pipeline stages in correct order
- Dependency tracking between stages
- Artifact tracking and management
- Resume from failed stage capability
- Parallel execution where possible
- Structured logging to `logs/{run_id}/`

### 3. **manifest.py** - Data Versioning
- Computes checksums for all artifacts
- Tracks changes between pipeline runs
- Verifies artifact integrity
- Generates manifest.json for each run
- Enables comparison between runs

### 4. **pipeline_cli.py** - Command-Line Interface
- Typer-based CLI with rich terminal output
- Colored output and progress indicators
- User-friendly commands for all pipeline operations

## Installation

Install the required dependencies:

```bash
cd /home/user/Research
pip install -r requirements-cli.txt
```

Or install manually:

```bash
pip install typer rich
```

## Usage

### Method 1: Using the wrapper script

```bash
./pipeline <command> [options]
```

### Method 2: Using Python directly

```bash
python3 src/pipeline_cli.py <command> [options]
```

## Commands

### 1. Run Pipeline

Execute the complete Phase 1 pipeline with custom parameters:

```bash
# Basic run with defaults
./pipeline run

# Run with specific symbols and date range
./pipeline run --symbols MES,MGC --start 2020-01-01 --end 2024-12-31

# Run with custom run ID
./pipeline run --run-id phase1_v1 --description "Initial baseline run"

# Run with multiple symbols and custom horizons
./pipeline run --symbols MES,MGC,MNQ --horizons 1,5,10,20

# Generate synthetic data
./pipeline run --synthetic

# Custom split ratios
./pipeline run --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2

# Custom purge and embargo settings
./pipeline run --purge-bars 30 --embargo-bars 576
```

**Options:**
- `--symbols, -s`: Comma-separated list of symbols (default: MES,MGC)
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format
- `--run-id`: Custom run identifier (auto-generated if not provided)
- `--description, -d`: Description of this run
- `--train-ratio`: Training set ratio (default: 0.70)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--purge-bars`: Bars to purge at split boundaries (default: 20)
- `--embargo-bars`: Embargo period in bars (default: 288)
- `--horizons`: Comma-separated label horizons (default: 1,5,20)
- `--synthetic`: Generate synthetic data instead of using real data
- `--project-root`: Project root directory (default: /home/user/Research)

### 2. Resume Pipeline

Resume a pipeline run from a specific stage:

```bash
# Resume from labeling stage
./pipeline rerun 20241218_120000 --from labeling

# Resume from splits creation
./pipeline rerun phase1_v1 --from create_splits

# Resume from last successful stage (auto-detect)
./pipeline rerun 20241218_120000
```

**Stage names:**
- `data_generation` or `data` - Data generation/acquisition
- `data_cleaning` or `cleaning` or `clean` - Data cleaning
- `feature_engineering` or `features` - Feature engineering
- `labeling` or `labels` - Triple-barrier labeling
- `create_splits` or `splits` - Create train/val/test splits
- `generate_report` or `report` - Generate completion report

### 3. Check Status

Check the status of a pipeline run:

```bash
# Basic status
./pipeline status 20241218_120000

# Detailed status with all artifacts
./pipeline status phase1_v1 --verbose
```

Shows:
- Run configuration (symbols, date range, etc.)
- Pipeline stage status (completed, failed, pending)
- Execution time per stage
- Artifact count and total size
- Overall progress percentage

### 4. Validate Configuration

Validate pipeline configuration and data integrity:

```bash
# Validate new configuration
./pipeline validate --symbols MES,MGC,MNQ

# Validate existing run
./pipeline validate --run-id 20241218_120000
```

Checks:
- Configuration parameter validity
- Split ratios sum to 1.0
- Date format correctness
- Label horizon values
- Barrier parameters
- GA parameters
- Artifact checksums (for existing runs)

### 5. List Runs

List all pipeline runs:

```bash
# List 10 most recent runs
./pipeline list-runs

# List 20 most recent runs
./pipeline list-runs --limit 20
```

Shows:
- Run ID
- Description
- Symbols
- Completion status
- Creation timestamp

### 6. Compare Runs

Compare artifacts between two runs:

```bash
./pipeline compare 20241218_120000 20241218_130000
```

Shows:
- Added artifacts (in run2 but not run1)
- Removed artifacts (in run1 but not run2)
- Modified artifacts (different checksums)
- Unchanged artifacts

Useful for:
- Understanding what changed between runs
- Debugging pipeline modifications
- Tracking data lineage

### 7. Clean Runs

Delete a pipeline run and all its artifacts:

```bash
# With confirmation prompt
./pipeline clean 20241218_120000

# Skip confirmation (use with caution!)
./pipeline clean phase1_v1 --force
```

## Pipeline Stages

The Phase 1 pipeline consists of 6 stages executed in order:

### Stage 1: Data Generation
- Generates synthetic data or validates existing raw data
- Creates 1-minute OHLCV bars for each symbol
- Output: `data/raw/{symbol}_1m.parquet`

### Stage 2: Data Cleaning
- Resamples 1-min bars to 5-min bars
- Handles missing data and outliers
- Output: `data/clean/{symbol}_5m_clean.parquet`

### Stage 3: Feature Engineering
- Generates 50+ technical indicators
- Creates price, momentum, volatility, volume features
- Output: `data/features/{symbol}_5m_features.parquet`

### Stage 4: Labeling
- Applies triple-barrier labeling method
- Computes labels for multiple horizons (1, 5, 20 bars)
- Calculates sample weights based on quality
- Output: `data/final/{symbol}_labeled.parquet`

### Stage 5: Create Splits
- Combines data from all symbols
- Creates train/val/test splits with purging and embargo
- Saves split indices
- Output: `data/splits/train_indices.npy`, `val_indices.npy`, `test_indices.npy`

### Stage 6: Generate Report
- Creates comprehensive completion report
- Includes dataset statistics, label distributions, split info
- Output: `results/PHASE1_COMPLETION_REPORT_{run_id}.md`

## Directory Structure

After running the pipeline, the following structure is created:

```
/home/user/Research/
├── data/
│   ├── raw/              # Raw 1-minute data
│   ├── clean/            # Cleaned 5-minute data
│   ├── features/         # Data with technical features
│   ├── final/            # Labeled data
│   └── splits/           # Train/val/test indices
├── runs/
│   └── {run_id}/
│       ├── config/
│       │   └── config.json           # Pipeline configuration
│       ├── logs/
│       │   └── pipeline.log          # Execution logs
│       └── artifacts/
│           ├── manifest.json         # Artifact manifest
│           └── pipeline_state.json   # Pipeline state
├── results/
│   └── PHASE1_COMPLETION_REPORT_{run_id}.md
└── src/
    ├── pipeline_config.py
    ├── pipeline_runner.py
    ├── pipeline_cli.py
    └── manifest.py
```

## Configuration System

### Creating a Configuration

```python
from pipeline_config import PipelineConfig, create_default_config

# Method 1: Use defaults
config = create_default_config(
    symbols=['MES', 'MGC'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    description='Baseline run'
)

# Method 2: Full customization
config = PipelineConfig(
    run_id='custom_run_v1',
    symbols=['MES', 'MGC', 'MNQ'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    label_horizons=[1, 5, 10, 20],
    train_ratio=0.60,
    val_ratio=0.20,
    test_ratio=0.20,
    purge_bars=30,
    embargo_bars=576,
    barrier_k_up=2.5,
    barrier_k_down=2.5,
    max_bars_ahead=100
)
```

### Saving and Loading Configuration

```python
# Save configuration
config.save_config()  # Saves to run_dir/config/config.json

# Or save to custom path
config.save_config(Path('/path/to/config.json'))

# Load configuration
config = PipelineConfig.load_config(Path('/path/to/config.json'))

# Load by run ID
config = PipelineConfig.load_from_run_id('20241218_120000')
```

### Validating Configuration

```python
# Validate and get issues
issues = config.validate()

if issues:
    print("Validation failed:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration is valid!")

# Print summary
print(config.summary())
```

## Pipeline Runner

### Basic Usage

```python
from pipeline_config import create_default_config
from pipeline_runner import PipelineRunner

# Create configuration
config = create_default_config(symbols=['MES', 'MGC'])

# Create runner
runner = PipelineRunner(config)

# Run complete pipeline
success = runner.run()

if success:
    print("Pipeline completed!")
else:
    print("Pipeline failed. Check logs.")
```

### Resume from Stage

```python
# Resume from specific stage
runner = PipelineRunner(config, resume=True)
success = runner.run(from_stage='labeling')
```

### Access Results

```python
# After running, access stage results
for stage_name, result in runner.stage_results.items():
    print(f"{stage_name}: {result.status.value}")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Artifacts: {len(result.artifacts)}")
```

## Manifest System

### Creating a Manifest

```python
from manifest import ArtifactManifest
from pathlib import Path

# Create manifest
manifest = ArtifactManifest(
    run_id='20241218_120000',
    project_root=Path('/home/user/Research')
)

# Add artifacts
manifest.add_artifact(
    name='clean_data_MES',
    file_path=Path('/home/user/Research/data/clean/MES_5m_clean.parquet'),
    stage='data_cleaning',
    metadata={'symbol': 'MES', 'rows': 100000}
)

# Save manifest
manifest.save()
```

### Verifying Artifacts

```python
# Verify single artifact
is_valid = manifest.verify_artifact('clean_data_MES')

# Verify all artifacts
verification = manifest.verify_all_artifacts()
print(f"Valid: {sum(verification.values())}/{len(verification)}")
```

### Comparing Runs

```python
from manifest import compare_runs

# Compare two runs
comparison = compare_runs('20241218_120000', '20241218_130000')

print(f"Added: {len(comparison['added'])}")
print(f"Removed: {len(comparison['removed'])}")
print(f"Modified: {len(comparison['modified'])}")
print(f"Unchanged: {len(comparison['unchanged'])}")
```

## Advanced Examples

### Custom Pipeline with GA Settings

```python
config = PipelineConfig(
    run_id='ga_experiment_v1',
    symbols=['MES', 'MGC'],
    description='GA parameter sweep',
    # Data settings
    label_horizons=[5, 10],
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    # GA settings (for Phase 2)
    ga_population_size=100,
    ga_generations=200,
    ga_crossover_rate=0.9,
    ga_mutation_rate=0.15,
    ga_elite_size=10
)

runner = PipelineRunner(config)
runner.run()
```

### Multiple Runs for Comparison

```bash
# Run 1: Baseline with default settings
./pipeline run --run-id baseline_v1 --symbols MES,MGC

# Run 2: Different split ratios
./pipeline run --run-id splits_60_20_20 --symbols MES,MGC \
  --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2

# Run 3: Different horizons
./pipeline run --run-id horizons_short --symbols MES,MGC \
  --horizons 1,2,3,5

# Compare runs
./pipeline compare baseline_v1 splits_60_20_20
```

### Resume After Failure

If a pipeline fails at any stage:

```bash
# Check what failed
./pipeline status failed_run_id --verbose

# Fix the issue (e.g., add missing data)

# Resume from the failed stage
./pipeline rerun failed_run_id --from labeling
```

## Error Handling

The CLI provides informative error messages:

### Configuration Errors
```
Error: Configuration error: Train/val/test ratios must sum to 1.0, got 1.1
```

### Missing Runs
```
Error: Run not found: invalid_run_id
```

### Validation Failures
```
Error: Configuration validation failed:
  • barrier_k_up must be > 0, got -1.0
  • Label horizon must be >= 1, got 0
```

## Logging

Each pipeline run creates detailed logs:

```bash
# View logs in real-time
tail -f runs/{run_id}/logs/pipeline.log

# Search for errors
grep ERROR runs/{run_id}/logs/pipeline.log

# Check specific stage
grep "STAGE 4" runs/{run_id}/logs/pipeline.log
```

Log format:
```
2024-12-18 12:00:00 - pipeline_runner - INFO - STAGE 1: Data Generation
2024-12-18 12:00:01 - generate_synthetic_data - INFO - Generating data for MES
2024-12-18 12:00:05 - pipeline_runner - INFO - ✅ Stage completed: data_generation (5.2s)
```

## Best Practices

1. **Always validate before running:**
   ```bash
   ./pipeline validate --symbols MES,MGC
   ./pipeline run --symbols MES,MGC
   ```

2. **Use descriptive run IDs:**
   ```bash
   ./pipeline run --run-id baseline_3symbols_v1 --description "Baseline with MES, MGC, MNQ"
   ```

3. **Check status during long runs:**
   ```bash
   # In another terminal
   ./pipeline status current_run_id
   ```

4. **Compare runs to understand changes:**
   ```bash
   ./pipeline compare baseline_v1 experiment_v1
   ```

5. **Clean up old runs:**
   ```bash
   ./pipeline list-runs --limit 50
   ./pipeline clean old_run_id
   ```

6. **Keep important runs:**
   ```bash
   # Use meaningful run IDs for important runs
   ./pipeline run --run-id production_baseline_20241218
   ```

## Troubleshooting

### Issue: Pipeline fails at feature engineering
```bash
# Check logs
tail -50 runs/{run_id}/logs/pipeline.log

# Validate data from previous stage
./pipeline validate --run-id {run_id}

# Resume from cleaning stage
./pipeline rerun {run_id} --from data_cleaning
```

### Issue: Artifacts missing
```bash
# Validate run
./pipeline validate --run-id {run_id}

# Check manifest
cat runs/{run_id}/artifacts/manifest.json | jq '.artifacts'
```

### Issue: Configuration not loading
```bash
# Check if config exists
ls runs/{run_id}/config/config.json

# Validate config format
cat runs/{run_id}/config/config.json | jq '.'
```

## Integration with Phase 2

The configuration system is designed to work seamlessly with Phase 2:

```python
# Load Phase 1 configuration
from pipeline_config import PipelineConfig

config = PipelineConfig.load_from_run_id('phase1_baseline')

# Use GA settings for Phase 2
print(f"Population size: {config.ga_population_size}")
print(f"Generations: {config.ga_generations}")
print(f"Crossover rate: {config.ga_crossover_rate}")

# Load prepared data
import numpy as np
import pandas as pd

train_idx = np.load(config.splits_dir / 'train_indices.npy')
df = pd.read_parquet(config.final_data_dir / 'combined_final_labeled.parquet')
train_df = df.iloc[train_idx]
```

## Summary

This comprehensive CLI system provides:

- **Complete control** over pipeline configuration
- **Reproducible runs** with versioned configurations
- **Data integrity** through checksums and manifests
- **Efficient development** with resume capabilities
- **Easy comparison** between different runs
- **User-friendly interface** with rich terminal output

Use `./pipeline --help` or `./pipeline <command> --help` for more information on any command.
