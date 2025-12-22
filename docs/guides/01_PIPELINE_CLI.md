# Pipeline CLI Complete Guide

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Installation](#installation)
3. [Commands](#commands)
4. [Configuration System](#configuration-system)
5. [Pipeline Stages](#pipeline-stages)
6. [Python API](#python-api)
7. [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Most Common Commands

```bash
# Run complete pipeline
./pipeline run --symbols MES,MGC

# Resume from specific stage
./pipeline rerun 20241218_120000 --from labeling

# Check status
./pipeline status 20241218_120000

# Validate configuration
./pipeline validate --symbols MES,MGC

# List recent runs
./pipeline list-runs

# Compare two runs
./pipeline compare run1_id run2_id
```

### Stage Names
- `data_generation` or `data` - Data acquisition/generation
- `data_cleaning` or `clean` or `cleaning` - Data cleaning
- `feature_engineering` or `features` - Feature engineering
- `labeling` or `labels` - Triple-barrier labeling
- `create_splits` or `splits` - Train/val/test splits
- `generate_report` or `report` - Completion report

---

## Installation

### Prerequisites

```bash
# Python 3.10+
python --version

# Install CLI dependencies
pip install typer rich
```

### Verify Installation

```bash
# Make wrapper executable
chmod +x pipeline

# Test CLI
./pipeline --help
```

---

## Commands

### 1. Run Pipeline

Execute the complete Phase 1 pipeline with custom parameters.

#### Basic Usage

```bash
# Default run (MES, MGC symbols)
./pipeline run

# Synthetic data for testing
./pipeline run --synthetic

# Custom symbols and date range
./pipeline run --symbols MES,MGC --start 2020-01-01 --end 2024-12-31

# Custom run ID and description
./pipeline run --run-id baseline_v1 --description "Initial baseline run"
```

#### Advanced Usage

```bash
# Multiple symbols with custom horizons
./pipeline run --symbols MES,MGC,MNQ --horizons 1,5,10,20

# Custom split ratios
./pipeline run --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2

# Custom purge and embargo
./pipeline run --purge-bars 60 --embargo-bars 576

# Full customization
./pipeline run \
  --run-id prod_baseline \
  --symbols MES,MGC \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --horizons 5,20 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --purge-bars 60 \
  --embargo-bars 288 \
  --description "Production baseline with H20 purge fix"
```

#### Options Reference

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--symbols` | `-s` | Comma-separated symbols | MES,MGC |
| `--start` | | Start date (YYYY-MM-DD) | None |
| `--end` | | End date (YYYY-MM-DD) | None |
| `--run-id` | | Custom run identifier | Auto-generated |
| `--description` | `-d` | Run description | None |
| `--train-ratio` | | Training set ratio | 0.70 |
| `--val-ratio` | | Validation set ratio | 0.15 |
| `--test-ratio` | | Test set ratio | 0.15 |
| `--purge-bars` | | Bars to purge at boundaries | 60 |
| `--embargo-bars` | | Embargo period in bars | 288 |
| `--horizons` | | Label horizons | 1,5,20 |
| `--synthetic` | | Generate synthetic data | False |
| `--project-root` | | Project root directory | Current dir |

#### Expected Output

```bash
✅ Pipeline completed successfully!

Run ID: 20241218_120000
Duration: 12m 34s
Artifacts: 28 files

Output directories:
  - data/final/MES_final_labeled.parquet
  - data/final/MGC_final_labeled.parquet
  - data/splits/train_indices.npy
  - results/PHASE1_COMPLETION_REPORT_20241218_120000.md
```

---

### 2. Resume Pipeline

Resume a pipeline run from a specific stage.

#### Basic Usage

```bash
# Auto-resume from last successful stage
./pipeline rerun 20241218_120000

# Resume from specific stage
./pipeline rerun 20241218_120000 --from labeling

# Resume with stage alias
./pipeline rerun baseline_v1 --from labels
```

#### Stage Aliases

| Canonical Name | Aliases |
|----------------|---------|
| `data_generation` | `data` |
| `data_cleaning` | `clean`, `cleaning` |
| `feature_engineering` | `features` |
| `labeling` | `labels` |
| `create_splits` | `splits` |
| `generate_report` | `report` |

#### Use Cases

**Scenario 1: GA optimization needs more generations**
```bash
# Original run
./pipeline run --run-id test1

# GA didn't converge, modify config and rerun
./pipeline rerun test1 --from labeling
```

**Scenario 2: Feature engineering bug fix**
```bash
# Fix bug in feature calculation
# Resume from feature engineering
./pipeline rerun 20241218_120000 --from features
```

**Scenario 3: Add more symbols**
```bash
# Original run with MES only
./pipeline run --symbols MES --run-id mes_only

# Add MGC and resume
# (Note: This requires manual config editing)
./pipeline rerun mes_only --from data_generation
```

---

### 3. Check Status

Check the status of a pipeline run.

#### Basic Usage

```bash
# Basic status
./pipeline status 20241218_120000

# Detailed status with all artifacts
./pipeline status 20241218_120000 --verbose
```

#### Example Output

```bash
Run ID: 20241218_120000
Description: Baseline run with MES, MGC
Created: 2024-12-18 12:00:00
Status: ✅ Completed

Configuration:
  Symbols: MES, MGC
  Date Range: 2020-01-01 to 2024-12-31
  Label Horizons: 1, 5, 20
  Split Ratios: 70/15/15
  Purge: 60 bars, Embargo: 288 bars

Pipeline Stages:
  ✅ data_generation      (5.2s)
  ✅ data_cleaning        (12.8s)
  ✅ feature_engineering  (45.3s)
  ✅ labeling            (128.7s)
  ✅ create_splits        (8.1s)
  ✅ generate_report      (2.4s)

Artifacts: 28 files (1.2 GB)
Total Duration: 12m 34s
Progress: 100%
```

---

### 4. Validate Configuration

Validate pipeline configuration and data integrity.

#### Basic Usage

```bash
# Validate new configuration
./pipeline validate --symbols MES,MGC --horizons 1,5,20

# Validate existing run
./pipeline validate --run-id 20241218_120000
```

#### What Gets Validated

**Configuration Validation:**
- Split ratios sum to 1.0
- Date format correctness (YYYY-MM-DD)
- Label horizons are positive integers
- Barrier parameters (k_up, k_down) are positive
- GA parameters are within valid ranges
- Purge/embargo bars are non-negative

**Data Integrity (for existing runs):**
- Artifact checksums match manifest
- All expected files exist
- File sizes are non-zero
- Parquet files can be read

#### Example Output

```bash
✅ Configuration is valid!

Checks passed:
  ✓ Split ratios sum to 1.0
  ✓ Date range is valid
  ✓ Label horizons are positive
  ✓ Barrier parameters are valid
  ✓ GA parameters are valid
  ✓ All artifacts present
  ✓ Checksums match
```

#### Error Example

```bash
❌ Configuration validation failed:

Issues found:
  • Split ratios sum to 1.1 (must be 1.0)
  • barrier_k_up must be > 0, got -1.0
  • Label horizon must be >= 1, got 0
  • Invalid date format: 2020/01/01 (use YYYY-MM-DD)
```

---

### 5. List Runs

List all pipeline runs.

#### Usage

```bash
# List 10 most recent runs
./pipeline list-runs

# List 20 most recent runs
./pipeline list-runs --limit 20

# List 50 most recent runs
./pipeline list-runs --limit 50
```

#### Example Output

```bash
Recent Pipeline Runs:
┌────────────────────┬─────────────────────┬────────────┬───────────┬─────────────────────┐
│ Run ID             │ Description         │ Symbols    │ Status    │ Created             │
├────────────────────┼─────────────────────┼────────────┼───────────┼─────────────────────┤
│ 20241218_150000    │ Production baseline │ MES,MGC    │ ✅ Done   │ 2024-12-18 15:00:00 │
│ 20241218_120000    │ Baseline test       │ MES,MGC    │ ✅ Done   │ 2024-12-18 12:00:00 │
│ 20241217_180000    │ GA experiment       │ MES        │ ❌ Failed │ 2024-12-17 18:00:00 │
│ baseline_v1        │ Initial run         │ MES,MGC    │ ✅ Done   │ 2024-12-16 09:00:00 │
└────────────────────┴─────────────────────┴────────────┴───────────┴─────────────────────┘
```

---

### 6. Compare Runs

Compare artifacts between two runs.

#### Usage

```bash
./pipeline compare run1_id run2_id
```

#### Example

```bash
./pipeline compare baseline_v1 experiment_v1
```

#### Example Output

```bash
Comparing runs: baseline_v1 vs experiment_v1

Added artifacts (3):
  ✨ data/final/MNQ_final_labeled.parquet
  ✨ config/ga_results/MNQ_ga_h5_best.json
  ✨ config/ga_results/MNQ_ga_h20_best.json

Removed artifacts (0):
  (none)

Modified artifacts (2):
  🔄 data/splits/train_indices.npy (checksum changed)
  🔄 results/PHASE1_COMPLETION_REPORT.md (checksum changed)

Unchanged artifacts (24):
  ✅ data/clean/MES_5m_clean.parquet
  ✅ data/clean/MGC_5m_clean.parquet
  ...
```

#### Use Cases

- Understanding what changed between runs
- Debugging pipeline modifications
- Tracking data lineage
- Identifying impact of parameter changes

---

### 7. Clean Runs

Delete a pipeline run and all its artifacts.

#### Usage

```bash
# With confirmation prompt (safe)
./pipeline clean 20241218_120000

# Force delete without confirmation (dangerous!)
./pipeline clean 20241218_120000 --force
```

#### What Gets Deleted

- Configuration files (`runs/{run_id}/config/`)
- Log files (`runs/{run_id}/logs/`)
- Artifacts (`runs/{run_id}/artifacts/`)
- **Note**: Data files in `data/` are NOT deleted (they may be used by other runs)

#### Example Output

```bash
⚠️  Warning: This will permanently delete run '20241218_120000'

Files to be deleted:
  - runs/20241218_120000/config/config.json
  - runs/20241218_120000/logs/pipeline.log
  - runs/20241218_120000/artifacts/manifest.json
  - runs/20241218_120000/artifacts/pipeline_state.json

Are you sure? [y/N]: y

✅ Run deleted successfully
```

---

## Configuration System

### Architecture

The configuration system consists of:
- **PipelineConfig**: Dataclass with all pipeline settings
- **Auto-generation**: run_id in YYYYMMDD_HHMMSS format
- **Validation**: Ensures all parameters are valid
- **Persistence**: Saved to `runs/{run_id}/config/config.json`

### Creating a Configuration

#### Method 1: Use Defaults

```python
from pipeline_config import create_default_config

config = create_default_config(
    symbols=['MES', 'MGC'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    description='Baseline run'
)
```

#### Method 2: Full Customization

```python
from pipeline_config import PipelineConfig

config = PipelineConfig(
    run_id='custom_run_v1',
    symbols=['MES', 'MGC', 'MNQ'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    label_horizons=[1, 5, 10, 20],
    train_ratio=0.60,
    val_ratio=0.20,
    test_ratio=0.20,
    purge_bars=60,
    embargo_bars=288,
    barrier_k_up=2.5,
    barrier_k_down=2.5,
    max_bars_ahead=100
)
```

### Saving and Loading

```python
# Save configuration
config.save_config()  # Saves to runs/{run_id}/config/config.json

# Load configuration
from pathlib import Path
config = PipelineConfig.load_config(Path('runs/20241218_120000/config/config.json'))

# Load by run ID
config = PipelineConfig.load_from_run_id('20241218_120000')
```

### Validation

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

---

## Pipeline Stages

### Stage 1: Data Generation
- Generates synthetic data or validates existing raw data
- Creates 1-minute OHLCV bars for each symbol
- **Output**: `data/raw/{symbol}_1m.parquet`

### Stage 2: Data Cleaning
- Resamples 1-min bars to 5-min bars
- Handles missing data and outliers
- **Output**: `data/clean/{symbol}_5m_clean.parquet`

### Stage 3: Feature Engineering
- Generates 50+ technical indicators
- Creates price, momentum, volatility, volume features
- **Output**: `data/features/{symbol}_5m_features.parquet`

### Stage 4: Labeling
- Applies triple-barrier labeling method
- Computes labels for multiple horizons (1, 5, 20 bars)
- Calculates sample weights based on quality
- **Output**: `data/final/{symbol}_labeled.parquet`

### Stage 5: Create Splits
- Combines data from all symbols
- Creates train/val/test splits with purging and embargo
- Saves split indices
- **Output**: `data/splits/train_indices.npy`, `val_indices.npy`, `test_indices.npy`

### Stage 6: Generate Report
- Creates comprehensive completion report
- Includes dataset statistics, label distributions, split info
- **Output**: `results/PHASE1_COMPLETION_REPORT_{run_id}.md`

---

## Python API

### Pipeline Runner

#### Basic Usage

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

#### Resume from Stage

```python
# Resume from specific stage
runner = PipelineRunner(config, resume=True)
success = runner.run(from_stage='labeling')
```

#### Access Results

```python
# After running, access stage results
for stage_name, result in runner.stage_results.items():
    print(f"{stage_name}: {result.status.value}")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Artifacts: {len(result.artifacts)}")
```

### Manifest System

#### Creating a Manifest

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
    file_path=Path('data/clean/MES_5m_clean.parquet'),
    stage='data_cleaning',
    metadata={'symbol': 'MES', 'rows': 100000}
)

# Save manifest
manifest.save()
```

#### Verifying Artifacts

```python
# Verify single artifact
is_valid = manifest.verify_artifact('clean_data_MES')

# Verify all artifacts
verification = manifest.verify_all_artifacts()
print(f"Valid: {sum(verification.values())}/{len(verification)}")
```

#### Comparing Runs

```python
from manifest import compare_runs

# Compare two runs
comparison = compare_runs('20241218_120000', '20241218_130000')

print(f"Added: {len(comparison['added'])}")
print(f"Removed: {len(comparison['removed'])}")
print(f"Modified: {len(comparison['modified'])}")
print(f"Unchanged: {len(comparison['unchanged'])}")
```

---

## Directory Structure

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

---

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

### Issue: All labels are neutral (0)

**Cause**: Barriers too wide or max_bars too small

**Solution**: Run GA optimization or adjust parameters manually

### Issue: GA fitness not improving

**Solution**: Check data quality

```python
import pandas as pd

df = pd.read_parquet('data/features/MES_5m_features.parquet')

# Ensure ATR values are valid
print(df['atr_14'].describe())
print(f"Missing ATR: {df['atr_14'].isna().sum()}")
```

---

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

---

## Typical Workflow

```bash
# 1. Validate configuration
./pipeline validate --symbols MES,MGC

# 2. Run pipeline
./pipeline run --run-id baseline_v1 --symbols MES,MGC

# 3. Check status
./pipeline status baseline_v1

# 4. If failed, check logs and resume
tail -50 runs/baseline_v1/logs/pipeline.log
./pipeline rerun baseline_v1 --from labeling

# 5. Compare with previous run
./pipeline compare baseline_v1 experiment_v1

# 6. Clean up old experiments
./pipeline list-runs --limit 20
./pipeline clean old_experiment_id
```

---

## Get Help

```bash
# General help
./pipeline --help

# Command-specific help
./pipeline run --help
./pipeline rerun --help
./pipeline status --help
./pipeline validate --help
./pipeline list-runs --help
./pipeline compare --help
./pipeline clean --help
```

---

For more information, see:
- [Getting Started Guide](00_GETTING_STARTED.md)
- [Labeling Guide](02_LABELING_GUIDE.md)
- [Phase 1 Specification](../phases/PHASE_1_Data_Preparation_and_Labeling.md)
