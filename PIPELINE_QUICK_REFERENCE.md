# Pipeline CLI Quick Reference

## Installation
```bash
pip install typer rich
```

## Basic Commands

### Run Pipeline
```bash
# Basic run
./pipeline run

# With symbols and dates
./pipeline run --symbols MES,MGC --start 2020-01-01 --end 2024-12-31

# Custom run ID
./pipeline run --run-id phase1_v1 --description "Baseline run"

# Multiple horizons
./pipeline run --horizons 1,5,10,20

# Generate synthetic data
./pipeline run --synthetic
```

### Resume Pipeline
```bash
# Resume from specific stage
./pipeline rerun 20241218_120000 --from labeling

# Auto-resume from last successful stage
./pipeline rerun 20241218_120000
```

### Check Status
```bash
# Basic status
./pipeline status 20241218_120000

# Detailed status
./pipeline status 20241218_120000 --verbose
```

### Validate
```bash
# Validate new config
./pipeline validate --symbols MES,MGC

# Validate existing run
./pipeline validate --run-id 20241218_120000
```

### List Runs
```bash
# List 10 recent runs
./pipeline list-runs

# List 20 recent runs
./pipeline list-runs --limit 20
```

### Compare Runs
```bash
./pipeline compare run1_id run2_id
```

### Clean Runs
```bash
# With confirmation
./pipeline clean 20241218_120000

# Force delete
./pipeline clean 20241218_120000 --force
```

## Stage Names
- `data_generation` or `data`
- `data_cleaning` or `clean` or `cleaning`
- `feature_engineering` or `features`
- `labeling` or `labels`
- `create_splits` or `splits`
- `generate_report` or `report`

## Common Options
- `--symbols, -s`: Comma-separated symbols (default: MES,MGC)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--run-id`: Custom run identifier
- `--description, -d`: Run description
- `--train-ratio`: Training ratio (default: 0.70)
- `--val-ratio`: Validation ratio (default: 0.15)
- `--test-ratio`: Test ratio (default: 0.15)
- `--purge-bars`: Purge bars (default: 20)
- `--embargo-bars`: Embargo bars (default: 288)
- `--horizons`: Label horizons (default: 1,5,20)
- `--synthetic`: Use synthetic data
- `--verbose, -v`: Verbose output

## Python API

### Configuration
```python
from pipeline_config import PipelineConfig, create_default_config

# Create config
config = create_default_config(
    symbols=['MES', 'MGC'],
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# Save/load
config.save_config()
config = PipelineConfig.load_from_run_id('20241218_120000')

# Validate
issues = config.validate()
```

### Runner
```python
from pipeline_runner import PipelineRunner

runner = PipelineRunner(config)
success = runner.run()

# Resume
runner = PipelineRunner(config, resume=True)
success = runner.run(from_stage='labeling')
```

### Manifest
```python
from manifest import ArtifactManifest, compare_runs

# Load manifest
manifest = ArtifactManifest.load('20241218_120000', Path('/home/user/Research'))

# Verify
verification = manifest.verify_all_artifacts()

# Compare
comparison = compare_runs('run1', 'run2')
```

## Directory Structure
```
/home/user/Research/
├── data/
│   ├── raw/              # Raw 1-min data
│   ├── clean/            # Cleaned 5-min data
│   ├── features/         # Features
│   ├── final/            # Labeled data
│   └── splits/           # Indices
├── runs/{run_id}/
│   ├── config/           # Configuration
│   ├── logs/             # Logs
│   └── artifacts/        # Manifest & state
└── results/              # Reports
```

## Typical Workflow

1. **Validate configuration:**
   ```bash
   ./pipeline validate --symbols MES,MGC
   ```

2. **Run pipeline:**
   ```bash
   ./pipeline run --run-id baseline_v1 --symbols MES,MGC
   ```

3. **Check status:**
   ```bash
   ./pipeline status baseline_v1
   ```

4. **If failed, resume:**
   ```bash
   ./pipeline rerun baseline_v1 --from labeling
   ```

5. **Compare with previous run:**
   ```bash
   ./pipeline compare baseline_v1 experiment_v1
   ```

## Get Help
```bash
./pipeline --help
./pipeline run --help
./pipeline rerun --help
./pipeline status --help
```
