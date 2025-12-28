# Pipeline CLI Guide

## Quick Reference

```bash
# Run pipeline
./pipeline run --symbols MES,MGC

# Resume from a stage
./pipeline rerun <run_id> --from initial_labeling

# Status and validation
./pipeline status <run_id>
./pipeline validate --run-id <run_id>

# Manage runs
./pipeline list-runs
./pipeline compare <run_id_1> <run_id_2>
./pipeline clean <run_id>

# Presets
./pipeline presets
```

## Commands

### run

```bash
./pipeline run [options]
```

Options:
- `--symbols`, `-s`: Comma-separated symbols
- `--preset`, `-p`: Preset name (see `pipeline presets`)
- `--timeframe`, `-t`: Target timeframe (e.g., `1min`, `5min`, `15min`)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--run-id`: Custom run id
- `--description`, `-d`: Run description
- `--train-ratio`: Training split ratio
- `--val-ratio`: Validation split ratio
- `--test-ratio`: Test split ratio
- `--purge-bars`: Purge bars (disables auto-scaling)
- `--embargo-bars`: Embargo bars (disables auto-scaling)
- `--horizons`: Comma-separated horizons (e.g., `5,10,15,20`)
- `--feature-set`: Feature set (see `src/phase1/config/feature_sets.py`)
- `--model-type`: Model-aware preparation hint (e.g., `xgboost`, `lstm`, `transformer`, `ensemble`)
- `--base-models`: Base models for model-aware ensemble prep (e.g., `xgboost,lstm,transformer`)
- `--meta-learner`: Meta-learner hint for stacking/blending ensembles (e.g., `logistic`)
- `--sequence-length`: Sequence length hint for sequence models (LSTM/Transformer)
- `--project-root`: Project root path

### rerun

```bash
./pipeline rerun <run_id> --from <stage_name>
```

Use a stage name from the stage registry:
- `data_generation`
- `data_cleaning`
- `feature_engineering`
- `initial_labeling`
- `ga_optimize`
- `final_labels`
- `create_splits`
- `feature_scaling`
- `build_datasets`
- `validate_scaled`
- `validate`
- `generate_report`

Note: the CLI has only a few aliases; use full stage names for reliability.

### status

```bash
./pipeline status <run_id> [--verbose]
```

### validate

```bash
./pipeline validate --run-id <run_id>
./pipeline validate --symbols MES,MGC
```

### list-runs

```bash
./pipeline list-runs --limit 20
```

### compare

```bash
./pipeline compare <run_id_1> <run_id_2>
```

### clean

```bash
./pipeline clean <run_id> [--force]
```

### presets

```bash
./pipeline presets
./pipeline presets <name> -v
```

## Artifacts and Paths

- Run config: `runs/<run_id>/config/config.json`
- Run logs: `runs/<run_id>/logs/pipeline.log`
- Manifest: `runs/<run_id>/artifacts/manifest.json`
- Scaled splits: `data/splits/scaled/*.parquet`
- Completion report: `results/PHASE1_COMPLETION_REPORT_<run_id>.md`
