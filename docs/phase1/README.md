# Phase 1 (Plain English)

Phase 1 takes raw OHLCV bars and produces **model-ready, leakage-aware** datasets:

`raw OHLCV → clean/resample → features → labels → splits (purge/embargo) → scaling → parquet outputs`

## How you run it

```bash
./pipeline run --symbols MES
./pipeline status <run_id>
./pipeline validate --run-id <run_id>
```

## What you get

Typical outputs (often gitignored because they’re large):

- Scaled splits for training: `data/splits/scaled/train_scaled.parquet`, `val_scaled.parquet`, `test_scaled.parquet`
- Run metadata/logs: `runs/<run_id>/`
- Human-readable report(s): `results/PHASE1_COMPLETION_REPORT_<run_id>.md`

## Where the detailed docs live

- Phase 1 technical details: `docs/phases/PHASE_1.md`
- Feature catalog: `docs/reference/FEATURES.md`
- CLI reference: `docs/getting-started/PIPELINE_CLI.md`
