# Architecture (Current)

This repo is an ML “factory” with a clear separation between:

- **Phase 1 (data)**: raw OHLCV → features/labels/splits (CLI: `./pipeline`)
- **Phase 2 (training)**: split parquets → trained model artifacts (CLI: `scripts/train_model.py`)
- **Phase 3 (evaluation)**: purged CV / walk-forward / CPCV-PBO (CLIs: `scripts/run_cv.py`, `scripts/run_walk_forward.py`, `scripts/run_cpcv_pbo.py`)
- **Phase 4 (ensembles)**: voting/stacking/blending (via Phase 2 training + Phase 3 OOF)
- **Phase 5 (inference/monitoring utilities)**: bundles + serving + drift utilities (CLIs: `scripts/serve_model.py`, `scripts/batch_inference.py`)

## Entry Points

- Phase 1 CLI wrapper: `./pipeline` → `src/pipeline_cli.py` → `src/cli/*`
- Training: `scripts/train_model.py`
- CV / stability:
  - Purged CV + OOF: `scripts/run_cv.py`
  - Walk-forward: `scripts/run_walk_forward.py`
  - CPCV + PBO: `scripts/run_cpcv_pbo.py`
- Inference:
  - HTTP server: `scripts/serve_model.py`
  - Batch inference: `scripts/batch_inference.py`

## Key Data Contracts

- **Raw OHLCV input** (Phase 1): files under `data/raw/` (schema validated by pipeline stages)
- **Model-ready splits** (Phase 1 output): `data/splits/scaled/train_scaled.parquet`, `val_scaled.parquet`, `test_scaled.parquet`
- **Training container**: `src/phase1/stages/datasets/container.py` (`TimeSeriesDataContainer`)
  - Horizon selection is by label columns (e.g. `label_h20`, `sample_weight_h20`)
- **Model interface**: `src/models/base.py` (`BaseModel` + standardized outputs/metrics)

## Artifact Layout (by responsibility)

- Phase 1 run metadata/logs: `runs/<run_id>/` (config + stage state + logs)
- Phase 1 reports: `results/` (markdown reports)
- Phase 2 training runs: `experiments/runs/<model>_h<horizon>_<timestamp>/`
  - `checkpoints/best_model/` is the canonical “loadable” checkpoint path
- Phase 3 outputs:
  - CV/OOF: `data/stacking/`
  - Walk-forward: `data/walk_forward/`
  - CPCV/PBO: `data/cpcv_pbo/`
- Phase 5 bundles: directory bundles created via `src/inference/bundle.py` (`ModelBundle`)

## Invariants (things the code assumes)

- **Symbol isolation by default**: multi-symbol Phase 1 runs require an explicit opt-in (`allow_batch_symbols`)
- **Leakage prevention**: purge/embargo gaps are used for splitting/CV; labels are horizon-based
- **Seeded runs**: there is a `random_seed` concept, but full determinism depends on backend/hardware

## Current Gaps (important if you want “production ML”)

- Phase 5 serving/batch expects **features**, not raw OHLCV; a real-time feature pipeline is not wired in here yet.
- Bundling for inference is not a first-class CLI step yet; it’s typically assembled in the notebook (or with small glue code using `src/inference/ModelBundle`).
