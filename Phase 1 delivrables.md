# Phase 1 — Data & Labeling Deliverables

## Table of Contents
- [Objective](#objective)
- [Scope](#scope)
- [Inputs](#inputs)
- [Process](#process)
- [Deliverables](#deliverables)
- [Pipeline Notes](#pipeline-notes)
- [Recommended Pipeline Shape](#recommended-pipeline-shape)
- [How To Run It](#how-to-run-it)
- [Key Design Rules](#key-design-rules)
- [One-Command Interface](#one-command-interface)
- [Quality Gates](#quality-gates)
- [Definition of Done](#definition-of-done)

## Objective
Produce clean, consistent market datasets (features + labels) suitable for Phase 2 model training—without lookahead leakage, broken sections, or misleading labels.

## Scope
- **Markets:** e.g., MES, MGC (extendable)
- **Bar size:** 1-minute OHLCV
- **History:** multi-year (as available)
- **Primary risks addressed:** gaps, duplicates, outliers, contract roll artifacts, label imbalance, leakage

## Inputs
- Raw 1-minute OHLCV per market
- Contract metadata / roll rules (for futures continuity)
- Feature definitions (what gets computed)
- Labeling rule family (how win/loss/neutral is defined)
- Optimization search space (if tuning label rules)

## Process

### 1) Acquire raw data
- Ingest multi-year 1-minute OHLCV per market.
- Standardize timestamp convention (timezone + session handling) and document it.

### 2) Clean & normalize
Must address and document:
- Missing minutes (gaps) and how they are handled
- Spikes/outliers and how they are handled
- Duplicate timestamps and how they are handled
- Contract roll / stitching behavior (continuous series rules)

**Output:** a single “clean truth” dataset per market.

### 3) Feature engineering (signals)
Add computed features that help the model interpret context, such as:
- Volatility measures
- Trend / momentum measures
- Volume features
- Time-of-day / session context (NY/London/Asia, etc.)

### 4) Label generation (training answers)
Label each timestamp based on what happens next under the chosen rules:
- **Win**: up move meets threshold
- **Loss**: down move meets threshold
- **Neutral / no-trade**: no meaningful move

### 5) Tune label rules (optional, recommended)
Use an optimizer (e.g., GA / Optuna) to search thresholds/parameters within bounds.

**Goal:** labels reflect realistic tradable movement, not noise or bias.

### 6) Time-based splitting (no leakage)
Create chronological splits:
- Train
- Validation
- Test

Persist split definitions (timestamp boundaries or explicit indices) so they can be reproduced.

### 7) Validate usability (baseline)
- Run sanity checks (counts, missingness, ranges).
- Run a simple baseline strategy/backtest to detect “obviously broken” pipelines.

## Deliverables

### A) Data artifacts (per market)
- Cleaned OHLCV dataset
- Final modeling dataset containing:
  - Clean prices
  - Features (signals)
  - Labels (answers)
  - Optional sample weights (so higher-quality examples matter more)

### B) Reproducibility artifacts
- Saved labeling settings (best configuration from tuning)
- Saved Train/Validation/Test splits (time-based)

### C) Phase 1 summary report
At minimum:
- Data health summary (gaps/outliers/duplicates/roll handling)
- Feature list summary
- Label distribution summary (win/loss/neutral balance)
- Baseline backtest summary (high-level results + sanity notes)

## Pipeline Notes
A good pipeline for Phase 1 is one that’s modular, repeatable, and produces versioned artifacts at each step—so you can re-run just the parts that change (like label params) without redoing everything.

## Recommended Pipeline Shape
The pipeline shape that works best (recommended): a staged, artifact-based ML pipeline.

### Stage 0 — Config
- One config file controls instruments, date ranges, feature set, label horizons, GA settings, split rules.
- **Output:** `config/run_YYYYMMDD.json` (frozen snapshot)

### Stage 1 — Ingest
- Pull raw 1m OHLCV into a consistent schema.
- **Output:** `data/raw/{symbol}.parquet`

### Stage 2 — Clean + Normalize
- Fix gaps, duplicates, timezone, roll/continuous series, outliers policy.
- **Output:** `data/clean/{symbol}.parquet`

### Stage 3 — Feature Build
- Compute all indicators + time/session features.
- **Output:** `data/features/{symbol}_features.parquet`

### Stage 4 — Labeling (Triple-Barrier)
- Apply initial labels for 1/5/20 horizons.
- **Output:** `data/labels/{symbol}_labels_init.parquet`

### Stage 5 — Optimize Label Params (GA)
- Run GA on a subset to pick best barrier settings per horizon.
- **Output:** `config/ga_results/ga_{horizon}_best.json` + `reports/ga_convergence.png`

### Stage 6 — Final Labels + Weights
- Re-label full dataset using GA params; add sample weights/tiers.
- **Output:** `data/final/{symbol}_final_labeled.parquet`

### Stage 7 — Time Split (Leakage-safe)
- Train/val/test indices with purge+embargo.
- **Output:** `splits/{symbol}_{run_id}/train.npy`, `splits/{symbol}_{run_id}/val.npy`, `splits/{symbol}_{run_id}/test.npy`

### Stage 8 — Validation Pack
- Label distribution checks, feature sanity, baseline backtest.
- **Output:** `reports/phase1_summary.md` + `baseline_backtest.html`

## How To Run It
What to use to run it (2 solid options):

### Option A (fast + simple, works great locally)
- Python scripts + a single CLI entrypoint (Typer or argparse)
- Orchestration: Makefile or justfile
- Storage: Parquet
- Compute: polars + numpy (+ numba if needed)
- Tracking: MLflow (optional but nice)

### Option B (more “real pipeline”, best if you’ll iterate a lot)
- Orchestrator: Dagster or Prefect
- Caching/retries built-in
- Each stage is a “job” with declared inputs/outputs
- Great for rerunning only “Label+GA” without rebuilding features

## Key Design Rules
These rules prevent downstream pain and make Phase 1 rerunnable and debuggable:
- Every stage writes an artifact (Parquet/JSON/NPY) and never silently mutates prior stages.
- Idempotent stages: same inputs + same config = same output.
- Run IDs: every run gets a unique `run_id` folder so you can compare runs.
- Data versioning: DVC or at least checksums in a manifest (`manifest.json`) so you know what changed.
- Leakage guards: splits happen after labels, and time rules are enforced consistently.

## One-Command Interface
Ideal UX for running and iterating on Phase 1:

```bash
pipeline run --symbols MES,MGC --start 2005-01-01 --end 2025-12-01 --run-id phase1_v1
```

Relabel without rebuilding raw/clean/features:

```bash
pipeline rerun --from labels --run-id phase1_v1
```

pipeline rerun --from labels --run-id phase1_v1 (relabels without rebuilding raw/clean/features)

If you want, I can sketch a clean folder layout + minimal `pipeline.py` CLI skeleton (Typer/Dagster style) that matches those stages exactly.

## Quality Gates
- **Integrity:** no duplicate timestamps; gaps quantified and handled
- **Continuity:** roll/stitch rules applied consistently (documented)
- **Label sanity:** not wildly imbalanced; thresholds produce tradable frequency
- **No leakage:** splits are chronological; features/labels do not depend on future data
- **Reproducible:** configs + split boundaries are persisted and rerunnable

## Definition of Done
- [ ] Clean datasets exist for each market in scope
- [ ] Modeling datasets include features + labels (and optional weights)
- [ ] Best labeling settings are saved and reproducible
- [ ] Train/Validation/Test splits are saved and time-based
- [ ] Summary report exists and indicates the pipeline is sane
- [ ] Ready handoff to Phase 2 (model training)
