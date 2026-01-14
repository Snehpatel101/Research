# Architecture Flow Map (ASCII)

**Repo intent (as written across docs/ + src/):** a *model factory* that can run a leakage-aware OHLCV pipeline (Phase 1), then train single models and ensembles (Phase 2+), and finally package/serve inference artifacts with train/serve parity.

**Primary code locations:**
- Pipeline orchestration: `src/pipeline/` (runner + stage registry)
- Pipeline stage implementations: `src/phase1/stages/`
- Pipeline configuration (Python): `src/phase1/pipeline_config.py`
- Model factory (registry + trainer + models): `src/models/`
- Cross-validation + OOF stacking utilities: `src/cross_validation/`
- Inference bundling + server: `src/inference/`
- Validation utilities (e.g., DSR, lookahead audit): `src/validation/`
- Trading sim (not tightly integrated): `src/simulation/`

---

## 1) High-Level System Goal

```
One canonical OHLCV source (per symbol)
    -> deterministic feature/label pipeline (leakage controls)
    -> adapters / containers (2D tabular, 3D sequences, etc.)
    -> plugin model registry (single models + ensembles/meta-learners)
    -> standardized artifacts (runs, metrics, predictions, bundles)
    -> reproducible evaluation + optional serving
```

---

## 2) Current Implemented Flow (What `src/` Actually Wires Up)

### 2.1 Entry Points / Control Plane

```
User
  |
  |  (preferred)
  |  `src/pipeline_cli.py`  (Typer app in `src/cli/`)
  v
CLI Commands
  |
  v
PipelineConfig (Python dataclass)
  `src/phase1/pipeline_config.py`
  |
  v
PipelineRunner
  `src/pipeline/runner.py`
  |
  v
Stage Registry (ordering + deps)
  `src/pipeline/stage_registry.py`
```

### 2.2 Phase 1 Pipeline (Stage-by-stage Artifacts)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ INPUTS (global)                                                            │
│  - data/raw/{SYMBOL}_1m.parquet | data/raw/{SYMBOL}_1m.csv                 │
│  - config/* (YAML) for models/ensembles/pipeline (used mostly by training) │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: data_generation                                                    │
│  - implementation: src/phase1/stages/ingest/run.py                          │
│  - output (GLOBAL): data/raw/validated/{SYMBOL}_1m_validated.parquet        │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: data_cleaning                                                      │
│  - implementation: src/phase1/stages/clean/run.py                           │
│  - output (RUN-SCOPED): runs/{run_id}/data/clean/{SYMBOL}_{tf}_clean.parquet│
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: feature_engineering                                                │
│  - implementation: src/phase1/stages/features/run.py                        │
│  - includes MTF generation via src/phase1/stages/mtf/* (mode + timeframes)  │
│  - output (RUN-SCOPED): runs/{run_id}/data/features/{SYMBOL}_{tf}_features. │
│                    parquet                                                  │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: initial_labeling                                                   │
│  - implementation: src/phase1/stages/labeling/run.py                        │
│  - output (RUN-SCOPED): runs/{run_id}/data/labels/*                         │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 5: ga_optimize                                                        │
│  - implementation: src/phase1/stages/ga_optimize/run.py                     │
│  - output (RUN-SCOPED): runs/{run_id}/artifacts/* (optuna/plots/params)     │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 6: final_labels                                                       │
│  - implementation: src/phase1/stages/final_labels/run.py                    │
│  - output (RUN-SCOPED): runs/{run_id}/data/final/*                          │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 7: create_splits                                                      │
│  - implementation: src/phase1/stages/splits/run.py                          │
│  - output (RUN-SCOPED): runs/{run_id}/data/splits/*                         │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 7.5: feature_scaling                                                  │
│  - implementation: src/phase1/stages/scaling/run.py                         │
│  - output (RUN-SCOPED): runs/{run_id}/data/splits/scaled/* (+ stats json)   │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 7.6: build_datasets                                                   │
│  - implementation: src/phase1/stages/datasets/run.py                        │
│  - output (RUN-SCOPED): dataset manifests + container-friendly assets       │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 7.7: validate_scaled                                                  │
│  - implementation: src/phase1/stages/scaled_validation/run.py               │
│  - output (RUN-SCOPED): validation artifacts                                │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 8: validate                                                           │
│  - implementation: src/phase1/stages/validation/run.py                      │
│  - output: validation summaries (run-scoped)                                │
└────────────────────────────────────────────────────────────────────────────┘

    |
    v
┌────────────────────────────────────────────────────────────────────────────┐
│ Stage 9: generate_report                                                    │
│  - implementation: src/phase1/stages/reporting/run.py                       │
│  - output: run summary report + charts                                      │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Phase 2+ Training (Model Factory)

```
Scaled splits / datasets
  |
  v
TimeSeriesDataContainer
  `src/phase1/stages/datasets/container.py`
  |
  v
Trainer (or training_utils.train_model)
  `src/models/trainer.py`
  |
  v
ModelRegistry (plugin system)
  `src/models/registry.py`
  |
  +--> Tabular models (boosting/classical): `src/models/boosting/*`, `src/models/classical/*`
  |
  +--> Sequence models (torch): `src/models/neural/*`
  |
  +--> Ensembles + meta-learners: `src/models/ensemble/*`
  |
  v
Artifacts (training outputs)
  experiments/runs/{train_run_id}/...
```

### 2.4 Cross-Validation + OOF Stacking (Research / Robustness)

```
Container + ModelRegistry
  |
  v
PurgedKFold / Walk-Forward / CPCV / PBO
  `src/cross_validation/*`
  |
  v
OOF predictions + stacking datasets
  (used by ensemble/meta-learner workflows)
```

### 2.5 Inference Packaging + Serving (Train/Serve Parity)

```
Trained model + preprocessing graph + metadata
  |
  v
Bundle (portable artifact)
  `src/inference/bundle.py`
  |
  v
Batch inference pipeline + optional HTTP server
  `src/inference/pipeline.py`, `src/inference/server.py` (Flask optional)
```

---

## 3) Where This Architecture Wants To Go (Target End-State)

```
Single source of truth for config:
  CLI args -> config YAML -> validated dataclasses (one merge layer)

Single “run” concept:
  runs/{pipeline_run_id}/... (data + reports)
  experiments/runs/{training_run_id}/... (models + metrics)
  bundles/{bundle_id}/... (deployment artifacts)

Single import surface:
  No “ghost” module names (e.g., `stages` vs `src.phase1.stages`)
  Optional deps don’t break base imports (boosting/neural loaded lazily)
```

