# TOPSTEPX ML Model Factory Documentation

---

## IMPORTANT: Current vs Intended Architecture

**Current State (TEMPORARY):** The codebase implements a universal pipeline where all models receive the same ~180 indicator-derived features. This is a comparison/baseline architecture.

**Intended State (THE GOAL):** Model-specific pipelines with 3 MTF strategies:
- **Strategy 1 (Single-TF):** Train on one timeframe only - NOT IMPLEMENTED
- **Strategy 2 (MTF Indicators):** Tabular models get indicator features from 9 timeframes - PARTIAL (5/9 TFs)
- **Strategy 3 (MTF Ingestion):** Sequence models get raw OHLCV bars from 9 timeframes - NOT IMPLEMENTED

**Current Limitations:**
- Only 5 of 9 timeframes implemented (15min, 30min, 1h, 4h, daily)
- No model-specific data preparation (all models get same features)
- Sequence models receive indicators when they should get raw bars

**READ FIRST:**
| Document | Description |
|----------|-------------|
| [Intended Architecture](INTENDED_ARCHITECTURE.md) | THE GOAL - Model-specific pipelines |
| [Current Limitations](CURRENT_LIMITATIONS.md) | What's wrong with current implementation |
| [Migration Roadmap](MIGRATION_ROADMAP.md) | How to fix (6-8 week plan) |
| [MTF Strategy Guide](guides/MTF_STRATEGY_GUIDE.md) | Choose your strategy |

---

## Quick Start

This `docs/` tree describes the **implemented** pipeline: Phase 1 data prep + labeling, Phase 2 training, Phase 3 evaluation/CV utilities, Phase 4 ensembles, and Phase 5 inference/monitoring utilities.

### Start Here

- **Notebook workflow** (Colab or local Jupyter): `../notebooks/ML_Pipeline.ipynb`
  - Notebook docs: `notebook/README.md`
  - Colab helper: `COLAB_GUIDE.md` (uses `../colab_setup.py`)
- **CLI workflow** (Phase 1 + scripts): `getting-started/QUICKSTART.md`
  - Phase 1 CLI reference: `getting-started/PIPELINE_CLI.md`
  - Phase 1 (plain English): `phase1/README.md`
  - Command quick reference: `QUICK_REFERENCE.md`

---

## Phase Status

| Phase | Status | Description | Primary entrypoint | Doc |
|------:|--------|-------------|--------------------|-----|
| 1 | **COMPLETE** | Data prep -> features -> labels -> splits -> scaling | `./pipeline run ...` | `phases/PHASE_1.md` |
| 2 | **COMPLETE** | Train any registered model (13 models) | `scripts/train_model.py` / notebook | `phases/PHASE_2.md` |
| 3 | **COMPLETE** | Purged CV / walk-forward / CPCV-PBO | `scripts/run_cv.py` | `phases/PHASE_3.md` |
| 4 | **COMPLETE** | Voting/stacking/blending ensembles | `scripts/train_model.py --model ...` | `phases/PHASE_4.md` |
| 5 | **PARTIAL** | Bundles + serving + drift utilities | `scripts/serve_model.py` | `phases/PHASE_5.md` |

**Note:** All phases marked "COMPLETE" use the universal pipeline (same ~180 features for all models). See [Current Limitations](CURRENT_LIMITATIONS.md) for details.

---

## Navigation by Persona

### For New Users
1. Read [Project Charter](planning/PROJECT_CHARTER.md) - Project goals and status
2. Read [Quick Reference](QUICK_REFERENCE.md) - Common commands
3. Follow [Quickstart](getting-started/QUICKSTART.md) - First pipeline run
4. Review [Pipeline CLI](getting-started/PIPELINE_CLI.md) - CLI options

### For ML Engineers
1. Read [Intended Architecture](INTENDED_ARCHITECTURE.md) - Understand the goal
2. Read [Current Limitations](CURRENT_LIMITATIONS.md) - Understand gaps
3. Review [Model Integration Guide](guides/MODEL_INTEGRATION_GUIDE.md) - BaseModel interface
4. Check [Implementation Tasks](analysis/IMPLEMENTATION_TASKS.md) - Code changes needed

### For Data Scientists
1. Read [Feature Engineering Guide](guides/FEATURE_ENGINEERING_GUIDE.md) - Feature strategies
2. Review [MTF Strategy Guide](guides/MTF_STRATEGY_GUIDE.md) - Choose data strategy
3. Check [Feature Catalog](reference/FEATURES.md) - Available features
4. See [Best OHLCV Features](research/BEST_OHLCV_FEATURES.md) - Research findings

### For DevOps/Deployment
1. Read [Model Infrastructure Requirements](guides/MODEL_INFRASTRUCTURE_REQUIREMENTS.md) - Hardware needs
2. Review [Architecture](reference/ARCHITECTURE.md) - System design
3. Check [Validation Checklist](VALIDATION_CHECKLIST.md) - Pre-deployment steps
4. See [Phase 5](phases/PHASE_5.md) - Serving & monitoring

---

## Pipeline Paths (What lands where)

| Path | Purpose |
|------|---------|
| `data/raw/` | Raw data inputs (e.g., `{SYMBOL}_1m.parquet`) |
| `runs/<run_id>/` | Phase 1 run metadata/logs |
| `data/splits/scaled/*.parquet` | Phase 1 dataset outputs (gitignored) |
| `experiments/runs/<run_id>/` | Training runs/artifacts |
| `data/stacking/`, `data/walk_forward/`, `data/cpcv_pbo/` | CV outputs |
| `ModelBundle` directories | Serving/batch inference (see `phases/PHASE_5.md`) |

---

## Reference Documentation

### Architecture Understanding (Critical)

| Doc | Purpose |
|-----|---------|
| [Intended Architecture](INTENDED_ARCHITECTURE.md) | Target state - model-specific pipelines |
| [Current Limitations](CURRENT_LIMITATIONS.md) | What's wrong with universal pipeline |
| [Current vs Intended](CURRENT_VS_INTENDED_ARCHITECTURE.md) | Detailed gap analysis |
| [Migration Roadmap](MIGRATION_ROADMAP.md) | 6-phase implementation plan |

### Core Guides

| Doc | Purpose |
|-----|---------|
| [MTF Strategy Guide](guides/MTF_STRATEGY_GUIDE.md) | Choosing data strategies |
| [Model Integration Guide](guides/MODEL_INTEGRATION_GUIDE.md) | Adding new models |
| [Feature Engineering Guide](guides/FEATURE_ENGINEERING_GUIDE.md) | Feature strategies |
| [Hyperparameter Guide](guides/HYPERPARAMETER_OPTIMIZATION_GUIDE.md) | GA + Optuna tuning |

### Technical Reference

| Doc | Purpose |
|-----|---------|
| `reference/ARCHITECTURE.md` | Architecture + design constraints |
| `reference/FEATURES.md` | Feature catalog (Phase 1) |
| `reference/PIPELINE_FIXES.md` | Notes on Phase 1 alignment + verification |
| `VALIDATION_CHECKLIST.md` | Pre/post-training validation checklists |
| `WORKFLOW_BEST_PRACTICES.md` | Best practices & recommended workflows |

### Troubleshooting

| Doc | Purpose |
|-----|---------|
| [MTF Troubleshooting](troubleshooting/MTF_TROUBLESHOOTING.md) | MTF-specific issues |
| `notebook/TROUBLESHOOTING.md` | Notebook/Colab troubleshooting |
| `MIGRATION_GUIDE.md` | Migration guide for recent improvements |

---

## Status & Monitoring

| Doc | Purpose |
|-----|---------|
| `../PIPELINE_STATUS.md` | Overall pipeline status dashboard |
| `../INTEGRATION_FIXES_SUMMARY.md` | Phase 3->4 integration improvements |
| `../ENSEMBLE_VALIDATION_SUMMARY.md` | Ensemble validation improvements |

---

## Implementation Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Models Implemented** | 13 of 19 | Boosting (3), Neural (4), Classical (3), Ensemble (3) |
| **MTF Timeframes** | 5 of 9 | Missing: 1min, 10min, 20min, 25min |
| **Strategy 1 (Single-TF)** | NOT IMPLEMENTED | Baseline support |
| **Strategy 2 (MTF Indicators)** | PARTIAL | Works but missing 4 timeframes |
| **Strategy 3 (MTF Ingestion)** | NOT IMPLEMENTED | Raw bars for sequence models |

See [Full Documentation Index](INDEX.md) for complete navigation.

---

## Legacy / Archived Docs

Spec-heavy or outdated documents are kept under `archive/` for reference and are not the current runbook.

---

*Last Updated: 2026-01-01*
