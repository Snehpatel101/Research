# ML Model Factory Documentation

This `docs/` tree describes the **implemented** pipeline: Phase 1 data prep + labeling, Phase 2 training, Phase 3 evaluation/CV utilities, Phase 4 ensembles, and Phase 5 inference/monitoring utilities.

## Start Here

- Notebook workflow (Colab or local Jupyter): `../notebooks/ML_Pipeline.ipynb`
  - Notebook docs: `notebook/README.md`
  - Colab helper: `COLAB_GUIDE.md` (uses `../colab_setup.py`)
- CLI workflow (Phase 1 + scripts): `getting-started/QUICKSTART.md`
  - Phase 1 CLI reference: `getting-started/PIPELINE_CLI.md`
  - Phase 1 (plain English): `phase1/README.md`
  - Command quick reference: `QUICK_REFERENCE.md`

## Phase Status

| Phase | Status | Description | Primary entrypoint | Doc |
|------:|--------|-------------|--------------------|-----|
| 1 | **COMPLETE** | Data prep → features → labels → splits → scaling | `./pipeline run ...` | `phases/PHASE_1.md` |
| 2 | **COMPLETE** | Train any registered model | `scripts/train_model.py` / notebook | `phases/PHASE_2.md` |
| 3 | **COMPLETE** | Purged CV / walk-forward / CPCV-PBO | `scripts/run_cv.py` | `phases/PHASE_3.md` |
| 4 | **COMPLETE** | Voting/stacking/blending ensembles | `scripts/train_model.py --model ...` | `phases/PHASE_4.md` |
| 5 | **PARTIAL** | Bundles + serving + drift utilities | `scripts/serve_model.py` | `phases/PHASE_5.md` |

## Pipeline Paths (What lands where)

- Raw data inputs: `data/raw/` (e.g. `{SYMBOL}_1m.parquet`)
- Phase 1 run metadata/logs: `runs/<run_id>/`
- Phase 1 dataset outputs (typically gitignored): `data/splits/scaled/*.parquet`
- Training runs/artifacts: `experiments/runs/<run_id>/`
- CV outputs: `data/stacking/`, `data/walk_forward/`, `data/cpcv_pbo/`
- Serving/batch inference: requires `ModelBundle` directories (see `phases/PHASE_5.md`)

## Reference Docs

| Doc | Purpose |
|-----|---------|
| `reference/ARCHITECTURE.md` | Architecture + design constraints |
| `reference/FEATURES.md` | Feature catalog (Phase 1) |
| `reference/PIPELINE_FIXES.md` | Notes on Phase 1 alignment + verification |
| `MIGRATION_GUIDE.md` | Migration guide for recent improvements |
| `VALIDATION_CHECKLIST.md` | Pre/post-training validation checklists |
| `WORKFLOW_BEST_PRACTICES.md` | Best practices & recommended workflows |
| `notebook/TROUBLESHOOTING.md` | Notebook/Colab troubleshooting |
| `QUANTITATIVE_TRADING_ANALYSIS.md` | Research notes (not a runbook) |

## Status & Monitoring

| Doc | Purpose |
|-----|---------|
| `../PIPELINE_STATUS.md` | Overall pipeline status dashboard |
| `../INTEGRATION_FIXES_SUMMARY.md` | Phase 3→4 integration improvements |
| `../ENSEMBLE_VALIDATION_SUMMARY.md` | Ensemble validation improvements |

## Legacy / Archived Docs

Spec-heavy or outdated documents are kept under `archive/` for reference and are not the current runbook.
