# Goal → Gap Matrix (Upload → 9 Timeframes → Train Single/Ensemble)

This matrix restates the target workflow and maps it to what exists today vs what is missing/contradictory.

## Target Workflow (Canonical)

1. **User uploads one OHLCV dataset** (typically `data/raw/{SYMBOL}_1m.parquet` or `.csv`)
2. Pipeline **derives nine timeframe datasets** automatically (e.g., `1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h`)
3. System can train:
   - A **single model** (tabular / sequence / advanced multi-res)
   - A **heterogeneous ensemble** (stacking w/ meta-learner) from the same underlying market data
4. Each model can choose per run:
   - single timeframe only,
   - single timeframe + higher timeframe indicators,
   - multi-timeframe ingestion (e.g., 3 timeframes at once).
4. Artifacts are tracked and reproducible:
   - data runs (pipeline) + model runs (training) + bundles (inference)
   - consistent config + deterministic labeling/splitting + leakage controls

## Capability Matrix

| Capability | Needed For Goal | Current State (as implemented) | Gap / Why It Blocks |
|---|---|---|---|
| One clear entrypoint (“how to run”) | Anyone can run end-to-end | `pipeline` wrapper → `src/pipeline_cli.py` → Typer CLI (`src/cli/`) | CLI has broken commands and inconsistent flags; docs reference non-existent scripts |
| One authoritative config layer | “Any shoe can be configured” | Pipeline uses `src/phase1/pipeline_config.py` (dataclass). Training has its own `TrainerConfig` + YAML loaders in `src/models/config/` | Split-brain: multiple config systems exist but are not consistently wired; defaults conflict |
| Single uploaded dataset becomes canonical “source of truth” | Reproducibility + parity | Ingest validates raw to `data/raw/validated/*` (global). Cleaning produces one target TF under `runs/{run_id}/data/clean/` | Not a single canonical dataset concept: global + run-scoped mix; “canonical 1m” is not preserved run-scoped |
| Automatic generation of **nine datasets** | Core ask | Only one `target_timeframe` per pipeline run; MTF features are added as columns to the base TF | No multi-timeframe output orchestration; existing multi-timeframe utilities are not wired into the pipeline stages |
| Timeframe-aware horizons + purge/embargo | Leakage-safe across TFs | Horizon scaling utilities exist in `src/common/horizon_config.py` | Pipeline produces labels in **bars**, not time; no “9 TF” orchestration that applies timeframe-aware horizon scaling per TF |
| Per-model primary timeframe selection | Model diversity + ensembles | Docs claim this is supported; code does not provide a per-model TF selection mechanism integrated into the Phase 1 outputs | Training reads one scaled dataset directory; no clean interface that maps model→TF→dataset |
| Per-model MTF strategy selection | “Configure any shoe” | MTF generator supports modes (`bars/indicators/both`) and configurable TF list | Pipeline applies MTF at feature engineering time without model-specific routing; dataset container serves mostly one view |
| Multi-timeframe ingestion (3 TF at once) | Advanced models + diversity | `MultiResolution4DAdapter` exists under `src/phase1/stages/datasets/adapters/` | No canonical pipeline output contract guaranteeing the multi-stream inputs; not exposed as a per-model config-driven view |
| Train single model + ensembles | Phase 2+ | Model registry, trainer, ensembles, meta-learners exist under `src/models/` | Training is not connected to pipeline runs by default; docs reference `scripts/train_model.py` which does not exist |
| Model management / bundling | Deployment | `src/inference/bundle.py` and `src/inference/pipeline.py` exist | Missing a clean “train → bundle” orchestrated flow linked to pipeline run artifacts and config |

## Summary

The repo has many of the *components* of a model factory, but the end-to-end “upload → 9 TF datasets → per-model TF/MTF strategies → train/bundle” workflow is blocked by:

- unclear and contradictory sources of truth (config, paths, doc claims),
- missing orchestration for multi-timeframe dataset production,
- weak coupling between pipeline artifacts (`runs/*`) and training artifacts (`experiments/runs/*`),
- multiple broken CLI/docs entrypoints.
