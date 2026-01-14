# Configuration: Collapse to a Single Source of Truth

The repo currently contains multiple configuration “centers”, which prevents the factory from being reliably configurable.

## What exists today

### Phase 1 (data pipeline)

- Primary driver: `src/phase1/pipeline_config.py` (dataclass)
- CLI creation path: `src/cli/run_commands_core.py` → `PipelineConfig(**kwargs)`
- Additional config modules: `src/phase1/config/*` (timeframes, labels, feature sets, etc.)

### Phase 2+ (training / ensembles / meta-learners)

- `src/models/config/*` can load YAML from `config/models/*.yaml` and global training/cv YAML in `config/pipeline/*.yaml`.
- `src/models/config/trainer_config.py` defines a dataclass that can also be constructed directly.

## Why this blocks “adaptive/dynamic”

- There is no consistent merge order like: `defaults < YAML < CLI`.
- Different subsystems validate different timeframe strings (`1h` vs `60min` vs missing `25min`).
- Docs imply “change YAML to change behavior”, but the pipeline is primarily driven by Python defaults/CLI.

## What “single source of truth” should mean (proposed)

One config model (conceptually):

- `ProjectConfig`
  - `DataRunConfig` (Phase 1): symbols, date range, base TF(s), label policy, split policy, scaling policy, output paths
  - `TrainingRunConfig` (Phase 2): model(s), per-model dataset view (TF, feature set, MTF strategy), tuning policy, output paths
  - `EnsembleRunConfig` (Phase 3): base models, OOF policy, meta-learner policy

One merge layer:

1. code defaults (safe baseline)
2. YAML file(s) (experiment presets)
3. CLI overrides (single-run changes)

## High-impact fixes to enable that outcome (documentation-only guidance)

- Declare which configs are **authoritative** vs **reference/templates**.
- Standardize timeframe vocabulary (choose one canonical set; define aliases; validate once).
- Define an explicit mapping from `PipelineConfig.feature_set` values to `FeatureSetDefinition` names (and document it).
- Decide whether multi-symbol runs are supported by default; align CLI defaults with that decision.

