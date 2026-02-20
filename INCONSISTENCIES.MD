# Inconsistency Audit (`src/`)

Generated: 2026-02-20 21:32:13 UTC  
Scope: `src/` implementation consistency review, focused on orchestration, OOF/stacking, config contracts, and CLI/runtime paths.

## Severity Legend
- `Critical`: Likely wrong results or broken core artifacts.
- `High`: Major behavior mismatch or frequent runtime risk.
- `Medium`: Inconsistent behavior, drift, or latent defect.

## Findings

### 1) OOF Predictions Cleared Before Artifact Save
- Severity: `Critical`
- Evidence:
  - `src/models/training/unified_orchestrator.py:428`
  - `src/models/training/unified_orchestrator.py:431`
  - `src/models/training/unified_orchestrator.py:435`
  - `src/models/training/unified_orchestrator.py:522`
- Explanation:
  - During ensemble builds, `_oof_predictions` is cleared before `_save_results()` runs.
  - Artifact saving depends on `request.oof_predictions` populated from that dict.
- Why this is inconsistent:
  - Intended behavior is to persist OOF artifacts when `save_oof=True`; actual behavior can drop them.
- Impact:
  - Missing OOF parquet outputs, reduced traceability, and downstream tooling/reporting blind spots.

### 2) Multi-Horizon OOFs Are Mixed Into a Single Ensemble Build
- Severity: `Critical`
- Evidence:
  - `src/models/training/training_ops.py:132`
  - `src/models/training/training_ops.py:163`
  - `src/models/training/unified_orchestrator.py:489`
  - `src/models/training/unified_orchestrator.py:508`
- Explanation:
  - OOF keys are horizon-scoped (`model_h{horizon}`), but ensemble build consumes the full dict together.
  - Result horizon is hard-set from `horizons[0]`.
- Why this is inconsistent:
  - Horizon-specific labels/features should not be stacked across horizons.
- Impact:
  - Potentially trains meta-learner on mixed targets and wrong temporal semantics.

### 3) Non-4D OOF Path Flattens Prepared Data and Uses Empty Model Config
- Severity: `High`
- Evidence:
  - `src/models/training/services/oof_generation.py:123`
  - `src/models/training/services/oof_generation.py:137`
  - `src/validation/cv/oof_generator.py:253`
- Explanation:
  - Prepared 3D data is flattened to 2D for OOF generation.
  - OOF model config is `{}`, so sequence length and tuned params are not propagated.
- Why this is inconsistent:
  - Base training and OOF generation should use equivalent model assumptions/config.
- Impact:
  - OOF distribution can drift from true base model behavior; stacking learns from mismatched signals.

### 4) 4D OOF Fallback Trains With Empty Config
- Severity: `High`
- Evidence:
  - `src/models/training/services/oof_generation.py:228`
- Explanation:
  - If cached fold models are absent, 4D fold models are created with `config={}`.
- Why this is inconsistent:
  - OOF predictions should reflect trained/tuned fold behavior, not default fallback behavior.
- Impact:
  - Meta-learner receives lower-fidelity or behaviorally different base predictions.

### 5) Diversity Alignment Uses Positional Slicing Instead of Original Index Mapping
- Severity: `High`
- Evidence:
  - `src/models/training/feature_selection.py:419`
  - `src/models/training/services/ensemble_service.py:410`
  - `src/models/training/services/ensemble_service.py:417`
- Explanation:
  - Alignment logic indexes by array position, not explicit original sample indices.
- Why this is inconsistent:
  - Heterogeneous OOF alignment contract is index-based (`original_indices`), not purely positional.
- Impact:
  - Diversity metrics may be miscomputed, especially for sequence models with offsets.

### 6) `pipeline run` CLI Uses Deprecated Orchestrator With Stale OOF Field Access
- Severity: `High`
- Evidence:
  - `src/cli/commands/pipeline.py:118`
  - `src/orchestrator.py:35`
  - `src/orchestrator.py:316`
  - `src/orchestrator.py:333`
- Explanation:
  - CLI path uses deprecated `MLPipeline`.
  - Backtest extraction expects fields like `ensemble_predictions`, `ensemble_probabilities`, and `oof.indices` that current OOF structures do not expose consistently.
- Why this is inconsistent:
  - Public CLI path should target current orchestration and current data structures.
- Impact:
  - Runtime failures or invalid backtest inputs on the legacy path.

### 7) Walk-Forward Mode Prepares Data Using Only First Model Contract
- Severity: `High`
- Evidence:
  - `src/models/training/training_ops.py:339`
  - `src/models/training/training_ops.py:362`
- Explanation:
  - Walk-forward builds model list, but preparation uses only `self.config.models[0]`.
- Why this is inconsistent:
  - Multi-model runs should honor each model’s adapter/contract shape.
- Impact:
  - Heterogeneous model sets can train/evaluate on data not matching each model’s required format.

### 8) Per-Model Feature Subsets Applied in Standard Mode but Bypassed in Other Modes
- Severity: `High`
- Evidence:
  - `src/models/training/training_ops.py:92`
  - `src/models/training/training_ops.py:157`
  - `src/models/training/unified_orchestrator.py:341`
  - `src/models/training/training_ops.py:539`
  - `src/models/training/training_ops.py:642`
- Explanation:
  - Standard mode uses `_prepare_with_cache` (which filters per-model features).
  - Regime-aware and meta-labeling call `_data_preparer.prepare` directly.
- Why this is inconsistent:
  - Feature-selection policy should not silently change by training mode.
- Impact:
  - Non-standard modes may ignore the intended feature caps/subsets.

### 9) CV Method Configuration Is Collapsed to PurgedKFold in Unified Orchestrator
- Severity: `Medium`
- Evidence:
  - `src/models/training/unified_orchestrator.py:282`
  - `src/models/training/unified_orchestrator.py:285`
  - `src/models/training/unified_orchestrator.py:288`
- Explanation:
  - `cpcv`, `walk_forward`, and `pbo` branches all return `PurgedKFold`.
- Why this is inconsistent:
  - Config value indicates specific CV semantics, but implementation homogenizes them.
- Impact:
  - User intent/config can be silently overridden.

### 10) Conflicting Model Requirement Registries (`MODEL_CONTRACTS` vs `MODEL_DATA_REQUIREMENTS`)
- Severity: `Medium`
- Evidence:
  - `src/core/contracts/model_contract.py:243`
  - `src/models/config/data_requirements.py:121`
  - `src/core/contracts/model_contract.py:328`
  - `src/models/config/data_requirements.py:179`
  - `src/core/contracts/model_contract.py:375`
  - `src/models/config/data_requirements.py:284`
- Explanation:
  - Max features and even family/type semantics differ between the two registries (e.g., `xgboost`, `lstm`, `patchtst`).
- Why this is inconsistent:
  - Different subsystems enforce different truths for the same model.
- Impact:
  - Validation, feature caps, and behavior can diverge by entrypoint.

### 11) Stacking Feature Schema Differs Across Training and Inference Paths
- Severity: `Medium`
- Evidence:
  - `src/data/adapters/alignment.py:85`
  - `src/models/ensemble/heterogeneous_stacking.py:307`
  - `src/validation/cv/oof_stacking.py:297`
  - `src/inference/ensemble_bundle.py:867`
- Explanation:
  - One path uses probs + 2 derived features; another adds entropy (+3); CV stacking builder adds a broader engineered set.
- Why this is inconsistent:
  - Meta-learner feature schema should be stable between training and inference.
- Impact:
  - Risk of feature-count/name mismatch and degraded deploy compatibility.

### 12) Ensemble Label Extraction Is Horizon-Ambiguous
- Severity: `Medium`
- Evidence:
  - `src/models/training/services/ensemble_service.py:276`
- Explanation:
  - Fallback label extraction chooses first `label_h*` column, not an explicit target horizon.
- Why this is inconsistent:
  - Multi-horizon training requires deterministic label-horizon selection.
- Impact:
  - Meta-learner may train on unintended target labels.

### 13) Lookahead Toggle Semantics Conflict Across Subsystems
- Severity: `Medium`
- Evidence:
  - `src/models/training/feature_selection.py:171`
  - `src/data/pipeline/stages/validation/__init__.py:304`
  - `src/data/pipeline/stages/validation/__init__.py:398`
- Explanation:
  - Unified training can skip lookahead check via `check_lookahead=False`.
  - Data pipeline validation declares lookahead audit mandatory and always-on.
- Why this is inconsistent:
  - Same configuration flag has contradictory meanings depending on path.
- Impact:
  - Confusing behavior and uneven safety guarantees across entrypoints.

### 14) Legacy and Modern Training Entrypoints Coexist With Different Semantics
- Severity: `Medium`
- Evidence:
  - `src/factory.py:707`
  - `src/cli/commands/train.py:550`
  - `src/models/training/features.py:26`
  - `src/models/training/feature_selection.py:222`
- Explanation:
  - `MLFactory` uses `UnifiedTrainingOrchestrator` (core contracts).
  - CLI `train model` uses `Trainer` path (validates against MODEL_DATA_REQUIREMENTS).
- Why this is inconsistent:
  - Different entrypoints apply different feature/contract policies and training flows.
- Impact:
  - Same experiment can produce materially different behavior depending on command path.

## Notes
- This document focuses on implementation consistency, not stylistic preference.
- Line numbers are direct evidence points where the inconsistency manifests or is enforced.
