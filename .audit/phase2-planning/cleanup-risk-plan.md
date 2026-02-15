# Cleanup & Risk Mitigation Plan

**Date:** 2026-02-15
**Scope:** Dead code removal, duplicate consolidation, security & correctness risks
**Phase:** 2c (parallel work, no dependencies on 2a/2b unless noted)

---

## 1. Dead Code: `validate_distribution()` — KEEP (Not Dead)

- **Current state:** `src/inference/bundle.py:832` — method on ModelBundle that compares inference-time feature distributions against training stats using KS/PSI tests
- **Finding:** This is NOT dead code. It's a distribution shift detection utility for production inference monitoring. It has a docstring with usage example and is part of the public ModelBundle API.
- **Proposed change:** None — leave as-is
- **Risk if not addressed:** N/A
- **Effort:** NONE

---

## 2. Dead Code: `_apply_regime()` — No-Op Method

- **Current state:** `src/inference/preprocessing_graph.py:702` — called at line 498 during `transform()`, but the method body just returns `df` unchanged with a comment: "Regime features are already added in `_apply_features` via `add_regime_features`"
- **Proposed change:** Remove the no-op method and its call site at line 498. The comment confirms regime features are handled elsewhere in the pipeline.
- **Risk if not addressed:** LOW — no runtime impact, but confusing for maintainers who expect it to do something
- **Effort:** LOW
- **Dependencies:** None

---

## 3. Dead Code: Unused `time` Imports — Investigation Result

- **Current state:** 46 files import `time` across the codebase. All checked usages are **legitimate** — used for `time.time()` timing of training, optimization, evaluation loops.
- **Proposed change:** None — all imports are actively used
- **Risk if not addressed:** N/A
- **Effort:** NONE

---

## 4. Duplicate Enum: `CVMethod`

- **Current state:**
  - Canonical: `src/core/types.py:163` — 5 values (PURGED_KFOLD, CPCV, WALK_FORWARD, PBO, STANDARD)
  - Duplicate: `src/config/cv.py:28` — identical 5 values
  - `src/core/config.py:60` imports from `types.py` (correct)
  - `src/config/__init__.py:142` imports from `cv.py` (duplicate)
- **Proposed change:**
  1. In `src/config/cv.py`, remove `class CVMethod` definition
  2. Add `from src.core.types import CVMethod` to `src/config/cv.py`
  3. `src/config/__init__.py` re-export still works (it imports from cv.py which now re-exports from types.py)
- **Risk if not addressed:** LOW — values are identical so no runtime bug, but violates single-source-of-truth; future divergence possible
- **Effort:** LOW
- **Dependencies:** None

---

## 5. Duplicate Enum: `LabelingMethod`

- **Current state:**
  - Canonical: `src/core/types.py:213` — 4 values (TRIPLE_BARRIER, DIRECTIONAL, THRESHOLD, REGRESSION)
  - Duplicate: `src/config/data.py:52` — identical 4 values
  - `src/core/config.py:60` imports from `types.py` (correct)
  - `src/config/__init__.py:161` imports from `data.py` (duplicate)
  - No external code directly imports from `config/data.py`'s LabelingMethod
- **Proposed change:**
  1. In `src/config/data.py`, remove `class LabelingMethod` definition
  2. Add `from src.core.types import LabelingMethod` to `src/config/data.py`
  3. `src/config/__init__.py` re-export still works
- **Risk if not addressed:** LOW — same as CVMethod above
- **Effort:** LOW
- **Dependencies:** None

---

## 6. PredictionResult Duplication — RESOLVED (Re-export Only)

- **Current state:**
  - Canonical definition: `src/core/interfaces.py:125` — single `@dataclass class PredictionResult`
  - `src/models/base.py:27` does `from src.core.interfaces import PredictionResult` and re-exports it
  - All model implementations import via `from ..base import PredictionResult` (which chains to interfaces.py)
  - Phase 27 consolidation comment at `interfaces.py:132` confirms this was already cleaned up
- **Finding:** There is NO duplication. `models/base.py` re-exports from the canonical location. This is clean.
- **Proposed change:** None
- **Risk if not addressed:** N/A
- **Effort:** NONE

---

## 7. Pickle Security

- **Current state:** 17 `pickle.load()` call sites across the codebase:
  - `src/factory.py:474` — loads training_result checkpoint
  - `src/models/boosting/xgboost_model.py:295` — loads model metadata
  - `src/models/boosting/catboost_model.py:281` — loads model metadata
  - `src/models/boosting/lightgbm_model.py:352` — loads model metadata
  - `src/models/calibration/conformal.py:482` — loads conformal state
  - `src/models/calibration/calibrator.py:307` — loads calibrator state
  - `src/models/ensemble/xgboost_meta.py:279` — loads meta-learner metadata
  - `src/data/pipeline/stages/scaling/scaler.py:499` — loads scaler state
  - `src/core/utils/checkpoint_manager.py:208,224` — loads checkpoints, results
  - `src/core/utils/cache.py:190` — loads cache entries
  - `src/inference/bundle.py:644,653` — loads scaler and calibrator from bundle
  - `src/inference/ensemble_bundle.py:561,580` — loads meta-learner and scaler
  - `src/inference/preprocessing_graph.py:431` — loads scaler
  - All sites have security comments but **no actual validation** beyond comments
- **Proposed change (Phase 2c):**
  1. Add a `safe_pickle_load(path, expected_type)` utility in `src/core/utils/` that:
     - Validates the file is within the project's output directory
     - Optionally checks the loaded object is an instance of `expected_type`
     - Logs a warning for unexpected types
  2. Replace all `pickle.load()` calls with `safe_pickle_load()`
  3. Document that pickle files are trusted internal artifacts only
- **Risk if not addressed:** MEDIUM — arbitrary code execution if an attacker can place a malicious pickle file in model/bundle directories. The security comments acknowledge this but don't mitigate it.
- **Effort:** MEDIUM (17 call sites to update, plus new utility)
- **Dependencies:** None

---

## 8. Label Mapping: -1,0,1 to 0,1,2

- **Current state:**
  - Centralized in `src/models/common/label_mapping.py`
  - Constants: `LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}`, `CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}`
  - Functions: `map_labels_to_classes()`, `map_classes_to_labels()` with validation
  - Exported via `src/models/common/__init__.py:14`
  - Vectorized implementation: `(arr + 1).astype(np.int32)`
- **Finding:** The mapping IS centralized and well-implemented with input validation. However, the audit report (Section 3.1) flags that this mapping is **not stored in bundles**. If the mapping convention ever changed, old models would break silently.
- **Proposed change:**
  1. Add `label_mapping_version: int = 1` to BundleMetadata
  2. At bundle load time, validate the version matches the current code version
  3. This is a small metadata addition — no logic change needed
- **Risk if not addressed:** LOW — the mapping is unlikely to change, but storing it makes bundles self-documenting
- **Effort:** LOW
- **Dependencies:** Coordinates with Phase 2a bundle metadata work (adding `scaling_source` field)

---

## 9. Feature Names in Boosting Bundles

- **Current state:**
  - All 3 boosting models (XGBoost, LightGBM, CatBoost) have:
    - `self._feature_names: list[str] | None` field
    - `set_feature_names(names)` method
    - Feature names saved/loaded in pickle metadata
  - `src/models/training/trainer.py:736-744` calls `set_feature_names()` after training if:
    - Model has the method AND `not self.model.requires_sequences`
    - Uses feature selector results if available, else raw feature_names
  - Feature names are stored in model metadata during `save()` and restored during `load()`
- **Finding:** Feature names ARE stored and loaded correctly. The `get_feature_importance()` methods use real names when available, falling back to `f0, f1, ...` only if `_feature_names` is None.
- **Potential gap:** When a bundle is built, BundleBuilder stores `feature_columns` in metadata, but the underlying model's `_feature_names` may not be set if the model was loaded from a checkpoint rather than trained fresh. This is an edge case.
- **Proposed change:** In BundleBuilder, after loading/extracting a model, call `set_feature_names(feature_columns)` if the model supports it. This ensures bundle-loaded models always have names set.
- **Risk if not addressed:** LOW — feature importance shows `f0, f1, ...` instead of real names in some edge cases
- **Effort:** LOW
- **Dependencies:** Coordinates with Phase 2a TrainerProtocol work

---

## Summary Table

| # | Item | Status | Action | Risk | Effort | Phase 2 Dep |
|---|------|--------|--------|------|--------|-------------|
| 1 | `validate_distribution()` | Not dead | None | — | NONE | — |
| 2 | `_apply_regime()` no-op | Dead code | Remove method + call | LOW | LOW | None |
| 3 | Unused `time` imports | All used | None | — | NONE | — |
| 4 | Duplicate `CVMethod` | Duplicate | Consolidate to types.py | LOW | LOW | None |
| 5 | Duplicate `LabelingMethod` | Duplicate | Consolidate to types.py | LOW | LOW | None |
| 6 | `PredictionResult` duplication | Already resolved | None | — | NONE | — |
| 7 | Pickle security | No validation | Add `safe_pickle_load()` | MEDIUM | MEDIUM | None |
| 8 | Label mapping in bundles | Not stored | Add version to metadata | LOW | LOW | 2a metadata |
| 9 | Feature names in bundles | Edge case gap | Call `set_feature_names()` in BundleBuilder | LOW | LOW | 2a TrainerProtocol |

---

## Recommended Execution Order

1. **Items 2, 4, 5** — Quick cleanup, no dependencies, parallelizable (30 min total)
2. **Item 7** — Pickle security utility (standalone, 2-3 hours)
3. **Items 8, 9** — Bundle metadata additions, coordinate with Phase 2a work

Items 1, 3, 6 require no action.
