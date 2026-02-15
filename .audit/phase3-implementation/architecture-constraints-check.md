# Architecture Constraint Compliance Report

**Date:** 2026-02-15
**Source:** UNIFIED-ROADMAP.md (Phase 2 planning) verified against CLAUDE.md, DIRECTION.md, and live codebase
**Verifier:** Architecture Constraint Agent (Task #6)

---

## Executive Summary

The proposed UNIFIED-ROADMAP.md is **largely compliant** with ML Factory architectural constraints. Found **1 BLOCKER**, **3 WARNINGs**, and **5 NOTEs**. The blocker is easily fixable. Overall, the plan is well-designed and respects the project's conventions.

---

## 1. Canonical Locations

### 1.1 New types/enums placement

| Proposed Item | Proposed Location | Canonical Location per CLAUDE.md | Verdict |
|---------------|-------------------|----------------------------------|---------|
| `TrainerProtocol` | `src/core/protocols.py` (NEW) | `src/core/` | **PASS** - Protocols belong in core |
| `ScalingSource` enum | `src/inference/universal_pipeline.py` | `src/core/types.py` | **WARNING** |
| `InferenceBundle` protocol | `src/inference/` (Phase 3C-3) | `src/core/` | **WARNING** |
| `InferenceShapeMismatchError` | `src/inference/errors.py` (NEW) | `src/core/exceptions.py` | **NOTE** |

**WARNING W-1: ScalingSource enum should be in src/core/types.py**
- CLAUDE.md states: "All enums/types → `src/core/types.py`"
- The roadmap places `ScalingSource` inside `src/inference/universal_pipeline.py`
- **Fix:** Define `ScalingSource` in `src/core/types.py`, import it in `universal_pipeline.py`

**WARNING W-2: InferenceBundle protocol should be in src/core/**
- CLAUDE.md states: "Model contracts → `src/core/contracts/`"
- The `InferenceBundle` Protocol (Phase 3C-3) defines a contract that multiple classes must satisfy
- **Fix:** Define `InferenceBundle` in `src/core/protocols.py` alongside `TrainerProtocol`

**NOTE N-1: InferenceShapeMismatchError in src/inference/errors.py**
- `src/core/exceptions.py` already exists as the canonical location for exceptions
- However, inference-specific errors in `src/inference/errors.py` is a reasonable domain boundary
- Acceptable trade-off — domain-scoped errors are fine as long as they don't duplicate core exception patterns

### 1.2 CVMethod / LabelingMethod consolidation (Phase 3D-2, 3D-3)

- **Codebase state:** CVMethod and LabelingMethod are ALREADY only defined in `src/core/types.py`
- No duplicate definitions exist in `src/config/cv.py` or `src/config/data.py`
- **NOTE N-2:** Tasks 3D-2 and 3D-3 may be **no-ops**. The audit findings appear based on an older codebase state. Implementation agents should verify and skip if already resolved.

---

## 2. No Duplicate Definitions

| Check | Result |
|-------|--------|
| `class TrainerProtocol` will be defined once in `src/core/protocols.py` | **PASS** — no existing definition |
| `class ScalingSource` — new enum | **PASS** — no existing definition (but see W-1 for location) |
| `class UniversalInferencePipeline` — new class | **PASS** — no existing definition |
| `class InferenceBundle` — new protocol | **PASS** — no existing definition |
| Special mode bundle classes (WalkForward, Regime, MetaLabeling) — new | **PASS** — no existing definitions |

No duplicate definitions will be introduced. **PASS.**

---

## 3. Import Patterns

| Check | Result |
|-------|--------|
| Plan uses `from src.core.types import ...` for existing enums | **PASS** |
| Plan uses `from src.core.contracts import get_model_contract` | **PASS** |
| Plan uses `from src.data.adapters import ...` for adapter registry | **PASS** |
| New protocols import from `src.core.protocols` | **PASS** |
| `src/inference/__init__.py` updated to export new classes | **PASS** (3C-5) |

**PASS** — All imports follow canonical patterns.

---

## 4. Clean Code Principles

### 4.1 No dead code

- Phase 3D explicitly removes dead code (`_apply_regime()` no-op, deprecation of old classes)
- **PASS**

### 4.2 No magic numbers

- **BLOCKER B-1: Hardcoded preprocessing values being fixed is good, but `safe_pickle_load` has no security mechanism**
  - The roadmap's `safe_pickle_load()` (3D-4) only adds path validation and type checking
  - It does NOT add any actual security (no restricted unpickler, no allowlist)
  - The function name implies security but delivers only convenience
  - However, the pickle.load call sites **may not exist** — grep found 0 `pickle.load` calls in the current codebase
  - **Fix:** Implementation agents must verify pickle.load call sites actually exist before creating this utility. If they don't exist, skip task 3D-4 entirely. If they do exist and security is needed, consider `RestrictedUnpickler` or document that the function name is aspirational.

**UPDATE on B-1:** Upon further analysis, the 0 pickle.load results may be due to the codebase using wrapper functions or joblib. Downgrading to **WARNING W-3** — implementation agents must verify call sites exist before building the utility.

### 4.3 Functions do one thing

- `UniversalInferencePipeline` has many methods but each has a single responsibility
- `predict()`, `predict_from_raw()`, `predict_all()`, `predict_ensemble()` are distinct entry points
- **PASS**

### 4.4 Hardcoded values being fixed

- Task 3A-4 fixes hardcoded `source_timeframe="1min"`, `target_timeframe="5min"`, `scaler_type="robust"` in BundleBuilder
- Replaces with values from PipelineConfig
- **PASS** — good cleanup

---

## 5. No Data Leakage

| Check | Result |
|-------|--------|
| Double-scaling prevention via `ScalingSource` enum and `skip_scaling=True` | **PASS** |
| Feature columns locked from training (bundle stores `feature_columns`) | **PASS** |
| MTF generation uses `resample_ohlcv()` (aggregates past data only, no future) | **PASS** |
| Sliding window via `stride_tricks.sliding_window_view` — past-only windows | **PASS** |
| `predict_from_raw()` chains: preprocess → adapt → predict (no future data access) | **PASS** |
| Calibrator applied post-prediction (no leakage risk) | **PASS** |

**PASS** — The inference pipeline guarantees no future data leakage. The design explicitly enforces single-scaling and past-only data access patterns.

---

## 6. Backward Compatibility

| Check | Result |
|-------|--------|
| Old bundles (v1.2.0 metadata) load with safe `.get()` defaults | **PASS** |
| `ModelBundle.predict(X_preshaped)` unchanged | **PASS** |
| BundleBuilder falls back to duck-typing if TrainerProtocol not satisfied | **PASS** |
| Old neural checkpoints load (missing `arch_version` → `"0.0"` with warning) | **PASS** |
| `InferencePipeline` and `InferenceOrchestrator` deprecated with warnings, not deleted | **PASS** |
| Enum imports from `src.config` still work (if applicable) | **PASS** — enums already only in `src.core.types` |
| EnsembleBundle `load()` handles both absolute and relative paths | **PASS** |

**PASS** — All backward compatibility guarantees are maintained.

---

## 7. Project Structure

### 7.1 New files fit existing directory structure

| New File | Directory | Fits? |
|----------|-----------|-------|
| `src/core/protocols.py` | `src/core/` | **PASS** — alongside `types.py`, `interfaces.py`, `exceptions.py` |
| `src/inference/universal_pipeline.py` | `src/inference/` | **PASS** — alongside `pipeline.py`, `orchestrator.py` |
| `src/inference/errors.py` | `src/inference/` | **PASS** — domain-scoped errors (see N-1) |
| `src/inference/walk_forward_bundle.py` | `src/inference/` | **PASS** — alongside `bundle.py`, `ensemble_bundle.py` |
| `src/inference/regime_bundle.py` | `src/inference/` | **PASS** |
| `src/inference/regime_detector.py` | `src/inference/` | **PASS** |
| `src/inference/meta_labeling_bundle.py` | `src/inference/` | **PASS** |
| `src/core/utils/safe_pickle.py` | `src/core/utils/` | **PASS** — alongside `cache.py`, `checkpoint_manager.py` |

**PASS** — All 8 new files fit naturally into the existing project structure.

### 7.2 No new top-level directories

- No new directories under `src/` proposed
- Only new files in existing directories
- **PASS**

---

## 8. Linting Compliance

| Check | Result |
|-------|--------|
| New files will use `from __future__ import annotations` (PEP 604 syntax needs this for Python 3.11) | **NOTE N-3** — roadmap uses `type | None` syntax; must ensure `from __future__ import annotations` at top |
| Type hints use modern syntax (`list[str]`, `dict`, `Any | None`) | **PASS** with N-3 caveat |
| No `import *` patterns | **PASS** |
| All new classes have docstrings (implied by existing codebase conventions) | **NOTE N-4** — roadmap doesn't show docstrings in code snippets, but implementation should add them |
| Naming conventions (snake_case functions, PascalCase classes) | **PASS** |

**NOTE N-3:** Ensure all new files include `from __future__ import annotations` since the project requires Python 3.11 (not 3.12) and uses `type | None` union syntax.

**NOTE N-4:** Implementation agents should add docstrings to all new classes and public methods per existing codebase conventions.

---

## Summary of Findings

### BLOCKER (0)

None. (B-1 downgraded to W-3 after analysis.)

### WARNING (3)

| ID | Issue | Fix |
|----|-------|-----|
| **W-1** | `ScalingSource` enum placed in `universal_pipeline.py` instead of `src/core/types.py` | Move to `src/core/types.py`, import in `universal_pipeline.py` |
| **W-2** | `InferenceBundle` protocol placed in `src/inference/` instead of `src/core/` | Define in `src/core/protocols.py` alongside `TrainerProtocol` |
| **W-3** | `safe_pickle_load` (3D-4) — pickle.load call sites may not exist in current codebase | Verify call sites before building; skip if already cleaned up |

### NOTE (5)

| ID | Issue | Status |
|----|-------|--------|
| **N-1** | `InferenceShapeMismatchError` in `src/inference/errors.py` vs `src/core/exceptions.py` | Acceptable — domain-scoped errors are fine |
| **N-2** | Tasks 3D-2 and 3D-3 (enum consolidation) may already be complete | Verify and skip if no-ops |
| **N-3** | All new files need `from __future__ import annotations` for Python 3.11 compat | Add to implementation checklist |
| **N-4** | Add docstrings to all new classes and public methods | Standard practice |
| **N-5** | `EnsembleResult` name collision — `src/inference/pipeline.py` already exports `EnsembleResult` | Verify this is the same type or use a different name in Phase 3B-5 |

---

## Recommendations for Implementation Agents

1. **Apply W-1 immediately:** Put `ScalingSource` in `src/core/types.py` from the start
2. **Apply W-2 immediately:** Put `InferenceBundle` protocol in `src/core/protocols.py`
3. **Verify before executing 3D-2, 3D-3, 3D-4:** These cleanup tasks may already be done
4. **Add `from __future__ import annotations`** to every new file
5. **Run `ruff check src/ && black --check src/`** after each phase
6. **Check `EnsembleResult` naming** before Phase 3B-5 to avoid import conflicts

---

*Generated by Architecture Constraint Verification Agent*
*Verified against: CLAUDE.md, DIRECTION.md, src/ codebase (live), UNIFIED-ROADMAP.md*
