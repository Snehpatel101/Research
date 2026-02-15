# Phase 0 Plan: Deployable Artifact First

**Date:** 2026-02-15
**Status:** Planning Addendum (No code changes in this document)
**Scope:** Deliver one deployable inference artifact per horizon after notebook completion
**Objective Owner:** Inference/Productization

---

## 1. Goal (User-Centric)

After `notebooks/ml_factory_colab.ipynb` finishes:

- If training produced a single best model, output **one deployable ModelBundle artifact**.
- If training produced an ensemble, output **one deployable EnsembleBundle artifact**.
- In both cases, caller should only provide raw bars (OHLCV) to a single method:

```python
prediction = artifact.predict_from_raw(raw_bars_df)
```

No manual feature shaping, no manual adapter calls, no manual model-family branching.

---

## 2. What This Plan Prioritizes

This plan intentionally narrows scope from the broader Phase 3A-3D roadmap to one product outcome:

- **P0:** Artifact is reliable, loadable, and callable from raw bars.
- **P1:** Artifact is portable (single directory or tarball).
- **P2:** Artifact has optional runtime export profile (ONNX where feasible).

This is a sequencing plan, not a replacement for the full architecture roadmap.

---

## 3. Architectural Constraints (Must-Haves)

Derived from `CLAUDE.md`, `DIRECTION.md`, and `.audit/phase3-implementation/architecture-constraints-check.md`.

### 3.1 Canonical Locations

- Enums/types must live in `src/core/types.py`.
- Protocol contracts must live in `src/core/protocols.py`.
- Model contracts must remain in `src/core/contracts/`.
- No duplicate class/type definitions across `src/`.

### 3.2 Inference Safety and Parity

- No data leakage in inference path (past-only windows; no forward lookups).
- Exactly one scaling stage per prediction path.
- Feature ordering and schema must be pinned to training-time artifact metadata.

### 3.3 Backward Compatibility

- Existing bundle loading behavior must remain backward-compatible where possible.
- Existing `predict(X)` should continue to work for pre-shaped inputs.
- Legacy pipeline/orchestrator can be deprecated later, not removed immediately.

---

## 4. Definition of Done (Artifact-First)

A run is complete only if all are true:

1. Notebook run outputs `deploy/` with one selected deployable artifact per horizon.
2. Artifact can be loaded in a clean process with one command.
3. Artifact accepts raw OHLCV bars and returns predictions in one call.
4. Same API shape for single-model and ensemble artifacts.
5. Simple smoke validation report is generated alongside artifact.

---

## 5. Artifact Contract

## 5.1 Runtime Contract

All deployable artifacts (single or ensemble) must satisfy:

- `predict_from_raw(raw_df) -> PredictionResult`
- `predict(features_or_tensor) -> PredictionResult`
- `validate() -> dict`
- `save(path)` / `load(path)`

Protocol location (when implemented): `src/core/protocols.py`.

## 5.2 Packaging Contract

Per horizon output:

- `deploy/h{horizon}/artifact/` (directory form), and optionally
- `deploy/h{horizon}/artifact.tar.gz` (packaged form)

Each artifact package should include:

- Model(s)
- Required scaler/calibrator metadata
- Feature column/order metadata
- Preprocessing graph/config
- Adapter routing metadata (sequence/4D requirements)
- Manifest with version + checksums
- Minimal validation report

---

## 6. Execution Plan (Phased)

## Phase P0-A: Bundle Reliability and Selection (Highest Priority)

**Purpose:** Ensure notebook reliably emits a loadable deployable artifact.

Tasks:

1. Stabilize training-to-bundle metadata extraction (model identity, feature schema, scaler lineage).
2. Ensure bundle generation covers configured training mode outputs (standard, walk-forward, regime, meta-labeling).
3. Define deterministic selection policy for deploy artifact per horizon:
   - Single-model mode: best model by agreed metric.
   - Ensemble mode: ensemble artifact selected by default.
4. Add run-level deployment manifest under `deploy/`.

Acceptance:

- Artifacts load successfully from produced run output.
- No "unknown model" or unresolved references in produced deploy artifact.

---

## Phase P0-B: Universal Raw-Bars Inference Path

**Purpose:** Make `predict_from_raw()` truly universal for all supported families.

Tasks:

1. Raw bars -> preprocessing -> adapter routing -> model prediction in one path.
2. Adapter routing based on model contract + bundle metadata:
   - 2D tabular
   - 3D sequence windows
   - 4D multi-timeframe streams
3. Enforce single-scaling rule and explicit scaling source.
4. Ensure ensemble artifact delegates raw inference to base artifacts internally.

Acceptance:

- Same method call works for boosting, sequence, transformer, and ensemble artifacts.
- No caller-side tensor shaping required.

---

## Phase P0-C: Notebook Deploy UX

**Purpose:** Make deploy artifact the default notebook output users care about.

Tasks:

1. Add final notebook cell that prints deploy artifact path(s).
2. Add artifact-only export/download (not full run zip only).
3. Add short inference demo cell using raw bars and deploy artifact.
4. Emit a compact deployment summary (selected artifact, horizon, model family, validation status).

Acceptance:

- User can run notebook end-to-end and copy one artifact path into serving code.

---

## Phase P0-D: Optional Runtime Profile (ONNX/Torch)

**Purpose:** Improve portability where model family supports it.

Profiles:

- `native` (default): existing framework runtime.
- `onnx` (optional): exported graph + ONNX runtime metadata.
- `torchscript` (optional for torch models): traced/scripted module where stable.

Rules:

1. Export profile is optional per model family; unsupported families fall back to `native`.
2. Bundle metadata records runtime profile + required runtime version.
3. Validation report must include profile-specific load/predict smoke result.

Acceptance:

- ONNX/Torch profile does not block native artifact generation.
- Profile selection is explicit in deploy manifest.

---

## 7. Deliverables

When this plan is implemented, each run should produce:

```text
experiments/runs/{run_id}/
  deploy/
    manifest.json
    h20/
      artifact/                # Single selected deployable artifact
      artifact.tar.gz          # Optional packaged form
      validation.json
```

`deploy/manifest.json` minimum fields:

- `run_id`
- `created_at`
- `horizons`
- `selected_artifacts` (single vs ensemble)
- `runtime_profile` (`native` / `onnx` / `torchscript`)
- `compatibility` (min runtime versions)

---

## 8. Risks and Mitigations

1. **Training mode mismatch (e.g., walk-forward artifacts not directly bundle-ready)**
   - Mitigation: explicit adapter/wrapper artifact strategy per mode.

2. **Scaling duplication / drift**
   - Mitigation: single scaling source policy + validation assertion.

3. **4D multi-timeframe inference mismatch**
   - Mitigation: contract-driven timeframe metadata embedded in artifact.

4. **Portability issues across environments**
   - Mitigation: profile-aware validation report; native fallback always available.

5. **Over-expansion of scope**
   - Mitigation: keep P0 strictly focused on deployable artifact + raw-bars API.

---

## 9. Out of Scope for This Phase-0 Plan

- Full deprecation/removal of old inference classes.
- Deep cleanup tasks unrelated to deploy artifact objective.
- Major retraining strategy redesign.
- Broad refactors that do not improve deployability in notebook workflow.

---

## 10. Implementation Order Recommendation

1. P0-A (reliability + selection)
2. P0-B (universal raw-bars path)
3. P0-C (notebook deploy UX)
4. P0-D (optional ONNX/Torch profile)

This order delivers user-visible value earliest while staying architecture-compliant.

---

## 11. Decision Log (Initial)

1. **One selected deploy artifact per horizon** is the primary product output.
2. **Artifact API standardization** (`predict_from_raw`) is mandatory.
3. **Architecture constraints are hard requirements**, not optional polish.
4. **ONNX is opt-in**, not a blocker for initial deployability.

