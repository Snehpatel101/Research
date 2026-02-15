# High-Level Architecture: Deployable Artifact After Notebook

**Date:** 2026-02-15
**Status:** High-Level Design (No code changes in this document)
**Audience:** Product, ML engineering, inference engineering

---

## 1. Product Outcome

After notebook training completes, output should be deployment-first:

- One selected deployable artifact per horizon.
- Same runtime interface regardless of single model or ensemble.
- Caller provides raw bars only; artifact handles preprocessing + adaptation + prediction.

---

## 2. Architecture at a Glance

```text
Notebook Run Complete
        |
        v
Training Outputs + Candidate Bundles
        |
        v
Deploy Selector (per horizon)
  - single mode -> best ModelBundle
  - ensemble mode -> EnsembleBundle
        |
        v
Deploy Artifact Package
  - metadata + manifest + validation
  - runtime profile: native/onnx/torchscript
        |
        v
Serving Call
  artifact.predict_from_raw(raw_bars_df)
```

---

## 3. Runtime Flow (Raw Bars to Prediction)

```text
raw OHLCV bars
   -> preprocessing graph replay (train/serve parity)
   -> adapter routing (2D / 3D / 4D)
   -> model inference (single or ensemble)
   -> calibration (if present)
   -> PredictionResult
```

### 3.1 Adapter Routing Rules

- Tabular models: 2D features.
- Sequence models: sliding-window 3D tensor.
- Multi-timeframe transformer models: 4D tensor with per-timeframe windows.
- Ensemble: base bundle predictions combined by meta-learner.

### 3.2 Invariants

- No caller-side tensor shaping.
- No future leakage in windowing/resampling.
- Exactly one scaling source per inference path.

---

## 4. Artifact Types

## 4.1 Model Artifact

Represents one base model with everything needed for inference:

- serialized model weights
- scaler/calibrator (if present)
- feature schema/order
- preprocessing graph/config
- adapter requirements metadata

Primary method:

```python
bundle.predict_from_raw(raw_df)
```

## 4.2 Ensemble Artifact

Represents meta-learner + base artifact references.

Responsibilities:

- load base artifacts
- run base `predict_from_raw(raw_df)`
- stack/align base outputs
- run meta-learner

Primary method:

```python
ensemble_bundle.predict_from_raw(raw_df)
```

---

## 5. Deployment Package Shape

Per horizon:

```text
deploy/h{horizon}/
  artifact/                 # selected single or ensemble artifact
  artifact.tar.gz           # optional packaged form
  validation.json
```

Run-level:

```text
deploy/manifest.json
```

Manifest should identify:

- selected artifact type (`model` or `ensemble`)
- horizon
- model family/families
- runtime profile
- compatibility/runtime requirements

---

## 6. Architecture Constraints Mapping

This design explicitly aligns with repository constraints:

1. Enums/types in `src/core/types.py`.
2. Protocol contracts in `src/core/protocols.py`.
3. Model contracts remain in `src/core/contracts/`.
4. No duplicate conceptual definitions across modules.
5. Backward compatibility for existing bundle loading where practical.

---

## 7. ONNX / Runtime Profile Strategy

ONNX should be implemented as a **runtime profile**, not a mandatory replacement.

Profiles:

- `native` (default): no export conversion risk.
- `onnx` (optional): use when model/export support is stable.
- `torchscript` (optional for torch families): family-dependent.

Design rule:

- Package generation never fails solely because ONNX export is unsupported.
- If ONNX export fails, artifact falls back to `native` and records reason in validation output.

---

## 8. Selection Policy

Per horizon selection should be deterministic and explicit.

- If ensemble exists and passes validation: choose ensemble artifact.
- Otherwise choose best base model artifact by configured metric.
- Persist decision in deploy manifest (with metric snapshot).

This removes ambiguity for downstream serving and CI automation.

---

## 9. Observability and Validation

Each deploy artifact should carry a minimal, machine-readable validation report.

Checks:

1. loadability
2. schema validation
3. smoke `predict_from_raw` on sample bars
4. timing summary
5. runtime profile info

The report is deployment-facing, not training-facing.

---

## 10. Non-Goals (Architecture Boundary)

- This design does not require replacing all legacy inference classes immediately.
- This design does not prescribe full internal refactors outside deployability path.
- This design does not require ONNX for every model family in phase one.

---

## 11. Why This Architecture Fits Your Goal

Your goal: one package from notebook output that you can feed bars into directly.

This architecture enforces exactly that contract:

- one selected artifact per horizon
- one raw-bars inference method
- one deploy manifest for automation
- optional portability profiles without blocking core usability

