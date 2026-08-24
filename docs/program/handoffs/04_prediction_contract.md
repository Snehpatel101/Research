# Handoff — Stage 4: Canonical Data Contracts (`PredictionSet`)

**Date:** 2026-08-24
**Predecessor:** Stage 3 (`03_package_organisation.md`)

---

## 1. Verification of predecessor

| Stage 3 claim | How I checked | Verdict |
|---|---|---|
| Suite 605p/2xf/0f | Re-ran full suite | **CONFIRMED** |
| `src.core` imports ~0.4s, no heavy deps | Re-measured after adding two modules: **0.37s, heavy=[]** | **CONFIRMED** |
| `MODEL_DATA_REQUIREMENTS` same object via both paths | Re-ran import smoke | **CONFIRMED** |
| `ensure_registered()` is a no-op today | Accepted; not relied upon | **CONFIRMED** |

Stage 3's warning to keep `src.core` light directly shaped this stage: both new
modules import **numpy only**, and a test enforces it in a subprocess.

## 2. What was built

`src/core/predictions.py`
- **`PredictionSchema`** — `n_classes`, `class_labels`, `horizon`, `task`.
  Frozen. Equality is the typed replacement for the `k.endswith("_h5")` string
  check that previously decided combinability.
- **`PredictionProvenance`** — model name, family, data rank, `source`
  (`oof`/`val`/`test`/`live`), calibration state, fingerprints.
- **`PredictionSet`** — `keys` (**mandatory, source-bar coordinates**),
  `probabilities (N,C)`, `valid (N,)` bool, optional `timestamps`/`fold_id`.

`src/core/prediction_panel.py`
- **`PredictionPanel.align()`** — a **join on source-bar keys**, replacing the
  positional set-intersection that Phase 0 proved wrong.

## 3. The four design rules, and why each exists

1. **Keys are mandatory and are source-bar coordinates.** A prediction without
   the identity of what it predicts is unusable. This is the direct structural
   fix for F1.
2. **`valid` is the ONLY missingness signal.** Measured before writing this:
   `-99` appears **92** times, `INVALID_LABEL` **40** times — and
   `INVALID_LABEL = -99` is independently redefined in **five** modules
   (`core/container.py`, `data/adapters/preparation.py`,
   `data/pipeline/stages/datasets/validators.py`,
   `data/pipeline/stages/meta_labeling/meta_labeler.py`,
   `models/training/services/hyperparameter_tuning.py`). A sentinel must never
   collide with real data — a promise numeric code cannot keep here, since
   `-1` is a genuine class label. `keys` containing a legacy sentinel now
   raises at construction.
3. **`class_predictions`, `confidence`, `margin` are DERIVED, never stored.**
   A stored copy can drift from the probabilities it claims to summarise.
4. **Frozen dataclasses.** Alignment bugs thrive on mutable shared state.

Validation refuses, at construction: duplicate keys, NaN/inf on rows marked
valid, probabilities that don't sum to 1, and length mismatches. Each with a
message saying what to do instead.

## 4. Live evidence — the contract demonstrably fixes F1

`docs/program/evidence/F3_keyed_vs_positional.py` runs the **same data** and
the **same two deliberately perfect models** through both paths:

```
OLD PATH — OOFAligner, positional indices
  aligned rows          : 341
  first aligned key     : 0        <- claims bar 0
  agreement (2 PERFECT) : 0.0%

NEW PATH — PredictionPanel, source-bar keys
  aligned rows          : 341
  first aligned key     : 59       <- the real first shared bar
  agreement (2 PERFECT) : 100.0%

RESULT: KEYED CONTRACT FIXES F1
```

Same row count either way — which is exactly why the old bug was invisible.
The difference is *which bars* those rows refer to.

## 5. Tests

`tests/test_prediction_contract.py` — **18 tests, all passing**:

- `TestF1CannotRecur` — two perfect models agree 100% when keyed; positional
  keys now produce *visible* disagreement (a diagnostic that did not exist).
- `TestKeyIntegrity` — duplicate keys rejected; legacy sentinel in key space
  rejected; `reindex` marks absent keys invalid rather than shifting.
- `TestMissingnessIsAMask` — NaN allowed on invalid rows, rejected on valid
  ones; unnormalised scores rejected.
- `TestDerivedNeverDrifts` — including `margin`, which distinguishes
  `0.90 vs 0.05` from `0.35 vs 0.33` where max-probability cannot.
- `TestCombinabilityFailsClearly` — schema mismatch, OOF-vs-test mismatch, and
  disjoint keys each fail with an actionable reason.
- `TestCoreStaysLight` — subprocess check that no ML stack leaks in.

## 6. Failures encountered

`F3_keyed_vs_positional.py` first crashed with
`TypeError: unhashable type: 'numpy.ndarray'` — I passed `OOFResult` fields
positionally and that dataclass's field order differs from the keyword form I
had used earlier. Fixed by using keywords. No product defect.

One ruff error (`B905`, `zip()` without `strict=`) in my own new code; fixed.

## 7. Quantitative results

| Metric | Before | After |
|---|---|---|
| Agreement, 2 perfect models, mixed rank | **0.0%** | **100.0%** |
| First aligned key (true value 59) | 0 | **59** |
| Missingness signals in the contract | 4 (`-99`,`-999`,`-1`,NaN) | **1** (`valid`) |
| `INVALID_LABEL` definitions | 5 | 5 *(untouched — see R9)* |
| `src.core` import time | 0.38s | **0.37s**, heavy=[] |
| Tests in suite | 607 | **625** |

## 8. Unresolved risks

- **R1–R8 (carried).** See Stages 2–3.
- **R9 (new).** **The contract exists but nothing produces it yet.** The five
  `INVALID_LABEL` definitions and 92 `-99` uses are untouched; the old
  `OOFAligner` still runs in production paths. This stage deliberately built
  the contract *without* migrating producers — migration is Stage 15
  (inference emits `PredictionSet`) and Stage 17 (ensemble consumes it).
  **Do not claim F1 is fixed in the product.** It is fixed *in the contract*;
  the F1 xfail pins in `tests/test_oof_alignment_contract.py` remain
  xfailing, correctly.
- **R10 (new).** `PredictionPanel.align` uses a Python dict for the key join —
  O(N) but with interpreter overhead. Fine at current scale (hundreds of
  thousands of bars); if it appears in a hot loop at Stage 17, switch to
  `np.searchsorted` on sorted keys.

## 9. Instructions for the next agent (Stage 5 — preprocessing/transform interfaces)

1. **Re-run first:** full suite (expect **625 passed, 2 xfailed, 0 failed**),
   plus `PYTHONPATH=. .venv/bin/python docs/program/evidence/F3_keyed_vs_positional.py`
   which must exit 0.
2. **Do not remove the F1 xfail markers.** They must keep failing until
   Stage 17 migrates the real producers. If they XPASS before then, something
   changed the live aligner and the handoff must say what.
3. When you touch any producer of predictions, prefer emitting `PredictionSet`
   over the legacy tuple/DataFrame shapes — that is how R9 gets retired
   incrementally rather than in one risky sweep.
4. `PredictionSet.reindex()` is the safe replacement for every positional
   slice like `prices[:len(oof)]` (Phase 0 F5) — it marks absent keys invalid
   instead of silently borrowing a neighbour's value.
5. Stage 5 goal: scaler fit/apply as a declared, fold-scoped step. Phase 0
   flagged a global scaler fit preceding CV folds at
   `src/data/adapters/preparation.py:597` — verify that before changing it.
6. Still **UNKNOWN**: whether any ensemble beats its constituents and a
   baseline. Blocked on Stage 17 (producer migration) and Stage 12 (F7 test
   metrics for 3D/4D).
