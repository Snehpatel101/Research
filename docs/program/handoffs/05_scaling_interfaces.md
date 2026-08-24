# Handoff — Stage 5: Preprocessing / Transform Interfaces

**Date:** 2026-08-24
**Predecessor:** Stage 4 (`04_prediction_contract.md`)

---

## 1. Verification of predecessor

| Stage 4 claim | How I checked | Verdict |
|---|---|---|
| Suite 623p/2xf/0f (625) | Re-ran full suite | **CONFIRMED** |
| Keyed alignment gives 100% agreement | Re-ran `F3_keyed_vs_positional.py`, exit 0 | **CONFIRMED** |
| F1 not yet fixed in the product (R9) | F1 pins still xfail | **CONFIRMED** |
| `src.core` stays light | 0.37s, heavy=[] | **CONFIRMED** |

## 2. Headline: a Phase 0 finding is REFUTED

Phase 0 (Agent 4) listed, as HIGH severity:

> global scaler fit precedes CV folds (`adapters/preparation.py:597`)

**That is wrong, and I nearly "fixed" a non-bug.** Two errors in the original
report:

1. **Mislocated.** `preparation.py:598` fits `AdapterScaler` on
   `train_result.X` — the train split only — then transforms val/test. That is
   correct for its purpose, and the code says so.
2. **The real plumbing issue is one layer down, and is benign.**
   `oof_generation.py:189` passes `prepared.X_train` (already scaled) into the
   CV machinery, whose own comment at `oof_core.py:242` called it
   "(raw, unscaled)". So folds scale already-scaled data.

I hypothesised this caused preprocessing leakage — each fold's training rows
normalised using statistics that saw that fold's validation rows — and built
`docs/program/evidence/F4_double_scaling.py` to prove it.

**The evidence refuted my own hypothesis.**

## 3. What the evidence actually shows

Constructed the worst realistic case: 1000 train rows whose last 200 (the
fold's validation data) carry **10× the volatility** of the first 800.

```
IQR from fold-train only (correct) : 1.3588
IQR from the whole split  (actual) : 1.7347
contamination                      : 27.7%

std(train | correct) = 0.7361
std(train | actual)  = 0.7361
max |difference|     = 0.0000
```

The scaler **statistics** are contaminated by 27.7%. The fold-training
**values** are bit-identical.

The reason is algebraic. Every reachable scaler is **affine**, and the fold
scaler re-derives its statistics from the pre-scaled fold-train data, so the
first transform cancels exactly:

```
pre  : z = (x - m) / s                     [global m, s]
fold : w = (z - med(z_tr)) / IQR(z_tr)
     =  (x - med(x_tr)) / IQR(x_tr)        [m and s cancel]
```

**Conclusion: no preprocessing leakage in the CV/OOF path.** What remains is
redundant computation and a comment that lied.

## 4. But the refutation is CONDITIONAL — and that is the real work

The safety depends on an invariant nobody had written down:

> every reachable scaler is affine

Verified: `FoldAwareScaler` accepts only `none`/`robust`/`standard`;
`AdapterScaler` accepts only `robust`/`standard`/`minmax` — and **both refuse
at construction**, not at fit time, which is stronger than I assumed.

But `ScalerType` (`src/config/data.py:37`) also declares **`QUANTILE`**, a
rank transform that is **not** affine. It is unreachable today. If anyone
wires it in as the pre-scaler, the cancellation breaks and the 27.7%
contamination becomes genuine leakage.

**So Stage 5's deliverable is not a fix — it is turning an accidental,
undocumented invariant into an enforced one.**

## 5. Implementation changes

| File | Change | Why |
|---|---|---|
| `tests/test_scaling_invariants.py` | **new**, 8 tests | Pins the affine-cancellation invariant and the reachable-scaler set |
| `src/validation/cv/oof_core.py` | Replaced the `(raw, unscaled)` comment with the truth + proof pointers | The comment is what made this look like leakage |
| `docs/program/evidence/F4_double_scaling.py` | **new** | Reproducible refutation |

**No production behaviour changed.** Deliberate: there was no defect to fix.

The tests are built to fail loudly in the right direction:
- `TestAffineCancellation` — cancellation holds for train *and* validation
  rows, for `robust` and `standard`.
- `TestScalerSetStaysAffine` — both scalers still refuse non-affine methods.
- `TestNonAffineWouldBreakIt` — a `QuantileTransformer` pre-scale demonstrably
  does **not** cancel. This tests the *reasoning*, so the guards above are
  meaningful rather than arbitrary.

## 6. Failures encountered

Two of my own tests failed first time: I expected the non-affine rejection at
`fit` time, but both scalers reject at `__init__`. Corrected the tests to
assert the stronger real behaviour.

## 7. Quantitative results

| Metric | Value |
|---|---|
| Scaler-statistic contamination (10× regime shift) | 27.7% |
| Effect on fold-training values | **0.0000** (bit-identical) |
| Phase 0 findings refuted this stage | **1** (Agent 4, "scaler leakage", HIGH) |
| Production behaviour changed | **none** |
| New invariant tests | 8 |

## 8. Unresolved risks

- **R1–R10 (carried).** See Stages 2–4. R9 (contract built, producers not
  migrated) remains the big one.
- **R11 (new).** Double scaling is wasted work on every OOF fold — two full
  passes over the fold matrices. Harmless numerically; a performance item, not
  a correctness one. Do **not** "fix" it by removing the pre-scaling without
  checking who else consumes `PreparedData.X_train` already-scaled (the
  train/val/test path needs it).
- **R12 (new).** `ScalerType.QUANTILE` is declared but unreachable. Either
  wire it deliberately (and then fix the pre-scaling first) or delete it from
  the enum. Leaving a non-affine option one line away from reachable is a
  latent trap.

## 9. Instructions for the next agent (Stage 6 — feature interfaces)

1. **Re-run first:** full suite (expect **631 passed, 2 xfailed, 0 failed**),
   plus `F3_keyed_vs_positional.py` and `F4_double_scaling.py`, both exit 0.
2. **Do not remove the F1 xfail markers.**
3. **Treat Phase 0 findings as hypotheses, not facts.** This stage refuted
   one that had sat in the reconciliation table as HIGH severity for four
   stages. Re-verify F15 (MLFactory discards feature/MTF config) before acting
   on it — that is Stage 6's central claim.
4. Stage 6 target is F15: `factory.py:753` constructs
   `FeatureEngineer(input_dir, output_dir)` with no config, while
   `features/run.py:242` threads `timeframe`/`enable_mtf`. Confirm both sites
   before changing either.
5. Fixing F15 unblocks a cheap feature set, which unblocks the fast test tier
   Agent 6 designed — and lets the control datasets exercise the real
   `FeatureEngineer` instead of a hand-rolled featurizer (risk R3).
6. Still **UNKNOWN**: whether any ensemble beats its constituents and a
   baseline. Blocked on Stage 17 and Stage 12.
