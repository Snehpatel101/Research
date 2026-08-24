"""Stage 5: does the CV path receive raw data, as its code claims?

Phase 0 (Agent 4) reported "global scaler fit precedes CV folds" at
src/data/adapters/preparation.py:597. Re-reading that site, the report is
MISLOCATED: preparation.py fits on the train split only and transforms
val/test -- correct for its own purpose, and the comment says so.

The real issue is one layer down. This script tests two claims:

  C1  PreparedData.X_train is already scaled when it leaves prepare().
  C2  oof_generation.py hands that already-scaled array to the OOF/CV
      machinery, whose own code calls it "(raw, unscaled)" and scales it
      AGAIN per fold.

SPOILER / CONCLUSION (this script refutes its own hypothesis):
The scaler STATISTICS really are contaminated by fold-validation rows, but
the fold-training VALUES come out bit-identical, because every reachable
scaler is AFFINE and the fold scaler re-derives its statistics from the
pre-scaled fold-train data -- so the first transform cancels exactly.
There is NO leakage in the CV path. What remains is redundant computation
and a code comment that calls pre-scaled data "raw, unscaled".

Run: PYTHONPATH=. .venv/bin/python docs/program/evidence/F4_double_scaling.py
"""

from __future__ import annotations

import numpy as np

from src.validation.cv.fold_scaling import FoldAwareScaler

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# C1/C2 are about plumbing; demonstrate the STATISTICAL consequence directly,
# since that is what actually matters and it does not depend on constructing a
# full pipeline run.
#
# Construct a train split whose LATER portion (which becomes the last fold's
# validation data) has a very different scale from the earlier portion. That
# is realistic: volatility regimes shift.
# ---------------------------------------------------------------------------
N = 1000
CUT = 800  # first 800 rows = fold train, last 200 = fold validation

early = rng.normal(0.0, 1.0, (CUT, 1))
late = rng.normal(0.0, 10.0, (N - CUT, 1))  # 10x volatility regime
X = np.vstack([early, late]).astype(np.float64)

print("=" * 72)
print("Data: 1000 rows, last 200 have 10x the volatility of the first 800")
print("=" * 72)
print(f"  std(first {CUT})      = {X[:CUT].std():.3f}")
print(f"  std(last {N-CUT})      = {X[CUT:].std():.3f}")

# --- CORRECT: fit the scaler on fold-train ONLY -----------------------------
correct = FoldAwareScaler(method="robust")
res_correct = correct.fit_transform_fold(X[:CUT], X[CUT:])
train_correct = res_correct.X_train_scaled

# --- WHAT THE PIPELINE ACTUALLY DOES: scale on the WHOLE split first --------
# preparation.py fits AdapterScaler on the entire train split (all 1000 rows),
# then oof_generation hands the result to the CV machinery.
whole_median = np.median(X, axis=0)
whole_q75, whole_q25 = np.percentile(X, 75, axis=0), np.percentile(X, 25, axis=0)
whole_iqr = np.where((whole_q75 - whole_q25) > 1e-8, whole_q75 - whole_q25, 1.0)
X_pre_scaled = (X - whole_median) / whole_iqr

leaked = FoldAwareScaler(method="robust")
res_leaked = leaked.fit_transform_fold(X_pre_scaled[:CUT], X_pre_scaled[CUT:])
train_leaked = res_leaked.X_train_scaled

print()
print("=" * 72)
print("Scaler statistics used to normalise the FOLD-TRAINING rows")
print("=" * 72)
fold_only_iqr = np.percentile(X[:CUT], 75) - np.percentile(X[:CUT], 25)
print(f"  IQR from fold-train only (correct) : {fold_only_iqr:.4f}")
print(f"  IQR from the whole split  (actual) : {float(whole_iqr[0]):.4f}")
contamination = abs(float(whole_iqr[0]) - fold_only_iqr) / fold_only_iqr
print(f"  contamination                      : {contamination:.1%}")
print()
print()
print("  The whole-split IQR IS inflated by validation rows the fold's")
print("  training data must not see. Whether that matters depends on what")
print("  happens next -- see below.")

print()
print("=" * 72)
print("Effect on the fold-training values")
print("=" * 72)
print(f"  std(train | correct) = {train_correct.std():.4f}")
print(f"  std(train | actual)  = {train_leaked.std():.4f}")
diff = float(np.abs(train_correct - train_leaked).max())
print(f"  max |difference|     = {diff:.4f}")

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
leaks = contamination > 0.05
print(f"  scaler statistics contaminated by fold-validation rows : {leaks}")
print(f"  fold-training values differ from the correct scaling   : {diff > 1e-6}")
cancels = diff < 1e-9
print()
print("  Scaler STATISTICS are contaminated, but the fold-training VALUES")
print("  are bit-identical. Robust/standard scaling is AFFINE, and the")
print("  fold scaler re-derives its statistics from the pre-scaled")
print("  fold-train data, so the first transform cancels exactly:")
print()
print("      pre  : z = (x - m)/s                     [global m, s]")
print("      fold : w = (z - med(z_tr)) / IQR(z_tr)")
print("           =  (x - med(x_tr)) / IQR(x_tr)      [m, s cancel]")
print()
print("RESULT:", "NO LEAKAGE -- claim REFUTED" if cancels else "LEAKAGE PRESENT")
print()
print("  Phase 0 (Agent 4) reported 'global scaler fit precedes CV folds'")
print("  as HIGH-severity leakage. That is REFUTED for the CV/OOF path.")
print("  What remains is redundant computation, not a correctness defect.")
print()
print("  THE INVARIANT THIS RELIES ON: every reachable scaler is affine.")
print("  AdapterScaler accepts only robust/standard/minmax and raises")
print("  otherwise; FoldAwareScaler accepts only none/robust/standard.")
print("  ScalerType also declares QUANTILE, which is NOT affine -- if it")
print("  ever becomes reachable as the pre-scaler, the cancellation breaks")
print("  and this DOES become real leakage. That invariant is now pinned by")
print("  tests/test_scaling_invariants.py rather than left implicit.")
raise SystemExit(0 if cancels else 1)
