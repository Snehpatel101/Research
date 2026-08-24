"""Empirical demonstration of the heterogeneous-ensemble alignment defect.

Claim under test (Agent 3): OOF producers emit POSITIONAL indices into each
model's own prepared array, while adapters compute SOURCE-BAR indices. The
OOFAligner intersects the positional ones as if they were a shared key, so a
2D model's row i is aligned with a 3D model's row i -- which is source bar
i + seq_len - 1. Silent, no crash.

This script does not simulate the bug; it exercises the real adapter and the
real aligner.
"""
import numpy as np
import pandas as pd

from src.data.adapters.sequence import SequenceAdapter
from src.data.adapters.alignment import OOFAligner, OOFResult

N = 400
SEQ_LEN = 60
rng = np.random.default_rng(0)

# A deliberately trivial "signal": the label at bar t is a function of t alone.
# That lets us detect misalignment by value, not by inspection.
df = pd.DataFrame(
    {
        "f0": rng.normal(size=N),
        "f1": rng.normal(size=N),
        "label": (np.arange(N) % 3) - 1,  # bar t -> class (t%3)-1
    }
)

# ---------------------------------------------------------------- adapter path
adapter = SequenceAdapter(sequence_length=SEQ_LEN, feature_columns=["f0", "f1"],
                          label_column="label")
res = adapter.transform(df)
oi = res.original_indices
print(f"[adapter] n_source_bars      = {N}")
print(f"[adapter] n_windowed_samples = {len(oi)}   (expected {N - SEQ_LEN + 1})")
print(f"[adapter] original_indices[:5] = {oi[:5]}  <- SOURCE-BAR coords")
print(f"[adapter] offset of window 0   = {oi[0]}   (expected seq_len-1 = {SEQ_LEN-1})")
assert oi[0] == SEQ_LEN - 1, "adapter indices are not source-bar coords"

# ------------------------------------------------------- what the OOF producer does
# oof_generation.py:383 -> valid_indices = np.where(~np.isnan(oof_preds))[0]
# oof_preds has length == n_samples of the model's OWN prepared array.
n_seq = len(oi)
producer_indices_3d = np.arange(n_seq)          # positional, 0..n_seq-1
producer_indices_2d = np.arange(N)              # positional, 0..N-1
print(f"\n[producer] 3D emits indices[:5] = {producer_indices_3d[:5]}  <- POSITIONAL")
print(f"[producer] 2D emits indices[:5] = {producer_indices_2d[:5]}  <- POSITIONAL")
print(f"[producer] adapter's真 index for 3D row 0 = {oi[0]}, producer says 0"
      .replace("真", " true "))

# ------------------------------------------------------------------ the aligner
# Give each model a probability vector that ENCODES the source bar it believes
# it is predicting, so misalignment is visible numerically.
def one_hot(labels, n_classes=3):
    p = np.zeros((len(labels), n_classes), dtype=np.float32)
    p[np.arange(len(labels)), labels + 1] = 1.0
    return p

# 2D model predicts the true label of source bar i (perfect model).
y_2d = df["label"].values[producer_indices_2d]
# 3D model predicts the true label of source bar oi[j] (also a perfect model).
y_3d = df["label"].values[oi]

oof_2d = OOFResult(model_name="xgboost", predictions=y_2d,
                   probabilities=one_hot(y_2d), indices=producer_indices_2d,
                   fold_ids=np.zeros(len(y_2d), dtype=int))
oof_3d = OOFResult(model_name="lstm", predictions=y_3d,
                   probabilities=one_hot(y_3d), indices=producer_indices_3d,
                   fold_ids=np.zeros(len(y_3d), dtype=int))

aligned = OOFAligner(n_classes=3).align([oof_2d, oof_3d])
print(f"\n[aligner] n_common = {aligned.n_common}")

# Both models are PERFECT. If alignment were correct, their aligned predictions
# would agree everywhere. Measure disagreement.
p2 = aligned.predictions[:, 0]
p3 = aligned.predictions[:, 1]
agree = (p2 == p3).mean()
print(f"[aligner] agreement between two PERFECT models = {agree:.1%}")
print(f"[aligner] expected if aligned correctly        = 100.0%")

# Show the actual shift
common = aligned.common_indices
print(f"\n[diagnosis] aligned row 0 -> 2D thinks source bar {common[0]}, "
      f"3D actually predicted source bar {oi[common[0]]}")
print(f"[diagnosis] systematic shift = {oi[common[0]] - common[0]} bars")

if agree < 1.0:
    print("\nRESULT: CONFIRMED -- two perfect models disagree after alignment.")
    print("        The ensemble is learning from misaligned rows, silently.")
else:
    print("\nRESULT: NOT REPRODUCED -- alignment agreed.")
