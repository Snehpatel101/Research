"""Side-by-side: the positional aligner vs the keyed contract (Stage 4).

Same data, same two DELIBERATELY PERFECT models, same source bars.
The only difference is which coordinate system the predictions carry.

Run: PYTHONPATH=. .venv/bin/python docs/program/evidence/F3_keyed_vs_positional.py
"""

from __future__ import annotations

import numpy as np

from src.core.prediction_panel import PredictionPanel
from src.core.predictions import PredictionProvenance, PredictionSchema, PredictionSet
from src.data.adapters.alignment import OOFAligner, OOFResult

SEQ_LEN = 60
N_BARS = 400
CLASSES = (-1, 0, 1)

truth = (np.arange(N_BARS) % 3) - 1
bars_2d = np.arange(N_BARS)  # tabular sees every bar
bars_3d = np.arange(SEQ_LEN - 1, N_BARS)  # 60-bar window loses the first 59


def one_hot(labels):
    lut = {c: i for i, c in enumerate(CLASSES)}
    p = np.zeros((len(labels), 3), dtype=np.float64)
    for r, v in enumerate(labels):
        p[r, lut[int(v)]] = 1.0
    return p


print("=" * 70)
print("OLD PATH — OOFAligner, positional indices")
print("=" * 70)
# What the producers actually emit today: 0..n-1 into each model's OWN array.
old_2d = OOFResult(
    model_name="xgboost",
    predictions=truth[bars_2d],
    probabilities=one_hot(truth[bars_2d]),
    indices=np.arange(len(bars_2d)),
    fold_ids=np.zeros(len(bars_2d), dtype=int),
)
old_3d = OOFResult(
    model_name="lstm",
    predictions=truth[bars_3d],
    probabilities=one_hot(truth[bars_3d]),
    indices=np.arange(len(bars_3d)),
    fold_ids=np.zeros(len(bars_3d), dtype=int),
)
old = OOFAligner(n_classes=3).align([old_2d, old_3d])
old_agree = (old.predictions[:, 0] == old.predictions[:, 1]).mean()
print(f"  aligned rows            : {old.n_common}")
print(f"  first aligned key       : {old.common_indices[0]}")
print(f"  agreement (2 PERFECT)   : {old_agree:.1%}")

print()
print("=" * 70)
print("NEW PATH — PredictionPanel, source-bar keys")
print("=" * 70)


def make(name, bars):
    return PredictionSet(
        keys=bars,
        probabilities=one_hot(truth[bars]),
        valid=np.ones(len(bars), dtype=bool),
        schema=PredictionSchema.ternary(horizon=5),
        provenance=PredictionProvenance(model_name=name),
    )


panel = PredictionPanel.align([make("xgboost", bars_2d), make("lstm", bars_3d)])
new_agree = panel.agreement_matrix()[0, 1]
print(f"  aligned rows            : {panel.n_keys}")
print(f"  first aligned key       : {int(panel.keys[0])}")
print(f"  agreement (2 PERFECT)   : {new_agree:.1%}")
print()
print(panel.summary())

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  positional agreement : {old_agree:6.1%}   <- silently wrong")
print(f"  keyed agreement      : {new_agree:6.1%}   <- correct")
print(f"  first key, positional: {old.common_indices[0]:>3d} (claims bar 0)")
print(f"  first key, keyed     : {int(panel.keys[0]):>3d} (the real first shared bar)")

ok = new_agree == 1.0 and old_agree < 1.0 and int(panel.keys[0]) == SEQ_LEN - 1
print()
print("RESULT:", "KEYED CONTRACT FIXES F1" if ok else "UNEXPECTED — investigate")
raise SystemExit(0 if ok else 1)
