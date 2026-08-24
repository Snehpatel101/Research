# Handoff — Stage 3: Package / Module Organisation

**Date:** 2026-08-24
**Predecessor:** Stage 2 (`02_architectural_cleanup.md`)

---

## 1. Verification of predecessor

| Stage 2 claim | How I checked | Verdict |
|---|---|---|
| Suite 605p/2xf/0f | Re-ran controls + F1 pins; full suite (§7) | **CONFIRMED** |
| Removals changed no behaviour | Full import smoke, full suite | **CONFIRMED** |
| F1 still pinned | 2 xfail, 0 xpass | **CONFIRMED** |

## 2. The stage goal was ALREADY MET — and DECISIONS.md #10 is overstated

`DECISIONS.md` #10 claims `src/config ⇄ src/models ⇄ src/data` form one SCC
such that "importing *any* of them loads torch+xgboost+lightgbm+catboost+
optuna (~4–5s)". **Measured, that is false for `src.core`:**

| Module | Time | Heavy deps pulled |
|---|---|---|
| `src.core` | **0.39s** | **none** |
| `src.core.contracts` | **0.42s** | **none** |
| `src.core.types` | **0.38s** | **none** |
| `src.config` | 3.74s | torch, xgb, lgbm, catboost, optuna, sklearn, numba |
| `src.data` | 3.96s | (same) |
| `src.models` | 3.87s | (same) |
| `src.validation` | 3.47s | (same) |
| `src.inference` | 3.66s | (same) |
| `src.factory` | 3.46s | (same) |

So the planned gate — "`import src.core.contracts` without the ML stack" —
**already passed before I changed anything.** Phase 114's "safe quick wins"
evidently fixed `src/core`. The SCC is real, but it is
`config ⇄ data ⇄ optimization ⇄ validation ⇄ inference ⇄ models`, with
`core` outside it.

## 3. What I changed, and why it is architectural rather than an import hack

**`src/models/config/data_requirements.py` → `src/core/model_requirements.py`
(831 lines).**

That module is **pure data**: stdlib + `src.core.types`, nothing else. It
declares per-model capability requirements. It lived inside `src.models` by
accident of packaging, and because Python executes a package's `__init__` when
you import any submodule, `from src.models.config.data_requirements import ...`
dragged in `src/models/__init__.py`, which eagerly imports every model family →
torch.

This is not just an import fix: **per-model capability declaration belongs in
`core`**, and Stage 8 will make exactly this data the single source of truth
for `ModelCapabilities`. The move is a down payment on that, not a workaround.

Also:
- `src/config/models/__init__.py` and `src/config/__init__.py`: PEP 562 lazy
  `__getattr__` for the model-config facade re-exports.
- `ModelRegistry.ensure_registered()`: an explicit alternative to
  registration-by-import-side-effect.

## 4. Honest result: partial. The goal was NOT fully achieved.

`src.config` is **still 3.89s**. My changes removed the direct
`config → models` and `data → models` edges, but `src.config` still reaches
torch through a five-hop chain:

```
src.config
  -> src.config.pipeline
  -> src.data.pipeline.config.barriers_config      (executes src/data/__init__)
  -> src.data.pipeline.stages.validation
  -> src.optimization.feature_selection
  -> src.optimization.five_dimension_objective     <-- DEAD CODE
  -> src.validation.deflated_sharpe                (executes src/validation/__init__)
  -> src.inference.backtesting
  -> src.models  -> torch
```

**Finding worth escalating: the import cycle is partly held together by dead
code.** `five_dimension_objective` is the 5-D Optuna island from
`DECISIONS.md` #5 — zero live consumers, kept alive only by tests. It sits on
the critical path from config to torch.

Edge count into `src.models` from outside it, when importing `src.config`:
**20 → 15.** The remaining 15 are all `validation.cv.* → models.registry/base`
and `inference.* → models.base`, which are *legitimate* (those modules really
do need models) — they are only a problem because `src/validation/__init__`
and `src/inference/__init__` import them eagerly.

**I stopped here deliberately.** Chasing the remaining edges means either
lazifying four more package `__init__`s or deleting the dead island — the
first is the 3–5 day refactor `DECISIONS.md` scoped, the second is gated on
your decision. Continuing would have been sprawl, and the mission warns
against cosmetic churn.

## 5. A correction to my own work

I initially reported `ensure_registered()` as working by clearing
`ModelRegistry._models` and calling it. **That test was invalid** — clearing a
dict does not un-import a module, so the guard's `import src.models` was a
no-op.

Re-tested in a fresh process, the real finding is stronger and less flattering:
importing `src.models.registry` **alone** already loads torch and registers 61
entries, because it is a submodule and `src/models/__init__.py` always
executes. **`ensure_registered()` is therefore a no-op today.** It is correct
scaffolding for when `__init__` stops eagerly importing families, but it buys
nothing right now, and I am not claiming otherwise.

## 6. Implementation changes

| File | Change |
|---|---|
| `src/core/model_requirements.py` | **moved** from `src/models/config/data_requirements.py` (git mv, history preserved) |
| `src/models/config/__init__.py` | re-exports from the new canonical location (in-package callers unaffected) |
| `src/config/pipeline/__init__.py`, `src/data/pipeline/config/{__init__,multi_model}.py` | repointed to `src.core.model_requirements` |
| `src/config/models/__init__.py` | eager facade → PEP 562 lazy |
| `src/config/__init__.py` | lazy `__getattr__` for 6 model-config names |
| `src/models/registry.py` | `+ ensure_registered()` |
| `src/config/ensemble.py`, `src/models/training/{artifacts,features}.py` | stale path references in comments |

## 7. Tests executed

`ruff check src/ tests/` — All checks passed. `black` — clean.
Full suite: see §12.

Import smoke: all of `src.config/core/data/models/validation/inference/factory`
import. `MODEL_DATA_REQUIREMENTS` is the **same object** via both
`src.core.model_requirements` and `src.models.config` (`A is B` → True,
24 entries).

## 8. Quantitative results

| Metric | Before | After |
|---|---|---|
| `import src.core.contracts` heavy deps | none | none (already met) |
| Edges into `src.models` when importing `src.config` | 20 | **15** |
| `config → models` direct edges | 2 | **0** |
| `data → models` direct edges | 2 | **0** |
| `import src.config` wall time | 3.74s | 3.89s (**unchanged** — see §4) |

The wall time did not improve. Reporting it as a non-result rather than
burying it: the direct edges are gone, but the transitive chain still lands on
torch, so a user sees no difference yet.

## 9. Unresolved risks

- **R1–R4 (carried)** from Stage 2.
- **R5 (carried).** Phantom `"mlp"` entries — now at
  `src/core/model_requirements.py` (moved) and
  `src/data/pipeline/config/feature_sets/core.py:64`. Still Stage 9.
- **R7 (new).** `ensure_registered()` is inert until `src/models/__init__.py`
  stops eager-importing families (§5). Do not rely on it yet.
- **R8 (new).** The SCC is partly load-bearing on dead code
  (`five_dimension_objective`). Deleting it — `DECISIONS.md` #5 — would remove
  a chain link *and* ~3,500 lines. Gated on the user.

## 10. Instructions for the next agent (Stage 4 — canonical data contracts)

1. **Re-run first:** full suite. Expect **605 passed, 2 xfailed, 0 failed**.
2. **Do not remove the F1 xfail markers.** They flip at Stage 17.
3. Stage 4 builds `PredictionSet` — keyed by source-bar/timestamp, explicit
   `valid` mask, retiring the `-99/-999/-1/NaN` sentinels. This is the
   precondition for fixing F1.
4. **Put it in `src/core/`.** Verified this stage that `src.core` imports in
   0.38s with zero heavy deps — keep it that way. Do **not** import from
   `src.data` or `src.models` inside it.
5. `src/core/model_requirements.py` is now the canonical home for per-model
   capability data. Stage 8 should build `ModelCapabilities` there or beside
   it, not back inside `src.models`.
6. Still **UNKNOWN**: whether any ensemble beats its constituents and a
   baseline. Blocked on F1 (Stage 17) and F7 (Stage 12).
