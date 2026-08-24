# Handoff — Stage 2: Architectural Cleanup (ungated portion)

**Date:** 2026-08-24
**Predecessor:** Stage 1 (`01_instrumentation.md`)
**Branch:** claude/timeseries-ml-ensemble-upgrade-t7gutw

---

## 1. Verification of predecessor

Re-executed rather than re-read, per the standing rule.

| Stage 1 claim | How I checked | Verdict |
|---|---|---|
| Controls + F1 pins + lookahead audit all green | Ran those 3 files: 42 passed, exactly 2 xfail, 0 xpass | **CONFIRMED** |
| F1 still reproduces | xfail pins still fail for the stated reason | **CONFIRMED** |
| Suite 605p/2xf/0f | Re-ran full suite | **CONFIRMED** |

## 2. Scope note — what was NOT done

The ~7,000-LOC deletion batch (`InferenceOrchestrator`, special-mode bundles,
`server.py` + monitoring, `second_level.py`, `meta_selection.py`) is **still
gated on user decision #1** and was not touched. Only `DECISIONS.md` items 6
and 7 — single-symbol, zero-consumer, already recommended by the previous
team — were executed.

## 3. Findings

- **"Zero consumers" needed a stricter test than grep.** `grep -c AdapterResult`
  returns 29 hits outside `src/core`, which looks alive. Those are all the
  *canonical* class from `src/data/adapters/base.py` — a different class with
  the same name. The correct test is by **import source**: symbols actually
  imported `from src.core.interfaces` are `PredictionResult` (×5),
  `OOFResult` (×2), `OOFPredictionProtocol` (×2), `TrainingResult` (×1).
  `AdapterResult` and `AdapterContract`: **zero**.
- **The two `AdapterResult` classes are not interchangeable**, so the
  `DECISIONS.md` #6 "keep them in sync" framing was never achievable:
  `X`/`y` vs `data`/`labels`, and `n_samples` is a **field** in one and a
  **property** in the other.
- **`TrainingResult` was worse than DECISIONS #7 described.** Not merely
  mis-annotated: its only consumer was `self._cached_training_result`, an
  attribute that is **declared once, never assigned, never read**.
- **A latent name collision:** an `AdapterType` TypeVar in `interfaces.py`
  (bound to `AdapterContract`) coexisted with an unrelated `AdapterType`
  StrEnum in `core/types.py`. `src/core/__init__` re-exports the **enum**, so
  the same name meant different objects depending on import path.

## 4. Architectural decisions

- **Deleted the legacy `AdapterResult` without a compatibility re-export.**
  `DECISIONS.md` #6 recommended re-exporting the canonical class from
  `src.core`. I deliberately did not: a `core -> data` import deepens exactly
  the import cycle Stage 3 must break, and there are zero consumers to serve.
  Deviation recorded here rather than silently taken.
- **Kept the `ModelContract`/`DataContract` re-export** in `interfaces.py`
  (see §6 — I removed it by accident and restored it).

## 5. Implementation changes

| File | Change | Why |
|---|---|---|
| `src/core/interfaces.py` | Removed legacy `AdapterResult`, `AdapterContract`, `TrainingResult`, `AdapterType` TypeVar; updated docstring + `__all__` | DECISIONS #6/#7; all zero-consumer |
| `src/core/__init__.py` | Dropped the three re-exports from imports, usage docstring, `__all__` | Same |
| `src/factory.py` | Removed dead `_cached_training_result` attr + `TYPE_CHECKING` import; corrected two false `PipelineRunner` docstring claims | Phase 0 proved `PipelineRunner` is never imported here |
| `tests/test_phases_1_3.py` | Rewrote `test_higher_tf_shift` behaviourally | See §6 |

Net: −137 lines in `src/`.

## 6. Failures encountered

**I broke the suite myself, and the mechanism is worth recording.**

`tests/test_phases_1_3.py::test_higher_tf_shift` failed at ~86% of a full run
while passing in isolation. Root cause: **I edited `src/factory.py` while the
suite was running.** `inspect.getsource()` resolves the code object's line
numbers against the file *on disk at call time*, so after my docstring
insertion it returned the body of `_needs_multi_stream` instead of
`_generate_additional_dfs`. The assertion message proves it.

Not a regression — `resampled.shift(1).dropna()` is intact at `factory.py:660`.

Two lessons, both acted on:

1. **Process:** never edit source during a running suite. The clean re-run is
   the number reported in §7.
2. **Test quality:** this is concrete evidence for `DECISIONS.md` #11. A
   source-grep test breaks on unrelated line moves and passes on text
   coincidence. Its "functional" half was **also** tautological — it
   re-implemented resample+shift inside the test and asserted pandas works,
   never calling `_generate_additional_dfs` at all.

**Rewritten behaviourally.** It now calls the real method (via a
`SimpleNamespace` stub for `self`, so no full `ExperimentConfig` is needed)
and asserts the property that matters: the value at bar T must come from the
window *before* T, never from T's own still-forming bar.

**Mutation-tested.** With `shift(1)` removed:

```
AssertionError: Bar at 2024-01-01 09:30:00 carries its own window's
closing value (100.04) — that is lookahead.
```

Restored → passes. The test provably catches the defect it guards.

## 7. Tests executed

```
tests/test_phases_1_3.py    14 passed
ruff check src/ tests/      All checks passed
black --target-version py311  clean
```

Full suite (clean run, no concurrent edits):

```
605 passed · 2 xfailed · 0 failed · 0 xpassed   (607 total)
```

Identical to Stage 1 — the removals changed no behaviour.

## 8. Live executions performed

Import smoke across the whole package after removal:
`src.core`, `src.core.interfaces`, `src.models`, `src.factory`,
`src.data.adapters`, `src.inference` — all import. `AdapterType` resolves to
`EnumType` (the intended enum). All three removed symbols confirmed absent
via `hasattr`.

## 9. Quantitative results

| Metric | Before | After |
|---|---|---|
| `AdapterResult` definitions | 2 (incompatible) | **1** |
| Dead classes in `core/interfaces.py` | 3 | **0** |
| Source-grep assertions in the suite | 26 | **24** |
| Tautological `test_higher_tf_shift` | 1 | 0 (behavioural, mutation-verified) |

## 10. Regressions checked

Full suite re-run clean. Import smoke across all top-level packages.

## 11. Unresolved risks

- **R1 (carried).** 24 source-grep assertions remain, including two in this
  same class (`test_hp_tuning_embargo_wired`). Replace opportunistically.
- **R2 (carried).** 164 swallow-without-reraise handlers → Stage 14.
- **R3 (carried).** Controls don't exercise `FeatureEngineer` → needs F15
  (Stage 6).
- **R4 (carried).** py311/py312 config inconsistency.
- **R5 (new).** Phantom `"mlp"` entries in
  `src/models/config/data_requirements.py:231` and
  `src/data/pipeline/config/feature_sets/core.py:64` — no such model class.
  **Deliberately deferred to Stage 9**, where registry/table parity can be
  enforced structurally by a test rather than fixed by hand here.
- **R6 (new).** `src.core` still imports from `contracts`; the SCC is
  untouched. Stage 3.

## 12. Instructions for the next agent (Stage 3 — package/module organisation)

1. **Re-run first:** full suite. Expect **605 passed, 2 xfailed, 0 failed** —
   unchanged from Stage 1. (I predicted 604 while writing this, reasoning that
   collapsing source-grep assertions would drop a test. Wrong: I removed two
   *assertions* inside one test, not two tests. Corrected against the actual
   run rather than left to mislead you.)
2. **Do not remove the F1 xfail markers.** Still pinned for Stage 17.
3. Stage 3 goal: `import src.core.contracts` **without** pulling in
   torch/xgboost/lightgbm/catboost. Measure before/after with
   `python -X importtime`.
4. **Known-fragile:** `src/models/__init__.py` imports every model family for
   registration side effects. Lazy registration needs an `ensure_registered()`
   guard at entry points, or parallel workers will see an empty registry.
   ~40 function-level "avoid circular import" workarounds exist — they are the
   map of the cycle.
5. **Do not add any `core -> data` or `core -> models` import** to restore a
   convenience re-export. That is why §4 declined the `AdapterResult`
   re-export.
6. Still **UNKNOWN**: whether any ensemble beats its constituents and a
   baseline. Blocked on F1 (Stage 17) and F7 (Stage 12).
