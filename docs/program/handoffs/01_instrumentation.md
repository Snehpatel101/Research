# Handoff — Stage 1: Instrumentation & Baseline Hardening

**Date:** 2026-08-24
**Predecessor:** Phase 0 reconciliation (`docs/program/RECONCILIATION.md`)
**Branch:** claude/timeseries-ml-ensemble-upgrade-t7gutw

---

## 1. Verification of predecessor

Phase 0 was my own work, so the duty here is to re-execute rather than re-read.

| Phase 0 claim | How I checked | Verdict |
|---|---|---|
| Suite is 593p/4f/597 | Re-ran full suite | **CONFIRMED** |
| 4 failures = 1 root cause, pandas 3 int/float | Fixed that one site; all 32 tests in the file pass | **CONFIRMED** |
| `factory.py:122-123` reads `val_f1` while base models store `macro_f1` | Read source; exercised the formatter | **CONFIRMED** |
| Adapter emits source-bar coords `[59,60,...]` | Re-ran F1 repro + new test | **CONFIRMED** |
| OOF emits positional coords `[0,1,...]` | Re-ran F1 repro | **CONFIRMED** |
| macro-F1 gains on pure noise | Re-ran control script | **CONFIRMED** (+0.1684) |

## 2. What I inspected

`src/validation/lookahead_audit.py`, `src/factory.py`, `tests/conftest.py`,
`src/data/adapters/{sequence,alignment}.py`.

## 3. Findings

- **F13 was worse than reported.** Phase 0 attributed the crash to the
  `random` corruption method. Reading the full function, the `nan` method has
  the *same* defect (NaN cannot live in an int64 column under pandas 3); it
  simply had no failing test. Fixed both. `shuffle` is genuinely safe —
  a permutation preserves dtype — and was left alone.
- **F16's `.get(key, 0.0)` default is the actual defect**, not the wrong key
  name. A wrong key with no default would have raised on day one. The silent
  0.0 is what made a broken summary survive.

## 4. Architectural decisions

- **Controls live in `tests/controls.py`, not `conftest.py`.** They are
  importable by evidence scripts and future stages, not just pytest, and
  depend on numpy/pandas only — usable without the ML stack.
- **Controls deliberately hand-roll their features** rather than calling
  `FeatureEngineer`. An instrument used to audit the machinery must not
  depend on that machinery. (Also unavoidable today: F15 means `MLFactory`
  cannot be asked for a cheap feature set.)
- **F1 pinned as `xfail(strict=True)`, not fixed here.** The real fix needs
  the keyed `PredictionSet` contract from Stage 4; a local patch to
  `oof_generation.py` would fix one producer and leave the other two.
  Strict mode converts XPASS into a suite failure, so the marker cannot
  outlive the bug.
- **`_format_model_metrics` returns `n/a`, never `0.0`.** A missing metric
  must look missing.

## 5. Implementation changes

| File | Change | Why |
|---|---|---|
| `src/validation/lookahead_audit.py` | `_widen_to_float()` helper; applied to `nan` + `random` paths | F13 — auditor crashed on any integer column |
| `src/factory.py` | `_first_metric` / `_format_model_metrics`; summary uses them | F16 — printed `F1=0.0000` regardless of truth |
| `tests/controls.py` | **new** — noise/signal generators, causal featurizer, temporal split, majority baseline | F6 — no baseline existed anywhere |
| `tests/test_controls.py` | **new** — 7 tests validating the instrument + pinning the metric ruling | Falsifiability for all later claims |
| `tests/test_oof_alignment_contract.py` | **new** — 1 passing guard + 2 strict-xfail pins | F1 regression pin |

## 6. Tests executed

```
tests/test_lookahead_audit.py    32 passed   (was 4 failed)
tests/test_controls.py            7 passed   (new)
tests/test_oof_alignment_contract.py  1 passed, 2 xfailed (new, as designed)
ruff check src/ tests/           All checks passed
black --target-version py311     clean
```

Full suite after Stage 1:

```
605 passed · 2 xfailed · 0 failed · 0 xpassed   (607 total)
```

Baseline was 593 passed / 4 failed / 597 total. Net: +10 tests, −4 failures,
and the 2 xfails are the deliberate F1 pins.

## 7. Live executions performed

- `docs/program/evidence/F1_oof_misalignment_repro.py` — re-run, still
  reproduces: two perfect models agree **0.0%**, 59-bar shift.
- `docs/program/evidence/F2_control_datasets.py` — re-run, `CONTROLS VALID`,
  all five assertions pass.

## 8. Failures encountered

`black` refused with *"Python 3.11 cannot parse code formatted for Python
3.12"* — `pyproject.toml` sets `target-version = ['py311','py312']` but the
venv is 3.11. Worked around with `--target-version py311`. **Unresolved
inconsistency for a later stage:** `[tool.ruff] target-version = "py312"` and
`[tool.mypy] python_version = "3.12"` while `requires-python = ">=3.11"` and
CI/dev here is 3.11.

## 9. Quantitative results

| Metric | Before | After |
|---|---|---|
| Suite failures | 4 | **0** |
| Lookahead-audit tests passing | 28/32 | **32/32** |
| Tests asserting a model *learned* something | **0** | **7** |
| Naive baselines available | 0 | 1 (`majority_baseline`) |
| Control separation (MCC signal − noise) | n/a | **+0.138** (0.155 vs 0.017) |

The control instrument's own numbers: noise acc 0.3452 vs baseline 0.3603
(no skill, correct); signal acc 0.4328 vs baseline 0.3737 (skill, correct);
shuffled-label MCC −0.0043 (skill destroyed, correct).

## 10. Regressions checked

Full suite re-run (§12). No source behaviour changed except the two fixed
defects; `_widen_to_float` only touches integer columns, and the summary
change is display-only.

## 11. Unresolved risks

- **R1.** The 26 source-grep tests still exist. Stage 1 did not ban them;
  they will be replaced opportunistically per `DECISIONS.md` #11.
- **R2.** 164 swallow-without-reraise handlers remain — the mechanism behind
  Phase 0 §4c. Scheduled for Stage 14.
- **R3.** Control datasets use a *hand-rolled* featurizer, so they validate
  models and metrics but **not** `FeatureEngineer`. Coverage of the real
  feature path needs F15 fixed first (Stage 6).
- **R4.** python-version config inconsistency (§8).

## 12. Instructions for the next agent (Stage 2 — architectural cleanup)

1. **Re-run before trusting anything here:**
   `pytest tests/test_controls.py tests/test_oof_alignment_contract.py tests/test_lookahead_audit.py -q`
   All must pass, with exactly 2 xfails and **zero** xpasses.
2. **Do not remove the xfail markers.** They flip at Stage 17. An XPASS is a
   suite failure by design — if you see one, alignment changed and the
   handoff must say so.
3. Stage 2 deletion work is **gated on user decision #1**. Until it arrives,
   do the ungated parts: retire the duplicate `AdapterResult`
   (`core/interfaces.py` copy has zero consumers) and the phantom
   `TrainingResult` annotation. Both are `DECISIONS.md` items 6 and 7 with a
   single defensible answer.
4. Verified-dead inventory (re-verify before deleting — every reference below
   is an `__init__.py` re-export only): `InferenceOrchestrator` (9 refs, all
   `src/inference/__init__.py`), `SecondLevelStacker` (2), `MetaLearnerSelector`
   (2), `MetaLearnerFactory` (15, all inside its own module), the three
   special-mode bundles (0 refs).
5. **Known-fragile:** `src/core` imports from `models`/`data`/`config`.
   Deleting re-exports can break import order. Run the *full* suite, not a
   subset.
6. Still **UNKNOWN**: whether any ensemble beats its constituents and a
   baseline. Unanswerable until F1 (Stage 17) and F7 (Stage 12) are fixed.
