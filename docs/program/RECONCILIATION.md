# Phase 0 — Research Reconciliation

**Date:** 2026-08-24
**Inputs:** six parallel research agents + independent orchestrator verification
**Status:** DRAFT — testing section pending Agent 6

---

## 0. How to read this document

Six research agents produced findings. **Their reports are claims, not facts.**
Every load-bearing claim below was re-checked by me directly against source or
by execution; the verification log is `docs/program/evidence/` and the
orchestrator verification notes (V1–V14).

Three agent claims were **overstated** and are downgraded here. One was
**understated** and is upgraded to proven. Recording this explicitly matters:
the mission warns against assumptions hardening into facts as context moves
down a 23-agent chain, and the failure mode starts here.

Tags: **OBSERVED** (demonstrated, cited) · **INFERRED** · **PROPOSED** ·
**UNKNOWN**.

---

## 1. Confirmed facts

### 1.1 What is genuinely sound

This deserves to lead, because it is the foundation everything else stands on
and because the repo's reputation for being "all marketing" is not fair to it.

- **OBSERVED — The feature/leakage layer is rigorous.** A repo-wide grep for
  `.shift(-`, `center=True`, `bfill` returns exactly one hit, and it is
  deliberate target construction (`final_labels/core.py:234`). `PurgedKFold`
  index arithmetic, walk-forward gapping, MTF resample+`shift(1)`, triple-
  barrier causality, and Wilder's ATR are correct as claimed. Train/inference
  MTF parity holds (`factory.py:615` ↔ `inference/bundle.py:1137`).
- **OBSERVED — `ModelRegistry` is a real plugin system**
  (`src/models/registry.py:31`): decorator registration, name→class dict,
  family index, aliases. Model classes are not referenced by name outside
  their own modules.
- **OBSERVED — All 23 registered models construct with zero arguments.**
  Verified by constructing every one (V12).
- **OBSERVED — `get_adapter` is a dict registry**, not an if/elif chain.
- **OBSERVED — A canonical prediction type already exists.**
  `PredictionResult` (`core/interfaces.py:156-159`) makes
  `class_predictions`, `class_probabilities`, `confidence` **required**.
- **OBSERVED — The roster is 16 base learners, not 12** (V1). `logistic`,
  `random_forest`, `svm`, `transformer` are implemented, registered and
  smoke-tested but absent from `CLAUDE.md`'s table.

Four of five marketplace primitives already exist. This is a repair job, not a
green-field rewrite.

### 1.2 What is broken — ranked by capacity to produce deceptively good numbers

| # | Finding | Severity | Evidence |
|---|---|---|---|
| F1 | Heterogeneous OOF aligns on **positional** indices, not source bars | CRITICAL | **PROVEN** (§2) |
| F2 | Feature selection runs once **outside** the reporting CV | CRITICAL | OBSERVED |
| F3 | Optuna tunes on the set that is then scored; purge falls back to 60 < horizon 120 | CRITICAL | OBSERVED |
| F4 | MDA ranking runs on **shuffled** rows, defeating purge | CRITICAL | OBSERVED V10 |
| F5 | Financial report head-slices prices → every trade on wrong bar, train-split only | CRITICAL | OBSERVED V11 |
| F6 | **No naive baseline exists anywhere** | CRITICAL (mission) | OBSERVED V8 |
| F7 | 3D/4D models return `test_metrics: None` — 10/16 have no held-out estimate | HIGH | OBSERVED V9 |
| F8 | Meta-learner uses bare 80/20 split, **no purge/embargo** | HIGH | OBSERVED |
| F9 | `sequence_length` disagrees across sources (tcn 64, transformer 128, global 60) | HIGH | OBSERVED |
| F10 | Capability truth duplicated class-vs-contract; 4 live drift items | HIGH | OBSERVED V12 |
| F11 | `assert len(ALL_MODELS) == 23` at import — a 24th model crashes | HIGH (mission) | OBSERVED V13 |
| F12 | Ensemble strategy space is a hardcoded 4-entry dict | HIGH (mission) | OBSERVED |
| F13 | Lookahead audit **crashes** under pandas 3.0 (int column, float noise) | MEDIUM | OBSERVED |
| F14 | ~7,000 LOC reachable only via `__init__` re-exports | MEDIUM | OBSERVED |

### 1.3 Documentation claims that are FALSE

Stated plainly because they are load-bearing for anyone trusting this repo:

- **`session_cumsum()` does not exist.** `CLAUDE.md` Phase 94 credits it with
  fixing cumulative-feature session leakage. Zero matches in `src/`. (V7)
- **"All 8 ensemble combinations now PASS" (Phase 60) is meaningless.** They
  passed by not crashing; see F1.
- **"Cross-family ensembles working together" (Phase 57)** — same.
- **Phase 103's "strided temporal subsampling"** is not present at the MDA
  call site, which still shuffles. (V10)
- **`MLFactory` uses `PipelineRunner`** (`factory.py:5,170` docstrings) — it
  never imports it. The live data path is 2 steps, not 12.
- **TFT is 4D** — it is `SEQUENCE_3D`, `requires_4d=False`. Only `patchtst`
  and `itransformer` are 4D.
- **"All training modes deployable"** — only standard-mode bundles are ever
  produced (`DECISIONS.md` item 2, independently confirmed: zero references
  to the three special-mode bundles).

---

## 2. F1 in detail — the defect that invalidates the product

The mission's core requirement is arbitrary heterogeneous composition. That is
precisely what is broken, and it is broken **silently**.

- `SequenceAdapter` computes **correct source-bar** coordinates:
  `original_indices = df_indices[label_positions]`, starting at `seq_len-1`
  (`sequence.py:234-236`). Measured: `[59 60 61 62 63]`.
- OOF producers **discard** that and emit **positional** indices into each
  model's own windowed array: `valid_indices = np.where(~np.isnan(oof_preds))[0]`
  (`oof_generation.py:383`), plus `"datetime": range(n_samples)` (`:373`).
  Measured: `[0 1 2 3 4]`.
- `OOFAligner` set-intersects those positional integers as a shared key
  (`adapters/alignment.py`).

Measured with two **deliberately perfect** models
(`docs/program/evidence/F1_oof_misalignment_repro.py`):

```
agreement between two PERFECT models = 0.0%
expected if aligned correctly        = 100.0%
systematic shift                     = 59 bars
```

No crash, no warning, plausible `val_f1`. The stacker trains on rows whose
features describe bar *t* and whose label describes bar *t−59*.

F9 compounds it independently: for `tcn` and `transformer`, the alignment
validator derives offsets from `contract.sequence_length` (64, 128) while
windowing actually used the global 60.

**Consequence:** every cross-family ensemble number this repo has reported is
void. This is the mission's "executes ≠ statistically valid" distinction in its
purest form.

---

## 3. Corrections to agent claims

Recorded so they do not propagate.

| Claim | Agent | Verdict |
|---|---|---|
| Missing `predict_proba` splits the roster (HIGH) | 2 | **Downgraded to LOW.** `PredictionResult.class_probabilities` is required; probabilities flow uniformly. Residual risk is duck-typing only. (V2) |
| Four per-model registries have drifted (HIGH) | 2 | **Downgraded to MEDIUM.** All 16 base learners present in all four. Only phantom: `mlp` in `MODEL_DATA_REQUIREMENTS`. Real issue is declaration-site duplication, not drift. (V3) |
| Capability drift across contracts/families | 5 | **Confirmed exactly** — 4 items, class-vs-contract (not contract-vs-`MODEL_TO_FAMILY`, which agrees). (V12) |
| `sequence_length` mismatch affects cross-rank ensembles | 1 | **Confirmed but narrowed** — 2 of 10 sequence models (`tcn`, `transformer`); the other 8 contracts equal the global 60. |
| Heterogeneous alignment is misaligned | 3 | **Upgraded to PROVEN by execution.** (V6) |
| Repo has ~380k lines of markdown | *orchestrator's own prompt* | **Wrong — my error.** ~43k lines; 378,901 was bytes. Corrected by Agent 1. |
| `.venv` has zero packages | 1 | **Stale** — Agent 1 sampled before the install finished. Its findings are static-only, which does not invalidate them. |

I also made and corrected one measurement error of my own: my first drift
script iterated `ModelRegistry.list_models()`, which returns a family→names
mapping, and so compared six family strings rather than 23 models. Re-run
correctly (V12/V14).

---

## 4. Disagreements between agents

1. **Is 4D heterogeneous stacking valid?** `ensemble/validator.py:141` and
   `tests/test_ensemble_input_ranks.py:31` reject mixed ranks; `CLAUDE.md`
   Phases 57/60 claim it works. Agent 5 flags this as blocking and UNKNOWN.
   **Resolution:** both are literally true — the validator is never invoked on
   the live path (Agent 5, OBSERVED), so the runtime never asks. On the merits,
   once every member emits 2D OOF probabilities, member input rank is
   irrelevant to the meta-learner. The rank restriction is **stale**. This is
   a design decision to confirm, not a fact to assume — flagged for the gate.
2. **Registry quality.** Agent 1 calls dispatch "100% registry-driven"; Agent 5
   counts 15 name-based special-cases in generic code. **Resolution:** both
   correct at different layers — *model class* lookup is registry-driven;
   *capability* and *meta-learner* dispatch are not.

---

## 4b. Measured runtime baseline (OBSERVED)

Environment built from scratch this session (`uv venv`, PyPI only — the
PyTorch CDN is blocked by the agent proxy). torch 2.13 (CPU), pandas 3.0.5,
numpy 2.4.6, sklearn 1.9, xgboost 3.2, lightgbm 4.7, catboost 1.2.10.
Hardware: 4 CPU, 15 GB RAM, **no GPU** — this bounds the combination matrix.

Full suite, counted from pytest progress characters (the run's final summary
line did not survive capture; counts are from the 9 progress lines):

```
593 passed · 4 failed · 0 skipped · 0 errors   (597 executed)
```

**The repo's "~600 tests passing" claim is accurate in count.** That is worth
stating plainly — the count was honest.

All 4 failures share **one** root cause and sit in **one** file
(`tests/test_lookahead_audit.py`): `src/validation/lookahead_audit.py:376`
assigns float noise into an int64 column. pandas 2.x silently upcast; pandas
3.0 raises `LossySetitemError`.

**Decision taken:** keep the modern stack, fix the incompatibility. Pinning
back to pandas 2.x would suppress a genuine defect in a *leakage-detection
tool* — the last component where a blind spot is acceptable. Practical
consequence today: **the lookahead audit crashes on any dataset with integer
volume.**

Note this measures count, not quality. §U2 remains open.

---

## 5. Unknowns requiring runtime investigation

- **U1.** Does the standard end-to-end path actually run to completion on real
  data today? (Agent 6 pending.)
- **U2.** What is the true test-suite composition — how many tests assert real
  numeric behaviour vs `is not None`? (Agent 6 pending.)
- **U3.** With F1 fixed, does any ensemble beat its best constituent and a
  naive baseline on held-out data? **This is the question the whole program
  exists to answer, and it is currently unanswerable** because F6 means there
  is no baseline and F7 means 10/16 models have no test metrics.
- **U4.** Does the 4D path produce correct results once alignment is keyed?

---

## 6. Target architecture (PROPOSED)

Capability-first spine, retrofitted — **no model's `fit`/`predict` body is
rewritten**.

1. **`PredictionSet` — keyed, never positional.** Mandatory `keys`/`timestamps`
   in source-bar coordinates; `probabilities (N,C)`; an explicit `valid` mask
   as the *only* missingness signal (retiring the `-99/-999/-1/NaN` sentinels);
   `fold_id`; frozen schema (n_classes, class labels, horizon) and provenance.
   This is the direct fix for F1 and the precondition for every ensemble
   strategy. `PredictionResult` already carries the right *fields*; what is
   missing is the *key*.
2. **`ModelCapabilities`** — one frozen declaration per model, and the **single
   source of truth**. `MODEL_CONTRACTS` becomes a derived view, so F10-style
   drift becomes structurally impossible.
3. **Open registry** — delete the `== 23` assertion (F11); decorator
   registration plus entry-point discovery.
4. **`CompositionValidator`** — runs *before data loads*, emits actionable
   diagnostics with registry-generated alternatives. Incompatible combinations
   must fail clearly and intentionally, which is an explicit Definition-of-Done
   item.
5. **`EnsembleStrategy` registry** replacing the hardcoded 4-entry dict (F12),
   with its own capabilities (`supports_mixed_rank`, `requires_oof`,
   `requires_proba`, missing-data policy).
6. **Experiment manifest** separating a `models:` pool from named
   `compositions:`, so model A is fitted **once** and its OOF reused across
   every composition — this is what makes `A+B+C → X` then `A+D+F → Y` cheap.
7. **Evaluation framework** — mandatory baselines (majority-class, always-flat,
   best single constituent), held-out test metrics for **all** ranks (F7), and
   significance testing (Diebold–Mariano, Model Confidence Set, block
   bootstrap) so "better" is a measured claim.

---

## 7. Changes that should NOT be made

- **Do not rewrite the feature engineering layer.** It is the most correct part
  of the repo (§1.1).
- **Do not add forecasting models** (ARIMA/ETS/Theta) — category error; the
  task is classification on triple-barrier labels.
- **Do not add foundation models or online learners** now — task/domain
  mismatch, and walk-forward already covers adaptation.
- **Do not chase pandas-3 churn broadly** — 4 failures, one root cause.
- **Do not build a second abstraction beside `BaseModel`.** Extend it.
- **Do not delete the feature-governance modules yet** — unwired, but cheap to
  keep and relevant to F2.

---

## 8. Decisions required from the user

1. **Dead-code deletion batch (~7,000 LOC, verified zero real consumers):**
   `InferenceOrchestrator`, three special-mode bundles, `server.py` +
   monitoring, `second_level.py`, `meta_selection.py`, dead config layer.
   Repo rules require sign-off for multi-file deletion.
2. **Adjudicate 4D mixed-rank stacking** (§4.1) before `supports_mixed_rank`
   is set.
3. **Contract `sequence_length`** (`DECISIONS.md` item 4): honouring contracts
   changes TCN/transformer results and raises memory.

---

*Testing section (Agent 6) and the 23-stage sequencing follow once the
end-to-end runtime result lands.*
