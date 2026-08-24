# Phase 0 — Research Reconciliation

**Date:** 2026-08-24
**Inputs:** six parallel research agents + independent orchestrator verification
**Status:** COMPLETE — all six agents reconciled

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

## 4c. The end-to-end runtime truth (OBSERVED) — the decisive result

`MLFactory.run()` on `data/raw/MES_1m_1week.parquet`, xgboost, h=5, 2 splits,
backtest on. **Exit 0. Status `SUCCESS`. 15m54s.** And it is worthless:

| Symptom | Value |
|---|---|
| Label balance | **96.0 % one class** (2194 neutral / 48 long / 43 short) |
| MCC | **−0.0176** — *worse than random* |
| Per-class F1 | short 0.0 · neutral 0.972 · long 0.0 |
| Test split | contains **zero** true "long" samples |
| Backtest | **0 trades** from 50 non-neutral signals; every metric `0.0` |
| Feature engineering | **66.3 % of rows dropped** (6825→2297); 12 wavelet cols 100 % NaN |
| `cfg.data.mtf.enabled = False` | **ignored** |
| Printed summary | `F1=0.0000, Acc=0.0000` while `result.metrics` holds `macro_f1=0.324` |

**No existing test goes red for any of it.** This single result vindicates the
mission's insistence on live execution: 593 passing tests coexist with an
end-to-end path that reports SUCCESS while producing a worse-than-random model
and silently trading nothing.

Two verified root causes:

- **F15 — `MLFactory` discards all feature/MTF config.** `factory.py:753`
  constructs `FeatureEngineer(input_dir=..., output_dir=...)` and nothing else;
  the other caller (`features/run.py:242`) threads `timeframe`, `enable_mtf`
  and the rest. Every `data.features` / `data.mtf` setting is dead on the
  documented entry point. Side effect: it is currently *impossible* to request
  a cheap feature set, which is what blocks a fast test tier.
- **F16 — the printed summary reads the wrong keys.** `factory.py:122-123`
  does `model_metrics.get("val_f1", 0.0)` / `.get("val_accuracy", 0.0)` while
  base-model metrics are stored as `macro_f1` / `accuracy`. Ensemble metrics
  *do* use `val_*`, which is why this was never noticed.

Related: backtest 0 trades traces to `MarketHoursFilter` treating naive
timestamps as UTC→ET when the data's volume profile shows **Central** time
(only 19.6 % of bars pass), compounded by `_open_position` silently returning
on `contracts <= 0`.

## 4d. Metric validity — macro-F1 credits pure noise (OBSERVED, PROVEN)

Control pair built and validated: `docs/program/evidence/F2_control_datasets.py`.
`noise` is a driftless random walk (unpredictable by construction); `signal`
carries a real AR(1) momentum relationship. Both run through the repo's own
`ModelRegistry`.

```
dataset        n      acc     base      MCC  macroF1   F1base  accShuf  MCCshuf
noise       5976   0.3452   0.3603   0.0170   0.3450   0.1766   0.3597   0.0394
signal      5976   0.4328   0.3737   0.1554   0.4280   0.1814   0.3296  -0.0043
```

All five control assertions pass: no skill on noise, real skill on signal,
and shuffling labels destroys it. The instrument works.

**The finding: macro-F1 gains +0.1684 on PURE NOISE.** A model with MCC 0.017
and accuracy *below* the majority baseline still scores 0.345 macro-F1 against
the baseline's 0.177. The mechanism is structural — the majority baseline earns
zero F1 on two of three classes, so any model that merely spreads its guesses
appears better without predicting anything correctly.

**Consequence, and it is serious:** `macro_f1` is what Phase 55 uses to select
`primary_model` for the **deploy manifest**, and `f1_weighted` is Optuna's
default objective (`core/config.py:213`). Model selection, tuning, and
deployment in this repo are all driven by a metric that rewards noise.

**Ruling for the rest of this program:** headline go/no-go claims use
**accuracy-vs-majority-baseline and MCC**, never macro-F1 alone. Any metric
that cannot separate `noise` from `signal` is not measuring skill.

## 4e. Test-suite quality (OBSERVED)

Count is honest (§4b); quality is not. AST classification of 511 test
functions:

| Category | Count |
|---|---|
| Source-grep / tautological (assert on `inspect.getsource()` substrings) | **26** (`DECISIONS.md` #11 estimated ~20) |
| Mocked (tests a mock, not the code) | 12 |
| Type-check-only | 35 |
| Train a model **and assert it learned anything** | **0** |

`test_model_smoke.py` trains all 15 models on random features against random
labels — it can only detect crashes. `test_factory_e2e.py::test_backtest_metrics_dict_returned`
hides its only real assertion behind `if result.backtest_metrics:`.
`test_cli_smoke.py` asserts `exit_code in (0, 1)`.

**≈95 tests (16 %) exercise modules nothing in `src/` imports.** Only 87 of
366 src modules are touched by any test. Ensembles, 3 of 4 training modes, and
the entire serving layer are effectively untested.

Also: **164 of 211 `except Exception` handlers swallow without re-raising.**
The two worst — `_run_evaluation` returns `{}` on any backtest crash, and
`generate_oof` returns `None`, silently dropping a model from the ensemble.
That is the mechanism by which the §4c run reported SUCCESS.

---

## 5. Unknowns requiring runtime investigation

- ~~**U1.** Does the standard end-to-end path run to completion?~~ **RESOLVED
  (§4c):** yes — exit 0, `SUCCESS`, and worse-than-random output.
- ~~**U2.** True test-suite composition?~~ **RESOLVED (§4e):** 597 tests,
  **zero** of which assert that a model learned anything.
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

## 9. The single sentence that summarises Phase 0

593 tests pass, the end-to-end pipeline exits `SUCCESS`, and the system it
certifies produces a worse-than-random model, executes zero trades, prints
`F1=0.0000`, silently ignores its own configuration, aligns heterogeneous
ensembles 59 bars apart, and selects models for deployment using a metric that
rewards pure noise.

Nothing here is a missing feature. Every one of these is finished code that
returns a confident wrong answer. That is what the 23-stage chain has to fix,
and why each stage must verify its predecessor **by execution**.
