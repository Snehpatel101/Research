# DECISIONS.md — Open Decisions After Phase 114

**Created:** 2026-08-20 (Phase 114 repository rehabilitation)
**Status:** All items below are PENDING USER DECISION. Nothing here was changed
during Phase 114 — the rehab deliberately fixed only things with one defensible
answer and left every product-level call to you.

Each item: what it is, why it matters, your options, and a recommendation.
Ordered by impact. Items 1–4 change what the product *does*; items 5–10 are
architecture/cleanup calls; items 11–12 are infra.

---

## 1. Serving / monitoring chain — wire it or delete it (~2,900 lines)

**What:** `src/inference/server.py` (FastAPI ModelServer), the drift-detection
and monitoring stack under `src/validation/monitoring/` (incl. the Slack
connector), and related production plumbing. None of it is reachable from any
entry point — no CLI command, no factory call, no test.

**Why it matters:** It's the largest block of maintained-but-dead code left.
The server also has a **known crash**: `/info` and `/predict` read
`.horizon` / `.feature_columns`, which `UniversalInferencePipeline` (the class
the server actually loads) doesn't have — every request would 500.

**Options:**
- **A. Wire it:** add an `ml serve` command through `UniversalInferencePipeline`,
  fix the attribute crash (add delegating properties), add a serve smoke test.
  ~1–2 days.
- **B. Delete it:** remove server.py + monitoring/ + drift chain; keep
  `load_deploy_artifact()` (which IS alive) as the deployment story. ~half day.

**Recommendation:** B, unless you actually plan to serve models over HTTP soon.
The alive deploy-artifact path already covers "load a trained model and
predict"; the HTTP layer can be rebuilt against a stable pipeline later.

---

## 2. Phase 52 special-mode inference bundles — wire or delete (~1,200 lines)

**What:** `walk_forward_bundle.py`, `regime_bundle.py`, `meta_labeling_bundle.py`,
`regime_detector.py` in `src/inference/`. Built in Phase 52 so non-standard
training modes would be deployable — but `BundleBuilder` never creates them and
nothing loads them. No producer, no consumer.

**Why it matters:** DIRECTION.md advertises "All training modes deployable."
Today that claim is false: only standard-mode bundles are ever produced. Either
the capability gets finished or the claim (and code) should go.

**Options:**
- **A. Finish it:** teach `BundleBuilder.build_from_training_result()` to emit
  the right bundle type per `training_mode`, and `UniversalInferencePipeline`
  to load them. ~2–3 days including tests.
- **B. Delete:** remove the four modules + their `src/inference/__init__`
  exports, correct DIRECTION.md. ~2 hours.

**Recommendation:** A if you use walk-forward/regime/meta-labeling modes for
real deployments; B if standard mode is what you actually ship.

---

## 3. Phase 99–102 feature-governance modules — wire, park, or delete

**What:** `bootstrap_stability.py`, `label_perturbation.py`,
`param_sensitivity.py`, `lifecycle.py`, `registry.py`, `economic_value.py`
(in `src/optimization/feature_selection/`) and `ticker_portability.py`
(in `src/validation/`). Only their own tests import them — the live
feature-selection pipeline never calls any of them. (By contrast, the Phase
98–99 pieces — timeframe budget, regime blend, robustness scoring — ARE wired.)

**Options:**
- **A. Wire the useful ones** into `_run_feature_selection_pipeline()` as
  opt-in diagnostic steps (config flags, like robustness scoring is today).
- **B. Park:** move to an `experimental/` package with their tests, so the live
  tree only contains reachable code.
- **C. Delete** modules + their ~80 tests.

**Recommendation:** A for `lifecycle`+`registry` (feature-governance bookkeeping
pairs naturally with the selection pipeline), B for the rest until you've used
them once in anger.

---

## 4. ModelContract.sequence_length not honored in standard mode (results change!)

**What:** Contracts say TCN=64 (its receptive field is 61) and transformers=128,
but standard-mode training windows everything at the global
`sequence_length=60`. Walk-forward mode DOES use the contract values — so the
same model trains with different windows depending on mode, and bundles record
the config value, creating train/serve skew for WF-trained models.

**Why it matters:** TCN at 60 is one bar SHORT of its receptive field — part of
the network literally never sees data. This is the biggest remaining
correctness-adjacent inconsistency, but fixing it **changes model results and
increases transformer memory**, so it needs your sign-off.

**Options:**
- **A. Honor contracts everywhere:** pass `model_contract.sequence_length` into
  adapter transforms in standard mode; bundles record the contract value.
  Results change for TCN/transformers; transformer memory rises (128-window).
- **B. Set contracts to 60:** consistency by fiat; TCN stays under-windowed.
- **C. Leave as-is** (documented inconsistency).

**Recommendation:** A — the TCN receptive-field argument is the whole reason
contracts carry per-model lengths. Do it as its own phase with before/after
metric comparison on a reference dataset.

---

## 5. The 5-dimension Optuna island — keep, wire, or delete (~3,500 lines)

**What:** `src/optimization/five_dimension_objective.py`, `hyperparameters.py`,
`base_feature_sets.py`, `artifact_saver.py`. Zero live consumers — the live
tuner is `TimeSeriesOptunaTuner` in `src/validation/cv/`. The island survives
because Phase 97/103 regression tests (D3, D4, the ATR Wilder-EMA test) pin it.

**Options:**
- **A. Keep as-is** (tests-only; costs import weight and maintenance).
- **B. Wire it:** give `run_5d_optimization()` a CLI/notebook entry point if
  joint label+feature+hyperparameter search is a workflow you want.
- **C. Delete the four modules + the pinning tests** (the D4 "-inf on
  degenerate labels" property is already implemented in the live tuner too —
  a replacement test against the live path would be written first).

**Recommendation:** C with the replacement test, unless 5-D search is on your
roadmap. Tests that only exercise dead code are weight, not safety.

---

## 6. Dual `AdapterResult` — retire the documented exception

**What:** Two classes named `AdapterResult`: the canonical one in
`src/data/adapters/base.py` and a legacy copy in `src/core/interfaces.py`.
CLAUDE.md documents the duplication as intentional (circular-import
prevention), but Phase 114's audit found the legacy copy has **zero consumers**
and the "kept in sync" bridge has drifted (incompatible `validate()`
semantics, read-only metadata, missing fields).

**Options:**
- **A. Delete the core copy**, re-export the canonical class from `src.core`
  for any external callers, update CLAUDE.md's Documented Exceptions table.
- **B. Keep and re-sync the bridge** (ongoing maintenance for no consumer).

**Recommendation:** A. The circular-import justification expired.

---

## 7. Core `TrainingResult` — phantom type

**What:** `src/core/interfaces.py::TrainingResult` is constructed nowhere.
The object that actually flows is `TrainingRunResult`
(from `src/models/training/unified_orchestrator.py`); `factory.py` merely
*annotates* with the phantom type.

**Options:**
- **A. Annotate with `TrainingRunResult`** and delete/deprecate core
  `TrainingResult`.
- **B. Adopt core `TrainingResult`** as the real contract and make the
  orchestrator return it (bigger refactor, cleaner layering).

**Recommendation:** A now; B only if you later formalize `src/core` as the
contracts layer for everything.

---

## 8. `ExperimentConfig.to_trainer_config / to_backtest_config / to_bundle_config` — adopt or delete

**What:** Three conversion methods with **zero callers**. Everything wired only
through them is dead config: CalibrationConfig details, CheckpointConfig,
ScalerConfig, SplitConfig, start/end dates, and part of FeatureConfig's
selection knobs are settable + serialized but never reach the pipeline.

**Options:**
- **A. Adopt:** make the factory use them (start/end date filtering, split
  ratios, scaler choice, calibration settings become real). ~2–3 days, real
  functionality gained.
- **B. Delete** the methods and prune the dead fields — honest config surface,
  smaller API.

**Recommendation:** A for `splits`, `start/end dates`, and `calibration`
(users reasonably expect those to work); B-style pruning for whatever you
decide you'll never wire.

---

## 9. Dead canonical-config layer + dead global.yaml sections

**What:** In `src/config/`: `model_configs.py`, `ensemble.py`, and the
canonical `BacktestConfig`/`OOMConfig`/`CheckpointConfig`/
`ParallelTrainingConfig` classes have no live consumers (the operational
twins elsewhere are what run). In `config/global.yaml`: the
`optimization.optuna`, `cross_validation`, `purge_embargo`, `mtf`, and
`features.sma_periods` sections are never read. Also `config/pipeline/`
loaders (`load_training_config`/`load_cv_config`) point at a directory that
doesn't exist and have no real callers.

**Options:** wire each to its operational twin, or delete. These are individually
small; the decision is really "does src/config stay the aspirational canonical
layer, or does it shrink to what runs?"

**Recommendation:** Shrink to what runs. Aspirational config classes are how
the Phase-114 class of "settable but ignored" bugs got created.

---

## 10. The 209-module import cycle (the big architecture item)

**What:** `src/config` ⇄ `src/models` ⇄ `src/data` (+validation, inference,
optimization) form one strongly-connected import component: importing *any* of
them loads torch+xgboost+lightgbm+catboost+optuna (~4–5s), driven by eager
facade re-exports and `src/models/__init__` importing every model family for
registration. Phase 114 took the safe quick wins only.

**Options:**
- **A. Staged SCC break:** lazy (PEP 562) re-exports in the facade `__init__`s
  + lazy model registration with an `ensure_registered()` guard at entry
  points. Main risk: code relying on import-side-effect registration timing
  (parallel workers, direct registry imports). ~3–5 days, staged commits.
- **B. Live with it** (costs: slow imports everywhere, `import src.config`
  needs a GPU stack installed, latent circular-import fragility — ~40
  "avoid circular" function-level imports exist as workarounds).

**Recommendation:** A, one facade at a time, full suite between stages.

---

## 11. Phase-regression tests that grep source text

**What:** ~20 remaining assertions in the phase-regression tier
(`test_phases_1_3.py`, `test_phases_4_11.py`, parts of d3) verify fixes by
`inspect.getsource(...)` substring checks — including asserting comments exist —
rather than testing behavior. (The worst offenders tied to deleted modules were
already rewritten behaviorally in Phase 114.)

**Options:** replace each with a behavioral assertion (the Phase 114 test files
show the pattern), or accept them as documentation-grade tests.

**Recommendation:** Replace opportunistically whenever a file is touched; not
worth a dedicated phase.

---

## 12. Python environment: adopt uv or drop the lockfile

**What:** `uv.lock` is checked in, but the actual dev environment is system
Python 3.12 with pip packages in `~/.local` (`--break-system-packages`).
The lockfile is stale fiction; reproducibility currently rests on
`requirements.txt` + Colab pins.

**Options:**
- **A. Adopt uv properly:** `uv venv` + `uv sync`, regenerate the lock, run
  tests through it. Cleanest reproducibility story.
- **B. Delete `uv.lock`** and declare requirements.txt/pyproject the truth.

**Recommendation:** A when convenient — the machine-level apt/pip shadowing
issues Phase 114 had to patch around (numpy2 vs apt bottleneck/numexpr,
mpl_toolkits hijack) simply don't happen inside a venv.

---

## Quick-reference matrix

| # | Decision | Default if you do nothing | Recommended | Effort |
|---|----------|---------------------------|-------------|--------|
| 1 | Serving/monitoring chain | Dead code + crashing server stays | Delete | ~0.5 day |
| 2 | Special-mode bundles | "All modes deployable" stays false | Wire if used, else delete | 2–3 days / 2 h |
| 3 | Governance modules | Tests-only forever | Wire lifecycle+registry, park rest | 1–2 days |
| 4 | Contract seq_len | TCN under-windowed, mode skew | Honor contracts (results change) | 1–2 days + eval |
| 5 | 5-D Optuna island | Dead code pinned by tests | Delete + replacement test | ~0.5 day |
| 6 | Dual AdapterResult | Drifted duplicate stays | Delete core copy | ~2 h |
| 7 | Core TrainingResult | Phantom annotation stays | Annotate real type, delete | ~1 h |
| 8 | to_*_config methods | Dead config fields stay | Adopt splits/dates/calibration | 2–3 days |
| 9 | Dead config layer/yaml | Aspirational surface stays | Shrink to what runs | ~1 day |
| 10 | Import SCC | 4–5s imports, GPU-stack coupling | Staged lazy break | 3–5 days |
| 11 | Source-grep tests | Documentation-grade tests stay | Replace opportunistically | rolling |
| 12 | uv adoption | Stale lockfile | Adopt uv venv | ~2 h |

---

*To act on any item: reference it by number. Items 1, 2, 3, 5 involve deleting
more than one file and per CLAUDE.md need your explicit go-ahead anyway; item 4
changes model results and should get a before/after comparison run.*
