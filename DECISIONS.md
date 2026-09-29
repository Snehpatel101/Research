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

> **RESOLVED in Phase 116 (2026-09-29):** option B — `inference/server.py`,
> `inference/production/`, `validation/monitoring/` (drift detectors, alert
> handler, Slack connector), `scripts/serve_model.py` and the unused
> `ServerConfig`/`AlertConfig` deleted (2,568 lines), plus the `river` dependency
> and the `serving` extra; `load_deploy_artifact()` is the deployment story.

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

> **RESOLVED in Phase 115 (2026-09-29):** option A for regime-aware and
> meta-labeling (`RegimeBundle` with per-bar routing, `MetaLabelingBundle`,
> both built by `BundleBuilder` and loaded by `load_deploy_artifact`); option B
> for walk-forward (`walk_forward_bundle.py` and the duplicate
> `inference/regime_detector.py` deleted — walk-forward deploys a standard
> bundle). "All training modes deployable" is now true and matrix-verified.

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

> **RESOLVED in Phase 117 (2026-09-29):** wired as opt-in, read-only diagnostics
> (`data.features.governance.report: true`), with three modules deleted.
> After selection, `_run_feature_selection_pipeline` hands the finished selection
> and the TRAIN rows to `models/training/feature_governance.py`, which writes
> `<output_dir>/feature_governance/h{h}.json` and updates a per-symbol
> `FeatureRegistry` (`<runs dir>/feature_registry_<SYMBOL>.json`). It never
> changes the selected features (pinned by an e2e test: identical with and
> without the report). Every ranking inside it is the selection's own purged-CV
> out-of-sample MDA.
> - `bootstrap_stability` — **wired, rewritten**: the i.i.d. row bootstrap was
>   wrong for autocorrelated bars; now stability selection over random contiguous
>   blocks (selection frequency in the top-K).
> - `label_perturbation` — **wired, rewritten**: relabels the train rows with
>   the triple-barrier widths scaled (default x0.75 / x1.25) and flags features
>   whose MDA rank moves; no duplicate RandomForest, it reuses the live ranking.
> - `lifecycle` + `registry` — **wired, merged**: one transition table
>   (the `FeatureLifecycle` class duplicated the registry's history and is gone);
>   `record_run` advances CANDIDATE -> SELECTED -> ACTIVE <-> DEGRADED ->
>   RETIRED across runs from selection + stability verdicts. Retirement is
>   recorded and reported, never applied.
> - `param_sensitivity` — **deleted**: needs the whole feature set recomputed
>   under parameter variants, which `FeatureEngineer` cannot do generically;
>   overlaps label perturbation.
> - `economic_value` — **deleted**: its "Sharpe" was label x prediction on a
>   shuffled split (not returns, leaks time), and leave-one-out differences on
>   one holdout are noise; the backtester is the honest economic test.
> - `ticker_portability` — **deleted**: needs two symbols in one process and
>   used a shuffled split; compare two symbols' governance reports instead.
>
> Cost when enabled: about `n_bootstrap + 2` extra MDA rankings (default 8 + 2).

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

> **RESOLVED in Phase 116 (2026-09-29):** option A. Every mode windows each
> model at its contract length (TCN 64, Transformer 128, others 60) via
> `PipelineConfig.sequence_length_for()`; `data.sequence.seq_len` is now an
> optional override (default None) applied to all models. Bundles record the
> contract value; prediction parity verified for TCN and Transformer.

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

> **Phase 116 status:** deletion + behavioral replacement tests are prepared on
> branch `worktree-agent-aeee8a73f7ed7b404` (commit 8c22b29). Merging it was
> blocked by the session's auto-mode permission check on removing the files, so
> it awaits your go-ahead (`git merge worktree-agent-aeee8a73f7ed7b404`, keep the
> deletions). The one live bug it found — the live tuner scored single-class
> labels as a perfect 1.0 — was fixed and ported separately.

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

> **RESOLVED in Phase 116 (2026-09-29):** option A — the legacy copy (and the
> unimplemented `AdapterContract` ABC that referenced it) deleted from
> `src/core/interfaces.py`; nothing imported it from `src.core`, so no
> re-export was needed. `src/data/adapters/base.py` is the only definition.

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

> **RESOLVED in Phase 116 (2026-09-29):** option A — `factory.py` annotated with
> `TrainingRunResult` throughout; the phantom core `TrainingResult` deleted.

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

> **RESOLVED in Phase 116 (2026-09-29):** the three methods are deleted, and
> every ExperimentConfig field now either reaches the pipeline or is gone.
> Wired: `data.splits` ratios → PipelineConfig split ratios,
> `training.calibration.enabled/method` → orchestrator *and* TrainerConfig
> (the Trainer self-calibrated from global.yaml before), `verbose` → MLFactory
> default (start/end dates were already wired in Phase 115). Pruned: scaler,
> checkpoint, device, feature-period/selection knobs, labeling method/extras,
> Optuna sampler/startup/penalty knobs, WF gap/embargo (come from
> `training.purge_bars/embargo_bars`), SHAP/report/bundle-format flags.
> `from_dict` warns on and ignores unknown keys, so older YAML still loads.

> **Partly addressed in Phase 115:** `data.start_date` / `data.end_date` are now
> honored by `MLFactory` (raw bars filtered before features), and the new
> `data.bar_timeframe`, `training.regime` and `training.meta_labeling` settings
> flow to the pipeline. The three `to_*_config` methods remain unused.

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

> **RESOLVED in Phase 116 (2026-09-29):** shrunk to what runs.
> `src/config/model_configs.py`, `ensemble.py`, `inference.py` deleted, along
> with the dead canonical twins in `training.py`/`cv.py`/`data.py`/`base.py`
> (Checkpoint/OOM/ParallelTraining/Conformal/GA/ExperimentTracking, CV/CPCV/
> PurgedKFold/PBO/DSR/PurgeEmbargo, Scaler/MultiResolution/Session(s)/Bar,
> config mixins). `global.yaml` lost `optimization.optuna`, `cross_validation`,
> `purge_embargo`, `mtf`, `features.sma_periods` (GlobalConfig + validators
> updated; `validate_config_file` passes). `load_training_config` /
> `load_cv_config`, the environment-override layer that read them, and the
> `config/pipeline/` path constants are gone.

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

> **RESOLVED in Phase 116 (2026-09-29):** option A — `uv venv .venv`
> (Python 3.11, CPU torch), `uv.lock` regenerated against pyproject, CI installs
> with uv (`.github/workflows/ci.yml`), `make install` uses uv.

> **Phase 115 note:** development/verification now runs in `uv venv .venv`
> (Python 3.11, CPU torch, pandas 3.0) — no apt/pip shadowing issues were hit.
> `uv.lock` itself was not regenerated; that part of the decision is still open.

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

## 13. Second data pipeline behind `ml data` / `ml train` / `ml cv`

**What:** `src/data/pipeline/runner.py` (PipelineRunner) runs a separate 12-stage
pipeline (ingest → clean → features → labeling → GA/Optuna barrier search →
final labels → splits → scaling → datasets → validation → reporting) with its
own config class (`DataConfig`) and writes parquet splits that the CLI
`ml train model|ensemble`, `ml cv`, `walk-forward` and `cpcv-pbo` commands
consume via `TimeSeriesDataContainer.from_parquet_dir`. `MLFactory` (the path
the notebooks, `ml run`, README and the mix-and-match matrix use) shares only
the feature code with it. ~17k lines are runner-only. A Phase 116 smoke run
stopped at the post-`feature_scaling` schema check (10 NaN cells); no test runs
it end to end.

**Options:**
- **A. Re-point** the CLI train/cv commands at `MLFactory` artifacts and delete
  the runner stack (one pipeline, one config class).
- **B. Fix and test** the runner as a supported second path (add an e2e test).
- **C. Delete** the runner and its CLI commands outright.

**Recommendation:** A — two pipelines with two config classes is how
train/serve and label/backtest drift keeps reappearing.

---

## 14. Unused public-API modules (~7k lines)

**What:** modules that are exported by a package `__init__` but used by nothing
in the repo (no notebook, script, CLI or pipeline path): `core/resilience`,
`inference/orchestrator` (InferenceOrchestrator), `inference/pipeline`,
`config/validators`, `core/utils/{notebook,colab_setup,cache,checkpoint_manager,
config_validator,device_utils}`, `core/data_contract`, `core/defaults`,
`core/coordination/alignment`, `data/adapters/factory`,
`validation/statistical_tests`; script-only: `inference/batch`
(`scripts/batch_inference.py`), `validation/cv/cv_orchestrator`
(`scripts/phase3_validation.py`). Phase 116's verified dead-code sweep left them
in place because they are importable public API.

**Options:** delete them (and the stale scripts), or keep them as supported API
and add tests. **Recommendation:** delete — the supported serving API is
`load_deploy_artifact` / `load_bundle` / `UniversalInferencePipeline`.

---

## Quick-reference matrix

| # | Decision | Default if you do nothing | Recommended | Effort |
|---|----------|---------------------------|-------------|--------|
| 1 | Serving/monitoring chain | ✅ Resolved (Phase 116) | — | — |
| 2 | Special-mode bundles | ✅ Resolved (Phase 115) | — | — |
| 3 | Governance modules | ✅ Resolved (Phase 117: opt-in diagnostics wired, 3 modules deleted) | — | — |
| 4 | Contract seq_len | ✅ Resolved (Phase 116) | — | — |
| 5 | 5-D Optuna island | Dead code pinned by tests | Merge prepared branch (needs your OK) | minutes |
| 6 | Dual AdapterResult | ✅ Resolved (Phase 116) | — | — |
| 7 | Core TrainingResult | ✅ Resolved (Phase 116) | — | — |
| 8 | to_*_config methods | ✅ Resolved (Phase 116) | — | — |
| 9 | Dead config layer/yaml | ✅ Resolved (Phase 116) | — | — |
| 10 | Import SCC | 4–5s imports, GPU-stack coupling | Staged lazy break | 3–5 days |
| 11 | Source-grep tests | Documentation-grade tests stay | Replace opportunistically | rolling |
| 12 | uv adoption | ✅ Resolved (Phase 116: uv venv, lock regenerated, CI on uv) | — | — |
| 13 | Second data pipeline | Broken parallel pipeline stays | Re-point CLI at MLFactory, delete runner | 2–3 days |
| 14 | Unused public-API modules | ~7k unused lines stay | Delete | ~2 h |

---

*To act on any item: reference it by number. Items 1, 2, 3, 5 involve deleting
more than one file and per CLAUDE.md need your explicit go-ahead anyway; item 4
changes model results and should get a before/after comparison run.*
