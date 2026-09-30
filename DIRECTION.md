# ML Factory: Direction & Architecture

**Last updated:** 2026-09-30 (rewritten after Phase 117)
**Status:** Phase 117 complete. Next: Phase 118, proving the factory on real data.
**Role:** This is the architectural source of truth. It describes the system as
it is and where it is going. History lives in [COMPLETION.md](COMPLETION.md),
phase plans live in [CLEANUP_PLAN.md](CLEANUP_PLAN.md) and
[CLEANUP_TASKS.md](CLEANUP_TASKS.md), and open product calls live in
[DECISIONS.md](DECISIONS.md). Changes to this file need user approval.

---

## What ML Factory is

ML Factory is a config-driven factory for classification models on financial
bars. You give it raw OHLCV bars and an `ExperimentConfig`. It returns a model
or stacked ensemble that has been trained without leakage and backtested net
of costs. The result is packaged so that production inference replays
training exactly.

- **One entry point:** `MLFactory(cfg).run()` (`src/factory.py`). Every CLI
  command (`python -m src.cli run | data | status | models | cv | walk-forward
  | cpcv-pbo | version`) runs on it.
- **Mix and match:** any subset of 16 base models (2D tabular, 3D sequence, 4D
  multi-timeframe) can be combined with any of 5 meta-learners and any of 4
  training modes (standard, walk-forward, regime-aware, meta-labeling).
- **One deployment story:** `load_deploy_artifact(path).predict_from_raw(raw_bars)`
  (`src/inference/deploy.py`).

## The one pipeline

```text
raw OHLCV
  └─► _load_raw_bars: sanitize_bars (UTC, sort, drop bad prices, dedupe) + optional bar_timeframe resample
  └─► FeatureEngineer.compute_features (every feature lagged one bar; spec recorded)
  └─► triple-barrier labels per horizon + label spans [i, i + bars_to_hit]   (one Wilder ATR, cost in price units)
        └─ optional: CUSUM event sampling (only event bars carry labels; threshold fit on the train prefix)
  └─► chronological train / val / test split with max(purge, embargo) gaps
  └─► per-model feature selection on TRAIN rows only: target-aware clustered MDA + decorrelation
  └─► PreparedData: 2D (n, f) | 3D (n, seq, f) | 4D (n, streams, seq, f), positional row indices
  └─► 16 base models ──► purged-CV out-of-fold predictions (early stopping on a purged train tail)
  └─► meta-learner on OOF (purged holdout for metrics, refit on all OOF rows)
  └─► cost-aware backtest (signal at bar i fills at i + 1, stops/targets at barrier prices)
  └─► bundles (model | ensemble | regime | meta-labeling) + deploy manifest + run_manifest.json
  └─► load_deploy_artifact(...).predict_from_raw(raw) replays: sanitize → resample → spec → scaler → model(s)
```

| Stage | Canonical code |
|-------|----------------|
| Config, purge/embargo derivation, config hash | `src/config/experiment.py` (`ExperimentConfig.validate`, `resolve_cv_gaps`, `config_hash`) |
| Orchestration, resume, deploy | `src/factory.py` (`run`, `prepare_data`, `resume_from_checkpoint`, `_create_deploy`) |
| Raw-bar cleaning (train and serve) | `src/data/pipeline/stages/clean/sanitize.py` (`sanitize_bars`) |
| Features | `src/data/pipeline/stages/features/engineer.py` (`FeatureEngineer`, `to_spec` / `from_spec`) |
| Labels, costs, spans, events | `src/data/labeling/triple_barrier.py`, `src/core/label_spans.py`, `src/data/labeling/event_sampling.py`, `src/core/utils/atr.py` |
| Feature selection | `src/models/training/feature_selection.py`, `src/optimization/feature_selection/` |
| Adapters | `src/data/adapters/` (`PreparedData` in `preparation.py`; `tabular`, `sequence`, `multi_stream`) |
| Models and contracts | `src/models/` (`BaseModel`, `registry.py`), `src/core/contracts/model_contract.py` (`MODEL_CONTRACTS`) |
| CV, OOF, tuning | `src/validation/cv/` (`purged_kfold`, `cpcv`, `pbo`, `early_stopping_split`, `oof_core`, `cv_tuner`) |
| Stacking | `src/models/ensemble/`, `src/models/training/services/ensemble_service.py` |
| Backtest | `src/inference/backtesting/` (`backtest.py`, `execution.py`, `costs.py`, `position_sizing.py`) |
| Bundles and serving | `src/inference/` (`builder.py`, `bundle.py`, `ensemble_bundle.py`, `regime_bundle.py`, `meta_labeling_bundle.py`, `preprocessing_graph.py`, `deploy.py`) |
| Reproducibility | `src/core/reproducibility.py` (`set_all_seeds`), `src/core/run_manifest.py`, `src/models/tracking/` |

## Invariants

Each invariant has a named mechanism and a test that fails if it breaks.

| Invariant | Mechanism | Pinned by |
|-----------|-----------|-----------|
| **No lookahead in features** | Every feature uses bars up to t−1 (`shift(1)`). Higher timeframes use completed bars only. Session-cumulative features such as VWAP reset each session. | `tests/property/test_feature_causality.py`, `tests/property/test_preprocessing_graph_parity.py` |
| **No label leakage in CV** | Purge = the longest label span (`max_bars` over the horizons); an explicit shorter purge is raised to it. Embargo = one trading day of bars (`EMBARGO_SPAN_MINUTES`), capped at `MAX_EMBARGO_FOLD_FRACTION` of a fold. On top, every CV path purges on each label's actual end bar (`LabelSpans`). Under event sampling, the CV embargo is counted in events and covers the bar embargo; the val/test gap stays in bars. | `tests/property/test_purged_cv_invariants.py`, `tests/integration/test_label_overlap.py` |
| **Fit on training rows only** | Feature selection, scalers, label-cost calibration, the CUSUM threshold and frac-diff `d` are fit on training rows. In walk-forward they are fit on the bars before the first test window. | `tests/unit/optimization/`, `tests/e2e/test_event_frac_bet_e2e.py` |
| **Honest OOF** | Fold models early-stop on a purged tail of their own training rows, never on the fold they predict. OOF rows are re-indexed to source bars, so 2D/3D/4D predictions stack on the same bar. | `tests/property/test_purged_cv_invariants.py`, `tests/unit/validation/` |
| **Train/serve parity** | Bundles record the bar timeframe, the `FeatureEngineer` spec (including fitted `d` and the CUSUM threshold), selected features, contract sequence length, 4D streams, scaler and calibrator. `PreprocessingGraph` replays them, and `sanitize_bars` is shared. | `scripts/mix_match.py` prediction parity per combination, `tests/integration/test_bundle_roundtrip.py` |
| **History-independent rows** | A row is kept (training) or served only after `FeatureEngineer.warmup_bars()` bars, derived from the feature definitions (longest window, EWM settling to `ewm_settle_tolerance`, MTF ratio), and after the session lookback for session-reset features (`warmup_mask`). Bundles record `FEATURE_ENGINE_VERSION`; a bundle from another engine is refused at load unless `allow_engine_mismatch=True`. Too little input history raises an error naming the raw bars needed. | `tests/unit/features/test_history_independent_features.py`, `tests/unit/inference/test_serving_guards.py`, `tests/e2e/test_meta_labeling_serve_parity.py` |
| **Labels and backtest play one game** | One Wilder ATR (`wilder_atr`), one barrier resolution (`MLFactory._resolve_barrier_params`) and one cost term (`transaction_cost_in_price`, causal `expanding_cost_in_atr`). A signal at bar i fills at bar i + 1, and stops and targets exit at the barrier price. | `tests/unit/labeling/test_label_backtest_atr_parity.py`, `tests/property/test_backtest_causality.py` |
| **Costs everywhere** | Per-symbol commission and slippage are included in labels, the backtest and metrics. PSR/DSR, CPCV and CSCV PBO measure overfitting. | `tests/unit/backtesting/`, `tests/unit/validation/` |
| **Determinism** | `random_seed` seeds Python, NumPy and torch before any data work and reaches every model, Optuna, MDA and the meta-learner. Rankings are quantized and break ties by name. Two interpreters produce bit-identical OOF, holdout and deployed predictions on CPU. | `tests/e2e/test_determinism_e2e.py`, `tests/integration/test_threaded_determinism.py` |
| **Provenance** | `run_manifest.json` records the config and `config_hash()`, seed, commit, packages, data SHA-256 and metrics. Checkpoints are keyed by `config_hash()`; a resume with changed settings is refused, never cleared. | `tests/integration/test_factory_provenance.py` |
| **One definition per concept** | Core classes are defined once. There is no second pipeline, feature engine or ATR. | `tests/unit/test_single_definitions.py` |

## Extension points

**Add a model.** The full recipe is in
[docs/mix-and-match.md](docs/mix-and-match.md#adding-a-model).

1. Subclass `BaseModel` (`src/models/base.py`) in the family package and
   decorate it with `@register(name=..., family=...)` (`src/models/registry.py`).
2. Keep the model contract. Map labels with `map_labels_to_classes` /
   `map_classes_to_labels`, and return `n_classes` probability columns in
   class order. Honor `sample_weights`. Early-stop only on the `X_val` the
   pipeline passes in. `save`/`load` must round-trip.
3. Add the name to `MODEL_FAMILIES` / `MODEL_TO_FAMILY` (`src/core/constants.py`)
   and a `ModelContract` to `MODEL_CONTRACTS`: input rank, sequence length,
   max features, scaler, MTF mode.
4. Optionally add an Optuna space in `get_param_space` (`src/validation/cv/param_spaces.py`).
5. Add the model to `BASE_MODELS` in `scripts/mix_match.py` and run `custom`,
   then `solo` / `pairs`. Add unit tests under `tests/unit/models/`.

**Add a feature.**

1. Write `add_<name>(df, feature_metadata, ...)` in the matching module under
   `src/data/pipeline/stages/features/` and call it from
   `FeatureEngineer.compute_features`. Training and serving share that one
   function, so there is nothing to add on the inference side.
2. Use only bars up to t−1 (`shift(1)`). Never drop rows.
3. Anything fitted on data must be fitted on the train prefix and stored in
   the spec (`_SPEC_FIELDS`), so inference replays it. `frac_diff`'s `d` is
   the example to follow.
4. Bump `FEATURE_ENGINE_VERSION` when values change, so feature caches are
   rebuilt. The feature-causality property test must still pass.

**Add a meta-learner.**

1. Subclass `BaseModel` in `src/models/ensemble/` and register it with
   `@register(name=..., family="meta_learner")`.
2. Add it to `META_LEARNER_REGISTRY` in `_load_meta_learners`
   (`src/models/ensemble/meta_factory.py`) and to the `meta_learner` list in
   `MODEL_FAMILIES`.
3. It trains on OOF probabilities only. It must work for `n_classes` 2 and 3
   and persist with `safe_pickle_dump`.
4. Add it to `META_LEARNERS` in `scripts/mix_match.py` and run the `meta` kind.

## Verification pyramid

| Layer | Where | What it proves | Runs |
|-------|-------|----------------|------|
| Unit | `tests/unit/<area>/` | Behavior of one component. No test asserts on source text. | Every commit (`make check`) |
| Property | `tests/property/` (hypothesis, `HYPOTHESIS_PROFILE=ci` in CI) | Causality, purging, early-stopping carve, labels, backtest causality, streaming parity, config round-trip, sanitizer, over generated inputs | Every commit |
| Integration | `tests/integration/` | Bundle round-trip, label overlap, provenance, threaded determinism | Every commit |
| End to end | `tests/e2e/` (heavy ones marked `slow`) | CLI, factory, multi-horizon, AFML options, cross-interpreter determinism, mix-and-match | Fast subset on every commit, `slow` weekly and before merge |
| Harness matrix | `scripts/mix_match.py` → [docs/MIX_AND_MATCH.md](docs/MIX_AND_MATCH.md) | Every model, meta-learner and mode through deploy, with alignment, feature and prediction parity | `make matrix` (hours), before landing a phase |
| CI | `.github/workflows/ci.yml` | `checks` (ruff, black, pyright 0 errors, vulture, `uv lock --check`, fast tests on py3.11 and 3.12), `package` (wheel smoke), `docs` (strict build, link check), weekly `slow` | Every push and PR |

Suite size after Phase 117: 1,267 tests, about 1,140 of them in the fast suite.

## Non-goals

- **Not a trading system.** There is no broker connectivity, order routing or
  position management. The boundary is the deploy artifact and
  `predict_from_raw`.
- **No HTTP server or monitoring stack in the core.** It was deleted in Phase
  116. A thin serving extra (Phase 120) will be optional and sit on top of
  `load_deploy_artifact`.
- **No second pipeline, feature engine or config layer.** Every config field
  either reaches the pipeline or is removed. Deleted APIs get no
  compatibility shims.
- **No models without evidence.** New architectures come in only when the
  benchmark (Phase 118) shows a gain net of costs.
- **Not general forecasting.** The scope is bar-based classification with
  triple-barrier labels on futures-style instruments. Regression targets and
  tick data are out of scope; order-book features wait for depth data
  (Phase 122).
- **No GPU requirement.** The examples, the test suite and the matrix run on
  a CPU. A GPU only makes large datasets faster.

## Open decisions

- **Open:** DECISIONS.md #10, the import cycle. It is scheduled with the
  `mlfactory` package rename in Phase 120.

## Trajectory

| Phase | Goal | Done when |
|-------|------|-----------|
| 118 | **Prove it on real data.** Build a MES/MGC benchmark leaderboard: OOS macro-F1, log loss, net Sharpe, drawdown, turnover, DSR and PBO per model and ensemble, one run manifest per row. Compare against naive baselines, audit futures rolls, and A/B the opt-in AFML options. | The leaderboard is reproducible from one command, and shipped configs beat the baselines net of costs. |
| 119 | **Scale and speed.** Lazy 3D/4D windowing, a `--dry-run` memory/time estimator, CPCV + pruning + DSR-gated tuning at scale, and GPU profiling. | 1.6M-row runs for every model on one GPU box, and the estimator lands within ±30%. |
| 120 | **Production readiness.** Streaming inference with incremental feature state, drift monitoring, a paper-trading replay harness, a risk layer over `ProbabilityBetSizer`, a thin serving extra, a validated config schema, and the `mlfactory` package with the import-cycle break. | Replay matches backtest trades, the p99 latency budget is met, and importing the package does not load the GPU stack. |
| 121 | **Breadth.** Multi-symbol pooled training, benchmark-justified new models, and an experiment comparison report over manifests and MLflow runs. | Each addition is justified by a benchmark delta. |
| 122 | **L2 order-book features** (needs depth data). Book state strictly before bar close; imbalance, microprice, spread, depth slope and OFI features, replayed through the `FeatureEngineer` spec. | Lift over the bar-only baseline net of costs. |

Ongoing gates: a coverage floor on labeling, CV, backtest and bundles; a
nightly full sweep on larger hardware; and a release flow (tags, CHANGELOG).
