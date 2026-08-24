# Agent 6 — Testing, Runtime Validation & Failure Analysis

**Repo:** `/home/user/Research` (ML Factory) · **Date:** 2026-08-24
**Env:** `/home/user/Research/.venv`, Python 3.11, no GPU, 4 vCPU / 15 GB RAM
**Scope:** investigation only. **No source files were modified.** Tests and scripts were run.

Tags: **OBSERVED** (I ran it) · **INFERRED** (from reading code) · **PROPOSED** (my
recommendation) · **UNKNOWN**.

---

# 0. Headline numbers

## 0.1 The real test-suite result

```
$ cd /home/user/Research && ./.venv/bin/python -m pytest tests/ -q -p no:cacheprovider

597 collected
593 passed
  4 failed
  0 skipped
  0 errored
  0 xfailed / xpassed

real  29m21.088s     user  41m02.238s     sys  0m19.021s
```

**OBSERVED.** (No `-n` / xdist; `pytest-timeout` is not installed so the `--timeout`
flag from the brief was dropped. Wall clock is inflated — another agent was running
the identical suite concurrently in the same container and `/proc/loadavg` sat at
8–12 on a 4-core box for the whole run. An idle-box estimate is **~10–12 min**,
INFERRED.)

The four failures are all one root cause:

```
FAILED tests/test_lookahead_audit.py::TestCleanFeature::test_sma_no_lookahead
FAILED tests/test_lookahead_audit.py::TestForwardLooking::test_forward_shift_detected
FAILED tests/test_lookahead_audit.py::TestForwardLooking::test_raises_in_blocking_mode
FAILED tests/test_lookahead_audit.py::TestCorruptionMethods::test_method_detects_lookahead[random]
```

So the docs' "~600 tests" is a true count, and the suite is *almost* green — but see
§2: the number is a poor proxy for how much of the product is actually verified.

## 0.2 The real end-to-end result

`MLFactory.run()` on `data/raw/MES_1m_1week.parquet`, 1 model (xgboost), horizon 5,
2 CV splits, no Optuna, backtest on. **OBSERVED, exit code 0:**

```
============================================================
ML Factory Experiment: SUCCESS
============================================================
Run ID: 20260824_073444
Duration: 947.4s
Models Trained: 1
Best Model: xgboost_h5

Model Performance:
  xgboost_h5: F1=0.0000, Acc=0.0000

Backtest Results:
  initial_equity: 100000.0
  final_equity: 100000.0
  total_return_pct: 0.0
  total_pnl: 0.0
  total_trades: 0
  win_rate_pct: 0.0
  profit_factor: 0.0
  sharpe_ratio: 0.0
  ...
  total_bars: 1607
  position_rate: 0.031113876789047916
  long_trades: 0
  short_trades: 0
============================================================
real 15m54.083s
```

**The pipeline runs to completion and reports SUCCESS while:**

1. the labels are **96.0 % one class** (2194 neutral / 48 long / 43 short);
2. the model's **MCC is −0.0176** — worse than random — with per-class F1 of
   `{short: 0.0, neutral: 0.972, long: 0.0}`;
3. the confusion matrix `[[0,12,0],[2,312,4],[0,0,0]]` shows the test split contains
   **zero true "long" samples**;
4. the backtest executes **0 trades** out of 50 non-neutral signals and reports every
   metric as `0.0`;
5. the human-readable summary prints **`F1=0.0000, Acc=0.0000`** even though
   `result.metrics` holds `macro_f1=0.324, accuracy=0.945` — the summary reads
   `val_f1`/`val_accuracy`, keys the metrics dict does not contain
   (`src/factory.py:122-123`);
6. **66.3 % of input rows are dropped** during feature engineering (6 825 → 2 297);
7. the config's `mtf.enabled = False` was **ignored** (MTF ran anyway);
8. not one existing test would go red for any of the above.

A second E2E on tiny synthetic data (2 500 rows, xgboost + lightgbm, ensemble,
bundling, deploy) **did work, in 22.7 s**: 2 models trained, ensemble built with
diversity analysis, 35 backtest trades, both bundles and a deploy manifest written.
So the machinery works; what is missing is any assertion that the *output* is sane.

---

# PART A — Audit of the existing test suite

## A.1 Claim vs reality

CLAUDE.md escalates from "212/212 passing" (Phase 66) to Phase 114's "suite grew
475 → ~600 tests". **OBSERVED:** `--collect-only` reports **597 tests** in **36 files**,
from **511 distinct test functions** (parametrization does the rest).

```
test_4d_followups.py                          4   test_leakage_detection.py             8
test_all.py                                  60   test_lookahead_audit.py              32
test_barrier_parity.py                        9   test_model_smoke.py                  63
test_bundle_roundtrip.py                     21   test_phase100_lifecycle_...py        44
test_calibrator_flow.py                      18   test_phase101_portability_...py      20
test_cli_smoke.py                            13   test_phase102_improvements.py        18
test_config_seams.py                         20   test_phase98_feature_governance.py   20
test_d10_entropy_shift.py                     4   test_phase99_advanced_governance.py  22
test_d11_determinism.py                       5   test_phases_1_3.py                   14
test_d12_config_hash.py                      12   test_phases_4_11.py                  18
test_d1_config_roundtrip.py                   4   test_purged_kfold.py                 18
test_d2_feature_leakage.py                    3   test_scaler_persistence.py           22
test_d3_feature_index.py                     10   test_triple_barrier.py               11
test_d4_degenerate_labels.py                  6   test_wf_oof_schema.py                 9
test_d5_binary_mode.py                        3   test_d6_atr_parity.py                 6
test_d7_barrier_exit.py                      13   test_d8_cost_parity.py               20
test_d9_rsi_parity.py                         7   test_ensemble_input_ranks.py          7
test_factory_e2e.py                          13   test_financial_metrics.py            20
                                                  TOTAL                               597
```

## A.2 Quality classification (AST analysis of all 511 test functions)

| Class | Count | Verdict |
|---|---|---|
| **SOURCE-GREP / TAUTOLOGICAL** — assert on `inspect.getsource()` substrings or `ast.parse` walks of `src/` | **26** | worthless; executes nothing |
| **MOCKED** — the fake collaborator is what is really under test | **12** | legitimate for config-threading, but proves nothing about behaviour |
| **IMPORT / TYPE-CHECK ONLY** — every assertion is `isinstance` / `callable` / `is not None` | **35** | smoke |
| **TRAINS a real model** (calls `.fit`/`.run()`/`.predict`) | **50** | see A.4 |
| **Behavioural / regression (the healthy remainder)** | **388** | genuinely useful |

### A.2.1 The 26 source-grep tests (full list, OBSERVED)

```
test_config_seams.py::test_save_yaml_uses_safe_load_compatible_output
test_d3_feature_index.py::test_walk_forward_selector_uses_permutation_importance
test_d3_feature_index.py::test_ohlcv_selector_uses_permutation_importance
test_factory_e2e.py::test_output_artifacts_exist
test_phase100_lifecycle_registry_drawdown.py::test_json_persistence
test_phases_1_3.py::test_trend_regime_shifted
test_phases_1_3.py::test_structure_regime_shifted
test_phases_1_3.py::test_cv_tuner_strided_subsampling
test_phases_1_3.py::test_hp_tuning_embargo_propagation
test_phases_1_3.py::test_higher_tf_shift
test_phases_1_3.py::test_class_weights_preserved_with_sample_weights
test_phases_1_3.py::test_multi_resolution_uses_as_tensor
test_phases_1_3.py::test_oof_generation_no_redundant_copy
test_phases_1_3.py::test_oof_sequence_single_backup
test_phases_4_11.py::test_stacking_only_copies_prob_columns
test_phases_4_11.py::test_single_experiment_config
test_phases_4_11.py::test_single_feature_selection_result
test_phases_4_11.py::test_k_up_zero_not_falsy
test_phases_4_11.py::test_confidence_zero_not_falsy
test_phases_4_11.py::test_sequence_dataset_no_copy
test_phases_4_11.py::test_val_batch_2x_train
test_phases_4_11.py::test_stop_slippage_applied
test_phases_4_11.py::test_kelly_min_trades
test_phases_4_11.py::test_worker_init_fn
test_phases_4_11.py::test_calibrated_meta_uses_timeseries_cv
test_phases_4_11.py::test_dead_code_removed
```

Archetype (`test_phases_4_11.py::test_k_up_zero_not_falsy`):

```python
source_path = Path("src/data/labeling/triple_barrier.py")
with open(source_path) as f:
    source = f.read()
assert "k_up is not None" in source
```

Passes if the string appears in a comment; fails on a semantically identical rewrite.
DECISIONS.md item 11 estimates "~20"; the real number is **26**.

### A.2.2 Conditional assertions — green whether or not the product works

`tests/test_factory_e2e.py::test_backtest_metrics_dict_returned`:

```python
assert isinstance(result.backtest_metrics, dict)
if result.backtest_metrics:          # post-fix behavior
    assert "total_trades" in result.backtest_metrics
```

Its own docstring says "backtest_metrics is currently ALWAYS empty". The only real
assertion is guarded by `if`. Combined with
`MLFactory._run_evaluation`'s terminal `except Exception: logger.warning(...); return {}`,
a completely broken backtest is invisible to this test **by construction**. Worse: on
my real-data run the guard *did* fire — `total_trades` was present — and its value was
**0**. The test still passes.

`tests/test_cli_smoke.py`:

```python
# Exit code should be 0; currently 1 due to the typer.Exit-swallow bug.
assert result.exit_code in (0, 1)
```

That covers every outcome. The docstring is also **stale**: `except typer.Exit: raise`
now exists at `src/cli/commands/pipeline.py:187, 249, 315, 382`, so the "known bug" it
documents is fixed — and the test could not have told anyone.

### A.2.3 Model-training coverage

**OBSERVED:** 50 test functions train something. Those that assert on real numeric
output:

* `test_bundle_roundtrip.py` (6) — save→load→predict bit-parity, shapes, ranges,
  column-order independence, missing/extra feature raises. **Best file in the suite.**
* `test_scaler_persistence.py` (10) — float32/float64 transform parity, inverse
  round-trip, clipping. Strong.
* `test_calibrator_flow.py` (4), `test_d2_feature_leakage.py` (3),
  `test_d5_binary_mode.py` (3), `test_factory_e2e.py` (3).
* `test_model_smoke.py` (8) — types, shapes and probability bounds only.

**No test anywhere asserts that any model learns anything.** `test_model_smoke.py`
trains all 15 registered models on `rng.randn` features against
`rng.choice([-1,0,1])` labels — pure noise, random targets — and asserts only
`isinstance(metrics, TrainingMetrics)` plus prediction shape. A model whose `fit()`
degenerated to a no-op would pass all 63 of those tests.

## A.3 Skips and xfails

**OBSERVED: 0 skipped, 0 xfailed in the actual run.** The suite has **no `xfail`
markers at all**. The `skip` sites that exist are conditional and did not fire here:

| Site | Condition |
|---|---|
| `test_model_smoke.py:195` | `pytest.mark.skip` for a model "registered but dependencies not available" |
| `test_model_smoke.py:298/328/345/363/382` | `pytest.skip` when `ModelRegistry.is_available(name)` is False |
| `test_phase99_advanced_governance.py:213` | `pytest.importorskip("statsmodels")` |
| `test_wf_oof_schema.py` (docstring) | the *integration* variant is "intentionally skipped… would train real models" — the real walk-forward OOF producer is never exercised |

The `is_available()` skips are a trap. `src/models/registry.py:408-435` catches
`ImportError` **and** `TypeError/ValueError/RuntimeError/AttributeError` and returns
`False`. A model broken by a genuine bug therefore reports as a green **skip**, not a
failure. Since the docs claim all 12 models are production-ready, these should be
hard assertions.

Absence of `xfail` also means every "KNOWN BUG" documented in a docstring is inert:
fixing the bug does not turn anything red, so the docstrings rot (see A.2.2).

## A.4 What is not covered

**OBSERVED (direct-reference analysis):** of **366** non-`__init__` modules under
`src/`, only **87** are referenced by any test — by dotted path or by a public symbol
name. **279 modules (~103 000 LOC) are never named in the suite.** That over-states the
gap a little, because `test_factory_e2e.py` drives `MLFactory.run()` and so transitively
executes the pipeline, adapters, trainer and backtester. But these areas have neither
direct nor meaningful transitive coverage:

| Area | Representative modules | Note |
|---|---|---|
| **Ensemble internals** | `diversity.py` (1 021 LOC), `blending.py`, `heterogeneous_stacking.py`, `meta_factory.py`, `meta_selection.py`, `mlp_meta.py`, `ridge_meta.py`, `xgboost_meta.py`, `second_level.py`, `orchestrator.py` | only `validator.py` + `stacking.py` are touched. The product's headline feature is essentially untested. |
| **Training modes** | `modes/walk_forward.py` (737), `regime_trainer.py` (843), `meta_labeling/*` | `test_wf_oof_schema.py` tests the *consumer* with hand-built frames; the producers never run. 3 of 4 advertised `training_mode` values have zero coverage. |
| **Inference / serving** | `server.py`, `orchestrator.py`, `batch.py`, `universal_pipeline.py`, `ensemble_bundle.py`, `walk_forward_bundle.py`, `regime_bundle.py`, `preprocessing_graph.py`, `deploy.py` | only `bundle.py`. DECISIONS #1/#2 confirm most is dead and `server.py` has a known 500-on-every-request bug. |
| **Data pipeline stages** | `runner.py`, `clean/cleaner.py`, `mtf/generator.py`, `features/engineer.py` (830), `features/microstructure.py`, `regime/hmm.py`, `sessions/calendar.py`, `store/*` | run transitively; no direct assertion on any stage's output. |
| **Neural internals** | `tft_model.py`, `nbeats_model.py`, `itransformer_model.py`, `resnet1d_model.py` | shape-level smoke only. |
| **CLI** | `commands/evaluate.py` (891 LOC: `cv`, `walk-forward`, `cpcv-pbo`) | `--help` and two error paths only. 3 of 8 commands never invoked. |
| **Metrics / reporting** | `models/metrics.py`, `evaluation/financial_report.py`, `regime_evaluation.py` (1 172), `deflated_sharpe.py` | untouched. |

## A.5 ≈16 % of the suite tests unreachable code

**OBSERVED.** These modules have **zero importers inside `src/`** — only their own tests
import them:

```
src/optimization/feature_selection/bootstrap_stability.py
src/optimization/feature_selection/label_perturbation.py
src/optimization/feature_selection/param_sensitivity.py
src/optimization/feature_selection/lifecycle.py
src/optimization/feature_selection/registry.py
src/optimization/feature_selection/economic_value.py
src/validation/ticker_portability.py
src/data/features/cusum_filter.py
src/data/features/frac_diff.py
```

Their tests: `test_phase100` (44) + `test_phase101` (20) + `test_phase102` (18) +
≈13 of `test_phase99` = **≈95 of 597 tests (16 %)**. By contrast `timeframe_budget`,
`regime_selection` and `robustness_scoring` **are** wired — into
`src/models/training/feature_selection.py:307 / 329 / 428` — so `test_phase98` is
legitimate. DECISIONS #3 already flags the modules; the testing consequence is that
the headline count is inflated by a sixth.

Related: 22 of the 32 `test_lookahead_audit.py` tests feed a **hand-written toy DAG**
declared inside the test file. There is no real feature dependency graph anywhere
(`grep -rn "FEATURE_DAG\|dependency_dag\|build_feature_dag" src/` → 0 hits), and
`LookaheadAuditor` / `scan_dependency_propagation` are **never called from `src/`** —
only re-exported.

## A.6 Harness-level problems

| Problem | Evidence |
|---|---|
| **No timeout plugin** | `pytest_timeout` not installed. `pyproject.toml`: `minversion=7.0`, `addopts="-ra -q --strict-markers"`, `testpaths=["tests"]`, `pythonpath=["."]`. One hung test hangs everything. |
| **No parallelism** | no `pytest-xdist`; 29 min serial on a 4-core box. |
| **`--strict-markers` with no registered markers** | a future `@pytest.mark.slow` would error, so tiering is currently impossible. |
| **`pythonpath=["."]` instead of an installed package** | `import src` **fails outside pytest** (§B.1). |
| **conftest is backtest-only** | 4 fixtures, all 100-bar random walks. No model, config, or ground-truth dataset fixtures. |
| **No dependency ceilings** | `pandas>=2.1.0`, `numpy>=1.26.0`, `scikit-learn>=1.4.0`, `xgboost>=2.0.0` with no upper bound. The venv legally resolved to pandas **3.0.5** — and that breaks production code (§B.3). |

---

# PART B — Runtime reality check

## B.0 Environment

**OBSERVED.**

```
Python 3.11 (/home/user/Research/.venv)
torch 2.13.0+cu130  (CUDA NOT available — CPU only)
pandas 3.0.5        numpy 2.4.6        scikit-learn 1.9.0
xgboost 3.2.0       lightgbm 4.7.0     catboost 1.2.10
optuna 4.9.0        numba 0.67.0       scipy 1.17.1
pytest 9.1.1        (no pytest-timeout, no pytest-xdist)
4 vCPU / 15 GB RAM ; /proc/loadavg peaked at 12.35 (shared container)
```

Every version above satisfies `requirements.txt`. The phase log's "all tests passing"
claims were made under a **different major pandas**.

## B.1 `import src` fails outside pytest

**OBSERVED.** First run of a standalone script:

```
Traceback (most recent call last):
  File ".../scratchpad/e2e_run.py", line 7, in <module>
    from src.config.experiment import ExperimentConfig
ModuleNotFoundError: No module named 'src'
```

`site-packages` contains no `.pth` and no dist-info for the project — `make install`
(`pip install -e .`) was never run. Tests work only because of
`[tool.pytest.ini_options] pythonpath = ["."]`. Re-running with
`PYTHONPATH=/home/user/Research` succeeded. Any non-pytest consumer (script, notebook
run from elsewhere, cron job) must set it by hand.

## B.2 `MLFactory` ignores the entire feature / MTF configuration

**OBSERVED.** The run set `cfg.data.mtf.enabled = False`. The log said:

```
INFO src.data.pipeline.stages.features.engineer: Timeframe: 5min, scale_periods: True
INFO src.data.pipeline.stages.features.engineer: MTF enabled: True, timeframes: ['15min', '60min']
INFO src.data.pipeline.stages.features.engineer: Wavelets enabled: True, type: db4, level: 3
INFO src.data.pipeline.stages.mtf.generator: Generating MTF features for 6825 rows (mode=both)
INFO src.data.pipeline.stages.features.engineer: MTF features: 28 columns from ['15min', '60min']
```

**Cause (OBSERVED in source, `src/factory.py:753`):**

```python
engineer = FeatureEngineer(
    input_dir=self.output_dir,
    output_dir=self.output_dir,
)
```

No configuration is passed. `FeatureEngineer.__init__`
(`src/data/pipeline/stages/features/engineer.py:175`) defaults to `timeframe="5min"`,
`enable_mtf=True`, `mtf_timeframes=['15min','60min']`, `enable_wavelets=True`,
`enable_microstructure=True`, `enable_volume_features=True`,
`enable_volatility_features=True`, `scale_periods=True`, `base_timeframe="5min"`.

The *other* caller, `src/data/pipeline/stages/features/run.py:242`, threads all of
these through properly. So there are two feature-engineering entry points with
divergent config handling, and **the documented single entry point uses none of it**.

Consequences:
* The whole of `ExperimentConfig.data.features` (mode, families, sma/ema/atr periods,
  rsi_period, macd_params, bb_period, bb_std) and `ExperimentConfig.data.mtf` are dead
  config — settable, written into `experiment_config.yaml`, ignored.
* `ExperimentConfig.to_pipeline_config()` *does* gate `mtf_timeframes` on
  `mtf.enabled` (`src/config/experiment.py:491`), so the config layer looks correct
  while the runtime path bypasses it.
* Indicator periods are scaled as if the data were 5-minute, on **1-minute input**,
  with no resampling anywhere. `cfg.data.timeframe` is never consulted.
* It is currently **impossible to request a cheap/minimal feature set from
  `MLFactory`** — which is exactly what a fast test tier needs.
* `tests/test_factory_e2e.py` sets the same flag with the comment
  `# keep feature count small / fast`. It has no effect and no test notices.

## B.3 `LookaheadAuditor` crashes under pandas 3 — all 4 suite failures

**OBSERVED**, verbatim (`--tb=short`):

```
.venv/lib/python3.11/site-packages/pandas/core/internals/blocks.py:1115: in setitem
    casted = np_can_hold_element(values.dtype, value)
.venv/lib/python3.11/site-packages/pandas/core/dtypes/cast.py:1705: in np_can_hold_element
    raise LossySetitemError
E   pandas.errors.LossySetitemError

During handling of the above exception, another exception occurred:
tests/test_lookahead_audit.py:85: in test_sma_no_lookahead
    result = auditor.audit_feature_function(
src/validation/lookahead_audit.py:271: in audit_feature_function
    df_corrupted = self._corrupt_data(df_clean.copy(), corruption_idx, price_cols)
src/validation/lookahead_audit.py:376: in _corrupt_data
    df.loc[df.index[start_idx:], col] = random_vals
.venv/lib/python3.11/site-packages/pandas/core/indexing.py:938: in __setitem__
    iloc._setitem_with_indexer(indexer, value, self.name)
.venv/lib/python3.11/site-packages/pandas/core/indexing.py:1953: in _setitem_with_indexer
    self._setitem_with_indexer_split_path(indexer, value, name)
.venv/lib/python3.11/site-packages/pandas/core/indexing.py:1997: in _setitem_with_indexer_split_path
    self._setitem_single_column(ilocs[0], value, pi)
.venv/lib/python3.11/site-packages/pandas/core/indexing.py:2181: in _setitem_single_column
    self.obj._mgr.column_setitem(loc, plane_indexer, value)
.venv/lib/python3.11/site-packages/pandas/core/internals/managers.py:1541: in column_setitem
    new_mgr = col_mgr.setitem((idx,), value)
.venv/lib/python3.11/site-packages/pandas/core/internals/managers.py:620: in setitem
    return self.apply("setitem", indexer=indexer, value=value)
.venv/lib/python3.11/site-packages/pandas/core/internals/managers.py:445: in apply
    applied = getattr(b, f)(**kwargs)
.venv/lib/python3.11/site-packages/pandas/core/internals/blocks.py:1118: in setitem
    nb = self.coerce_to_target_dtype(value, raise_on_upcast=True)
.venv/lib/python3.11/site-packages/pandas/core/internals/blocks.py:468: in coerce_to_target_dtype
    raise TypeError(f"Invalid value '{other}' for dtype '{self.values.dtype}'")
E   TypeError: Invalid value '[1961.95484858 2240.67322061 5556.25591897 2128.94846331 4057.48470211
E    2034.21017076 4165.63465537 3530.53872022 2901.29061738 3035.92107708
E    2322.51244052 3601.55277531 2554.65610465 1659.30422005 1598.5022082
E    3283.80350418 4917.9574108  3269.16841653 2945.44063579 3415.36997656
E    ...
E    3830.73414674 2624.80704282 3077.33038001 2893.38418841]' for dtype 'int64'
```

**Root cause:** `src/validation/lookahead_audit.py:376` writes a float array into
`volume`, which is `int64` in the fixture *and* in the real
`data/raw/MES_1m_1week.parquet`. pandas 2 upcast silently; pandas 3 raises. Only the
`random` corruption method is affected — and `random` is the **default** — so
`LookaheadAuditor.audit_feature_function()` is unusable out of the box on any frame
with an integer OHLCV column.

**Mitigating:** the auditor is not on the live path. The pipeline's pre-training
"lookahead validation" step only validates the resample config
(`Lookahead audit passed: resample config validated`). The live leakage guard is
`check_feature_label_correlation`, wired at `src/models/training/trainer.py:287` and
`src/models/training/feature_selection.py:514`. So this is a broken *tool* whose 32
tests give the misleading impression the lookahead guardrail is exercised.

## B.4 Real data → 96 %-degenerate labels, and nothing objects

**OBSERVED.**

```
  Loaded: 6825 rows from data/raw/MES_1m_1week.parquet
INFO ...nan_handling: Total columns: 227, Total rows: 6,825
WARNING ...nan_handling: Columns with 100% NaN (12): ['wavelet_close_approx', 'wavelet_close_d1',
   'wavelet_close_d2', 'wavelet_close_d3', 'wavelet_close_energy_approx', ...] and 2 more
WARNING ...nan_handling: Dropping 12 columns exceeding NaN threshold
INFO ...nan_handling: NaN cleanup: 227 cols -> 215 cols (-12 dropped)
INFO ...nan_handling: NaN cleanup: 6,825 rows -> 2,297 rows (-4,528 dropped, 66.3%)
WARNING ...nan_handling: High row drop rate (66.3%). This may indicate insufficient data
   for indicator warmup periods.
  Pipeline complete: 2297 rows, 217 columns
  Data sufficiency check: 2297 bars >= 290 required (embargo=30, purge=15, splits=2) — OK

INFO ...triple_barrier: Computing triple-barrier labels for horizon 12
INFO ...triple_barrier:   k_up=1.500, k_down=1.000, max_bars=12
INFO ...triple_barrier:   Transaction costs applied: symbol=MES, regime=low_vol, cost_in_atr=6.9651
INFO ...triple_barrier: Label distribution for horizon 12:
INFO ...triple_barrier:   Short/Loss          :     43 (  1.9%)
INFO ...triple_barrier:   Neutral/Timeout     :   2194 ( 96.0%)
INFO ...triple_barrier:   Long/Win            :     48 (  2.1%)
INFO ...triple_barrier:   Invalid samples: 12 (excluded from training)
```

`cost_in_atr = 6.97` puts the cost-adjusted barriers ~7 ATR away, so almost nothing can
be hit within 12 bars. **Three separate signals of an unusable dataset — 12 all-NaN
wavelet columns, a 66 % row drop, a 96 % single-class label distribution — and the
"data sufficiency check" passes because it only compares row count to
`n_splits*100 + n_splits*(purge+embargo)` = 290** (`src/factory.py:623-655`). Nothing
checks label balance, class presence, or feature validity.

Resulting metrics (**OBSERVED**, `result.metrics['xgboost_h5']`):

```
accuracy      0.9455        macro_f1  0.3240        weighted_f1 0.9366
mcc          -0.0176        per_class_f1  {short: 0.0, neutral: 0.9720, long: 0.0}
confusion_matrix [[0,12,0],[2,312,4],[0,0,0]]      n_samples 330
logloss_unweighted 0.2544   logloss_weighted 2.3568
trading: position_rate 0.0182, position_win_rate 0.0, directional_edge 0.0,
         max_consecutive_losses 6, expectancy -1.0, profit_factor 0.0
```

`logloss_unweighted` fell from 1.07 to 0.25 during training — a number that looks
excellent and is entirely an artifact of a 96 % majority class. Any test asserting
"loss went down" would be fooled; a test comparing macro-F1/MCC to the majority-class
baseline would not. **This is the strongest argument for the control-pair fixtures in
Part C.**

Quiet degradations along the way (**OBSERVED**):

```
WARNING src.data.adapters.preparation: PreparedData validation issues:
  ['y_test contains 12 invalid labels (-99)']
WARNING src.models.calibration.calibrator: Calibration: class 2 has only 1 unique
  label(s) in OOF — skipping calibrator (pass-through).
WARNING src.models.training.training_ops:     xgboost doesn't support predict_proba, skipping
WARNING src.models.evaluation.financial_report:   Insufficient trades for bootstrap CIs (need > 10)
```

## B.5 The backtest executed 0 trades and reported success

**OBSERVED.** `Backtest complete: 0 trades / Win rate: 0.0% / Sharpe: 0.00`, from
1 607 prediction rows carrying 47 long and 3 short signals (`position_rate` 3.1 %).

Two contributing causes:

1. **Market-hours filter, likely timezone-mismatched.** `MarketHoursFilter`
   (`src/inference/backtesting/execution.py:97-133`) assumes a naive timestamp is
   **UTC**, converts to US/Eastern, and admits only 09:30–16:00 ET for MES. Measured
   against the real file (**OBSERVED**):

   ```
   bars: 1365  tradeable: 267 (19.6%)
   data time range: 2020-01-06 00:00:00 -> 2020-01-12 23:55:00
   volume by hour of the file's own clock (top): 14, 8, 9, 10, 11, 13, 12, 15
   ```

   The volume profile peaks at hours **8 and 14** — the CME Chicago RTH open (08:30 CT)
   and close (15:00 CT). So the data's clock is almost certainly **Central Time**, not
   UTC, and the filter's admitted window is offset from the actual liquid session.
   There is no timezone metadata on the parquet and nothing in the pipeline validates
   or records one. (INFERRED from the volume profile; the file carries no tz.)

2. **Silent zero-size drop.** `Backtester._open_position`
   (`src/inference/backtesting/backtest.py:562`) does
   `contracts = self._calculate_position_size(...)` then `if contracts <= 0: return` —
   no counter, no warning, no record. A signal that cannot be sized simply disappears.

Either way the product reports `SUCCESS`, and
`test_factory_e2e.py::test_backtest_metrics_dict_returned` passes.

## B.6 `ExperimentResult.summary()` always prints `F1=0.0000, Acc=0.0000`

**OBSERVED** (`src/factory.py:119-124`):

```python
for model_name, model_metrics in self.metrics.items():
    f1  = model_metrics.get("val_f1", 0.0)
    acc = model_metrics.get("val_accuracy", 0.0)
    lines.append(f"  {model_name}: F1={f1:.4f}, Acc={acc:.4f}")
```

Model metrics use the keys `macro_f1` and `accuracy`
(`src/models/metrics.py:compute_classification_metrics`), never `val_f1`/`val_accuracy`.
So the headline number a human reads is *always* 0.0000 for every model. Ensemble
metrics *do* use `val_f1`/`val_accuracy`, which is why the ensemble line renders
correctly — masking the bug. No test asserts on `summary()`.

## B.7 The ensemble path does work — on synthetic data, in 22 seconds

**OBSERVED.** `MLFactory.run()` on 2 500 synthetic 5-min rows,
`models=["xgboost","lightgbm"]`, `build_ensemble=True`, `create_bundle=True`,
`deploy_artifact=True`:

```
success: True | n_models: 2 | best: xgboost_h5
metrics keys: ['xgboost_h5', 'lightgbm_h5']
ensemble_metrics: {'val_f1': 0.1903, 'val_accuracy': 0.2440, 'train_loss': 1.0695,
  'val_loss': 1.1021, 'training_time': 0.0283, 'diversity_score': 0.1402,
  'diversity_q_statistic': 0.9662, 'diversity_correlation': 0.7776,
  'diversity_disagreement': 0.1505, 'diversity_double_fault': 0.5388,
  'diversity_entropy': 0.0512, 'diversity_kl_divergence': 0.0418}
backtest_metrics: {..., 'total_trades': 35, 'win_rate_pct': 48.57,
  'profit_factor': 0.3779, 'sharpe_ratio': -40.3714, 'sortino_ratio': -20.6773,
  'calmar_ratio': -192.3263, 'max_drawdown_pct': -0.2953, 'total_bars': 1043,
  'signals': {'long': 0, 'short': 184, 'neutral': 96}, 'long_trades': 0, 'short_trades': 35}
bundle_path: .../bundles
deploy_path: .../deploy
real 0m22.690s
```

Good news: ensemble alignment, diversity analysis, the ridge meta-learner, backtest,
per-model bundles (`manifest.json`, `metadata.json`, `features.json`,
`feature_spec.json`, `preprocessing_graph.json`, `calibrator.pkl`, `model/`) and a
deploy manifest all work, fast. Problems visible in the same output:

* **`sharpe_ratio = −40.37`, `calmar_ratio = −192.33`** from 35 trades. These are
  annualization artifacts, reported without any sanity bound. Nothing tests plausible
  ranges.
* **`signals` sums to 280 but `total_bars` is 1043** — the prediction frame covers 27 %
  of the price bars and no test checks alignment coverage.
* **The ensemble is worse than its members and worse than chance** (`val_accuracy`
  0.244 on 3 classes) and is still reported as a success. The service *does* warn:
  `Low ensemble diversity detected: score=0.140 < threshold=0.300` and
  `High Q-statistic (0.966) indicates near-identical predictions` — warnings only.
* **The ensemble never reaches the deploy manifest.** Both entries are
  `is_ensemble: false`; `primary_model: "xgboost"`. Confirms DECISIONS #2 — only
  single-model standard-mode bundles are produced.
* The deploy manifest happily ships a model with `mcc: -0.0253` (xgboost) alongside
  `mcc: -0.0772` (lightgbm) on what is a pure random walk. **There is no quality gate
  between "trained" and "deployed."**

## B.8 Invalid combinations *are* handled well

**OBSERVED** (direct calls):

```
ModelRegistry.create('does_not_exist')
  -> ValueError: Unknown model 'does_not_exist'. Available models: [...]
validate_ensemble_config(['xgboost','lstm'],    'voting')   -> (False, "Cannot mix models with different input ranks in voting...")
validate_ensemble_config(['xgboost','patchtst'],'stacking') -> (False, "Cannot mix models with different input ranks in stacking...")
validate_ensemble_config(['xgboost','lstm'],    'stacking') -> (True, "")
validate_ensemble_config(['xgboost'],           'stacking') -> (False, "Need at least 2 base models for ensemble, got 1")
validate_ensemble_config([],                    'stacking') -> (False, "No base models specified")
validate_ensemble_config(['xgboost','bogus'],   'stacking') -> (False, "Model 'bogus' is not registered...")
```

This is the healthiest part of the product's error handling. Two caveats:

* **Docs contradict code.** CLAUDE.md Phase 60 claims "All 8 ensemble combinations now
  PASS … 2D+4D, 2D+3D+4D all working". The validator rejects any 4D model mixed with a
  2D or 3D model in **every** ensemble type.
* **Two different functions share the name `validate_ensemble_config`** —
  `src/models/ensemble/validator.py` and `src/core/utils/config_validator.py:365` —
  with different signatures and return types.

Contract vs implementation ranks **agree** for all 15 registered models (OBSERVED):
2D = xgboost/lightgbm/catboost/random_forest/logistic/svm; 3D = lstm/gru/tcn/
inceptiontime/resnet1d/**tft**/nbeats; 4D = patchtst/itransformer. Note TFT is **3D**,
contradicting several CLAUDE.md phase notes (e.g. Phase 57's `_generate_4d_oof`
docstring still lists "PatchTST, iTransformer, TFT").

## B.9 164 exception handlers swallow without re-raising

**OBSERVED (AST scan of `src/`).** 211 `except Exception` clauses; **164 `except`
handlers contain no `raise` anywhere and terminate in `pass` / `return None` /
`return {}` / `return []` / `continue`.** Top files:

```
  8  src/models/device.py                        3  src/core/checkpoint.py
  6  src/cli/commands/evaluate.py                3  src/inference/builder.py
  5  src/core/common/timeframes.py               3  src/models/training/services/ensemble_service.py
  5  src/inference/production/monitor.py         3  src/models/training/training_ops.py
  4  src/factory.py                              3  src/data/pipeline/stages/features/wavelets.py
  4  src/models/registry.py                      3  src/models/config/environment.py
  4  src/models/training/feature_selection.py    3  src/core/utils/notebook.py
```

The two that matter most for validation:

* `MLFactory._run_evaluation` — `except Exception as e: logger.warning(f"Backtest
  failed: {e}"); return {}`. A broken backtest is indistinguishable from
  `run_backtest=False`.
* `OOFGenerationService.generate_oof` — `except Exception as e: logger.warning(f"Failed
  to generate OOF for {request.model_name}: {e}"); return None`. A model that fails OOF
  generation silently vanishes from the ensemble. In the same file,
  `_generate_oof_inner` runs a post-hoc fold-leakage check that on failure does
  `logger.error("OOF fold leakage detected for %s: %d violations", ...)` **and
  continues**.

`EnsembleService.build_ensemble` is better — it returns
`ensemble_metrics={"error": "..."}` — but nothing in `ExperimentResult` surfaces it.

## B.10 Other runtime observations

* `src/inference/server` is imported on every run — the first log line of any job is
  `INFO src.inference.server: prometheus-client not available`. The known-dead FastAPI
  module is on the import path of every process.
* Hundreds of `PerformanceWarning: DataFrame is highly fragmented` from
  `src/data/pipeline/stages/features/wavelets.py:237/247/286/293/298` (repeated
  `frame.insert` in a loop) — and the wavelet columns it produces are then dropped as
  100 % NaN on 1-minute data.
* Real-data cost: **15m54s wall (20m04s CPU)** for one xgboost model on 6 825 raw rows,
  because 209 features are computed unconditionally and XGBoost runs
  `n_estimators=1000, n_jobs=-1` (`src/models/boosting/xgboost_model.py:129`). The same
  pipeline on 2 500 synthetic rows took **22.7 s**. Feature-count and estimator count,
  not row count, dominate.
* Triple-barrier cost calibration uses a **global** median ATR over the whole dataset
  (`src/data/labeling/triple_barrier.py:601-605`) — a full-sample statistic used to set
  label thresholds. The comment explicitly rejects an expanding median, while
  CLAUDE.md Phase 96 (item C13) claims "Triple barrier ATR uses expanding median".
  Docs and code disagree, and the code's choice is mildly lookahead-flavoured.

---

# PART C — Proposed validation strategy

## C.0 Principles

1. **Every test must be able to fail for the right reason.** The suite's biggest
   defect is not coverage — it is tests that cannot distinguish a working product from
   a broken one (conditional assertions, `assert x is not None`,
   `assert exit_code in (0,1)`, source-substring greps).
2. **Statistical claims need controls.** Nothing asserts a model learns. Two synthetic
   datasets with known ground truth turn "did it run" into "did it work".
3. **Silent degradation is the enemy.** 164 swallowing handlers mean tests must assert
   on the *presence and sanity of results*, not the absence of exceptions.
4. **Tier by cost.** A ≤90 s tier on every change; a ≤20 min tier pre-merge; the rest
   nightly.
5. **Pin the dependency ceiling.** Half of the observed breakage is pandas 3 arriving
   through an unbounded `pandas>=2.1.0`.

## C.1 `conftest.py` fixture design (PROPOSED)

### C.1.1 The three ground-truth datasets

```python
# tests/conftest.py  (PROPOSED)
import numpy as np, pandas as pd, pytest

FREQ = "5min"

def _ohlcv_from_close(close, seed):
    rng = np.random.RandomState(seed); n = len(close)
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.05, n)
    eps = np.abs(rng.normal(0, 0.3, n)) + 0.05
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq=FREQ)
    df = pd.DataFrame({
        "open":  open_,
        "high":  np.maximum(open_, close) + eps,
        "low":   np.minimum(open_, close) - eps,
        "close": close,
        # float, NOT int — an int volume column breaks the lookahead auditor
        # under pandas 3 (finding B.3)
        "volume": rng.randint(100, 5000, n).astype(float),
    }, index=idx)
    df.index.name = "datetime"
    return df

@pytest.fixture(scope="session")
def noise_ohlcv():
    """PURE NOISE. Ground truth: no model may beat the majority-class baseline.
    A model winning here is detecting leakage, not skill."""
    rng = np.random.RandomState(11); n = 6000
    return _ohlcv_from_close(5000.0 + np.cumsum(rng.normal(0, 1.5, n)), 11)

@pytest.fixture(scope="session")
def signal_ohlcv():
    """PLANTED SIGNAL. A hidden AR(1) state drives the NEXT bar's drift, so lagged
    returns are genuinely predictive. Ground truth: any competent model MUST beat
    the majority-class baseline out of sample."""
    rng = np.random.RandomState(12); n = 6000
    st = np.zeros(n)
    for i in range(1, n):
        st[i] = 0.92 * st[i-1] + rng.normal(0, 1.0)
    drift = 0.6 * np.r_[0.0, st[:-1]]
    return _ohlcv_from_close(5000.0 + np.cumsum(drift + rng.normal(0, 0.8, n)), 12)

@pytest.fixture(scope="session")
def leaky_ohlcv(signal_ohlcv):
    """Signal set PLUS a forward-shifted close. Ground truth: the leakage guard
    MUST reject it. A completed run means the guardrail is dead."""
    df = signal_ohlcv.copy()
    df["oracle"] = df["close"].shift(-5)
    return df
```

**These are not hypothetical — I ran them.** OBSERVED, using the repo's own
`ModelRegistry.create("xgboost")` on 10 lagged-return features with a 70/30 temporal
split and a 50-bar purge:

```
== NOISE  ==  train=4188 test=1746
  model    macro_f1=0.3465  acc=0.3465
  baseline macro_f1=0.1741  acc=0.3534      <- model does NOT beat baseline accuracy

== SIGNAL ==  train=4188 test=1746
  model    macro_f1=0.5624  acc=0.5601
  baseline macro_f1=0.1539  acc=0.3001      <- model beats baseline by 26 pts
```

Total runtime: a few seconds. Note the important nuance the experiment exposed:
**macro-F1 is higher than the baseline's even on pure noise**, because a
single-class baseline scores badly on macro-F1 by construction. The correct
statistics are **accuracy (or MCC) versus the majority baseline**, plus a
**label-shuffle permutation control**. Asserting on macro-F1 alone would produce a
false "model has skill" on noise.

### C.1.2 Supporting fixtures

| Fixture | Scope | Purpose |
|---|---|---|
| `tiny_2d(n=400,f=12)` | session | tabular `(X,y)`, 3-class, seeded |
| `tiny_3d(n=400,seq=16,f=12)` | session | lstm/gru/tcn/inceptiontime/resnet1d/tft/nbeats |
| `tiny_4d(n=400,tf=2,seq=16,f=8)` | session | patchtst/itransformer |
| `data_for(model_name)` | function | dispatch on `requires_4d`/`requires_sequences` — one source of truth for shape |
| `fast_config(model_name)` | function | minimal hyper-params, and **asserted to be honoured** (`epochs_trained <= max_epochs`) |
| `parquet_of(df)` | function | writes to `tmp_path`, returns path (E2E entry) |
| `tiny_experiment_config(path,out,**over)` | function | canonical fast `ExperimentConfig` (n_splits=2, purge=15, embargo=30, n_trials=0, max_epochs=2, batch=64) |
| `degenerate_variants` | params | one-class labels · all-NaN column · constant column · 30-row frame · duplicate timestamps · unsorted index · int-dtype volume · float32 vs float64 |
| `strict_logs` | marker-gated | fails the test if any `logger.warning`/`logger.error` is emitted from `src.*` — this is how the 164 silent swallows become visible |
| `rss_guard` | autouse | fails a test whose peak RSS exceeds 3 GB (psutil) |

### C.1.3 Baseline helpers (the statistical spine)

```python
def majority_baseline(y_train, y_test) -> np.ndarray
def bootstrap_ci(y_true, y_pred, metric, n=1000) -> tuple[float, float]
def assert_no_skill(y_true, y_pred, y_train)   # noise_ohlcv: CI must contain baseline
def assert_has_skill(y_true, y_pred, y_train)  # signal_ohlcv: CI must exclude baseline
def assert_shuffle_destroys_skill(fit_fn, X, y)  # permutation control
```

Use bootstrap CIs over test rows rather than bare thresholds so the assertions are
stable across seeds instead of flaky.

## C.2 Coverage axes and what each must assert

| Axis | Required assertion (not "no exception") |
|---|---|
| **Imports** | every `src/**/*.py` imports standalone (walk + `importlib`); `import src` works from an **installed** package, not just cwd (fixes B.1); cold-import wall time under a budget |
| **Configuration** | `to_dict`/`from_dict`/`save_yaml`/`from_yaml` round-trip is lossless for **every** field — walk `dataclasses.fields()` recursively and diff, not spot checks; mutating any field changes the dict; invalid values raise at construction. **Plus: a "config actually took effect" test** — set `mtf.enabled=False` / a minimal feature family and assert the produced column set changes (this is what would have caught B.2) |
| **Model registration** | registry equals the expected set exactly (fail on *extras* too — today's `count() >= 12` hides duplicates); every name instantiable; `requires_*` flags match `get_model_contract(name).input_rank`; **`is_available()` must not be allowed to convert a bug into a skip** |
| **Model construction** | default config instantiates; unknown config key rejected or explicitly documented as ignored; `get_default_config()` keys ⊇ what `fit` reads |
| **Training** | `TrainingMetrics.epochs_trained <= max_epochs` (catches silently-ignored config); loss decreases on `signal_ohlcv`; `is_fitted` flips; refit resets state |
| **Inference** | shapes; **probabilities sum to 1 ± 1e-5 per row** (today only bounds-checked); labels ⊂ declared set; `predict` is pure (same input twice → identical); 1-row predict works |
| **Serialization round-trip** | save→load→predict bit-identical for **every** model family (today only xgboost); scaler stats survive as float32; column-order independence; missing/extra feature raises |
| **Individual models** | 15 registered models × {construct, fit, predict, save/load, determinism}, matrix-driven |
| **Ensemble construction** | voting/blending/stacking each build; meta-learner really fitted (assert coefficients exist); **ensemble ≥ best single model on `signal_ohlcv`**; the diversity warnings must be assertable, not just logged |
| **Heterogeneous ensembles** | 2D+3D stacking end-to-end through `MLFactory`. Resolve the docs/code conflict on 2D+4D and 3D+4D (B.8) and test whichever is the truth |
| **Invalid combinations** | typed exception with a readable message, raised **before** training starts: unknown model · `build_ensemble` with 1 model · mixed ranks in voting · unknown meta-learner · unknown `training_mode` · `n_splits=1` · purge/embargo larger than data · horizon larger than data · unknown symbol. Assert exception *type* and that the message names the offending value |
| **Edge cases** | single class present · all-NaN column · constant column · rows < `seq_len` · rows < purge+embargo · duplicate timestamps · unsorted index · int-dtype volume · empty test split · **one model raising mid-ensemble** (others still produce results **and** the failure is reported in the result object, not only logged) |
| **End-to-end** | on `signal_ohlcv`: `success=True`, non-empty `metrics`, **non-empty `backtest_metrics` with `total_trades > 0`** (unconditional — no `if`), loadable bundle, deploy manifest naming a real model, and **`summary()` text containing the real F1** (catches B.6). On `noise_ohlcv`: completes with metrics *at* baseline. On `leaky_ohlcv`: must fail loudly |
| **Data-quality gates** | label distribution not more than X % single class · row-drop rate below a threshold · no 100 %-NaN feature columns survives · prediction/price alignment coverage above a threshold (all four would have fired on the real-data run) |
| **Backtest sanity** | `total_trades > 0` when signals exist, or an explicit typed "no trades because …" reason · `\|sharpe\| < 20` · `signals` sums to the covered bar count · a timezone assertion: the tradeable-bar fraction must match the data's own volume profile (catches B.5) |
| **Determinism / repeat** | same config+seed → identical metrics **and** identical saved-model bytes; run twice in-process and twice in fresh processes (catches global-state bleed: `ModelRegistry`, `_prepared_cache`, torch RNG, numba caches, the module-level label cache in `five_dimension_objective`) |
| **Failure recovery** | checkpoint → kill → `resume` reproduces the uninterrupted result; resume with a changed config refuses; corrupt checkpoint raises; the OOM-retry path's reduced batch size actually reaches the model |
| **Statistical validity** | purged/embargoed folds never overlap (already covered); OOF coverage equals expectation; **shuffling `y` destroys the score**; the DSR/PBO gate rejects a synthetic over-fit study |

## C.3 Combination matrix (4 CPU / 15 GB / no GPU)

Generated from one parametrization table so adding a model or mode is a one-line change.

### Tier 0 — "tiny", target ≤ 90 s wall, every change

| Dimension | Values |
|---|---|
| Dataset | `noise_ohlcv`, `signal_ohlcv` (6 000 rows @ 5 min) |
| Preprocessing | minimal features, robust scaler, MTF off |
| Model | `xgboost`, `lightgbm`, `lstm`, `patchtst`, `logistic` (one per rank/family archetype) |
| Horizon | `[5]` |
| CV | `PurgedKFold(n_splits=2, purge=15, embargo=30)` |
| Ensemble | none, `stacking(xgboost+lightgbm)` |
| Eval | metrics only |

Contents (not a full cross product):
* 5 models × `signal` × {fit, predict, save/load} = **15 model checks**
* 2 datasets × `xgboost` × full `MLFactory.run()` = **2 E2E runs** (the control pair)
* 1 stacking E2E = **1 run**
* ~40 pure-unit tests (config, registry, invalid combos, edge cases), no training

**Cost basis (OBSERVED):** the 2 500-row / 2-boosting-model / ensemble / bundle /
deploy E2E ran in **22.7 s** on a *contended* box; the direct control-pair fits took
seconds. 90 s with `-n 4` is realistic.

### Tier 1 — "full", target ≤ 20 min, pre-merge

| Dimension | Values |
|---|---|
| Dataset | `noise`, `signal`, `leaky`, `MES_1m_1week.parquet` |
| Preprocessing | {minimal / full features} × {MTF off / on} × {robust / standard} → sample 4 |
| Models | boosting ×3, rnn ×2, cnn ×3, transformer ×3, mlp ×1, classical ×3 = 15 |
| Training mode | `standard`, `walk_forward`, `regime_aware`, `meta_labeling` |
| Ensemble | none, voting(2D+2D), voting(3D+3D), stacking(2D+2D), stacking(2D+3D), stacking(4D+4D) |
| Eval path | metrics · backtest · bundle+reload · deploy manifest |
| Classes | `n_classes ∈ {2, 3}` |

Sampled, not crossed:

1. **Model sweep** — 15 models × `signal`, `max_epochs=2`, `seq_len=16`, `batch=64`,
   `n_estimators<=60` → ≈ 5 min
2. **Rank/ensemble sweep** — 6 ensemble configurations × `signal`, E2E → ≈ 6 min
3. **Mode sweep** — 4 training modes × `xgboost` × `signal` → ≈ 3 min
4. **Preprocessing sweep** — 4 variants × `xgboost` × real `MES_1m_1week` → ≈ 3 min
   (**note:** requires fixing B.2 first, otherwise the preprocessing axis is a no-op)
5. **Control pair + permutation** — noise/signal statistical assertions, 3 models → ≈ 2 min
6. **Guardrails** — `leaky` must fail; `n_classes=2` must complete → ≈ 1 min

Memory: 6 000 rows × ≤200 float32 features ≈ 5 MB per copy; 3D/4D tensors at
`seq_len=16` ≈ 30 MB. 15 GB is ample. Run `pytest -n 4` with the `rss_guard` fixture
capping any single test at 3 GB.

**Critical constraint discovered (OBSERVED):** on real 1-minute data the *unconditional*
209-feature pipeline plus `n_estimators=1000` costs ~16 minutes for a single model. Tier 1
is only feasible if (a) B.2 is fixed so a minimal feature set can be requested, and
(b) `n_estimators` is overridable per run. Otherwise cap the real-data axis at one
config and move it to Tier 2.

### Tier 2 — "soak", nightly/manual
Real 5-year MGC parquet, `n_splits=5`, full features + MTF, all 12 models, walk-forward.
Purpose: memory and wall-clock regression, not correctness.

## C.4 Harness hygiene rules (mechanically enforceable)

1. **Ban conditional assertions** — a meta-test that AST-walks `tests/` and fails any
   test function whose only `Assert` sits inside an `If`.
2. **Ban tautologies** — `assert x in (0, 1)`-style, and `assert result is not None` as
   a test's sole assertion.
3. **Retire the 26 source-grep tests** — convert to behavioural or delete (DECISIONS #11).
4. **Resolve the ~95 dead-module tests** — wire the modules or delete both (DECISIONS #3).
5. **Every skip needs a reason and an owner**; the suite fails if the skip count exceeds
   a pinned number. `is_available()`-driven skips become failures.
6. **Every documented "KNOWN BUG" becomes `pytest.mark.xfail(strict=True)`** so fixing
   it turns the test red and forces the docstring update.
7. **Pin ceilings** — `pandas>=2.1,<3`, `numpy<3`, `scikit-learn<2` in
   `requirements.txt`/`pyproject.toml`; a pinned CI job plus a nightly "latest" job that
   may fail loudly.
8. **Add `pytest-timeout`** (per-test cap), **`pytest-xdist`** (`-n 4`), and register
   markers (`slow`, `e2e`, `tier0`, `tier1`) so `--strict-markers` becomes useful.
9. **`pip install -e .` in CI** so `import src` does not depend on cwd.
10. **Coverage gate on `src/` only**, excluding dead modules, with a ratcheting floor.

## C.5 Build order

1. `conftest.py` with `noise_ohlcv` / `signal_ohlcv` / `leaky_ohlcv` + baseline helpers.
   Nothing else is provable without them. (Validated — §C.1.1.)
2. Tier-0 matrix (registry × rank × fit/predict/round-trip), replacing
   `test_model_smoke.py`, which trains on random labels and cannot detect a model that
   stopped learning.
3. Unconditional E2E assertions on `MLFactory.run()` — metrics, `total_trades > 0`,
   bundle load, `summary()` text, determinism — on the control pair.
4. The data-quality and backtest-sanity gates (§C.2), which would have caught every
   Part B finding.
5. Invalid-combination table (typed exceptions, fail-fast before training).
6. `strict_logs` fixture + a `--strict-swallow` flag so the 164 silent handlers become
   failures in CI.
7. Delete/convert the source-grep and dead-module tests; pin dependency ceilings.

---

# Appendix — reproduction commands

```bash
cd /home/user/Research

# Full suite (29m21s here, ~10-12m on an idle box)
./.venv/bin/python -m pytest tests/ -q -p no:cacheprovider

# The 4 failures, with tracebacks
./.venv/bin/python -m pytest tests/test_lookahead_audit.py -q --tb=short

# Real-data E2E (15m54s)  — script at scratchpad/e2e_run.py
PYTHONPATH=/home/user/Research ./.venv/bin/python \
  scratchpad/e2e_run.py data/raw/MES_1m_1week.parquet

# Synthetic ensemble E2E (22.7s) — scratchpad/e2e_ens2.py
PYTHONPATH=/home/user/Research ./.venv/bin/python scratchpad/e2e_ens2.py xgboost,lightgbm

# Control-pair proof (seconds) — scratchpad/ctrl_fast.py
PYTHONPATH=/home/user/Research ./.venv/bin/python scratchpad/ctrl_fast.py
```

Artifacts kept under
`/tmp/claude-0/-home-user-Research/920050fe-65f3-560d-aec9-d5f9c2df70b1/scratchpad/`:
`pytest_full.log`, `e2e_week.log`, `e2e_ens.log`, `e2e_run.py`, `e2e_ens2.py`,
`ctrl_fast.py`, `classify.py`.
