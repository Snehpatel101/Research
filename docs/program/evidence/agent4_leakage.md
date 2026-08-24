# Agent 4 — Statistical Correctness Audit (Data, Evaluation, Leakage)

Repo: `/home/user/Research`  ·  Date: 2026-08-24  ·  Method: source read, no code changes.
Markdown claims from `CLAUDE.md` / `COMPLETION.md` (114 phases) were treated as **unverified** and
checked against source. Tags: **OBSERVED** = read in source; **INFERRED** = deduced from
call-graph but not executed; **UNKNOWN** = could not determine statically.

---

## 0. Verdict up front

The **feature-engineering layer is genuinely rigorous** — shift(1) discipline is real and
near-universal, MTF resampling is correctly `closed='left', label='left'` + `shift(1)`, the
triple-barrier labeler is causal, and `PurgedKFold` / `WalkForwardEvaluator` index arithmetic is
correct. That part of the repo's claims holds up.

The **evaluation layer does not**. The numbers this system reports as its headline results are
in-sample-adjacent by construction: per-model metrics are validation-split metrics on the split
used for early stopping; the ensemble metric is a meta-learner val score contaminated by base-model
fold structure; the "financial report" (Sharpe, equity curve) is built from **train-split OOF
predictions against positionally-misaligned prices**; 10 of 12 model families never touch the held-out
test set at all; feature selection and hyperparameter search both consume the entire training set
before the CV that reports the score; and there is **no naive baseline anywhere in the codebase**.

Several documented "fixes" are absent or inert in code (see §7).

---

## 1. CRITICAL findings — capable of producing deceptively strong results

### C-1. Financial report / equity curve is computed from train-split OOF against misaligned prices
**Severity: CRITICAL** · OBSERVED
`src/models/training/feature_selection.py:678-758` (`_generate_financial_reports`), called from
`src/models/training/unified_orchestrator.py:458`.

```python
prices = df["close"].values                      # feature_selection.py:700  (FULL df)
...
predictions = oof.get_class_predictions()        # :719  (OOF over the TRAIN split only)
valid_mask = ~np.isnan(predictions)              # :722
prices_slice = prices[: len(oof.get_class_predictions())]   # :742  positional head-slice
```

Two compounding defects:

1. **Wrong evaluation population.** The equity curve, Sharpe, Sortino, Calmar and expectancy that
   the pipeline advertises as "realistic financial metrics" are computed on **out-of-fold predictions
   over the training split** — the same rows that drove feature selection (§C-2) and Optuna
   hyperparameter search (§C-3). They are never computed on the held-out test split.
2. **Positional misalignment.** `src/models/training/services/model_training.py:146` and
   `src/models/training/training_ops.py:399` call `prepared.filter_invalid_labels()` *before* OOF
   generation, dropping every `-99` row. `-99` rows occur both at the **head** (ATR warm-up:
   `src/data/labeling/triple_barrier.py:164-167` sets `labels[i] = -99` when `entry_atr` is NaN) and
   at the **tail** (`triple_barrier.py:225-231` marks the last `max_bars` rows). Sequence models
   additionally lose `seq_len - 1` leading rows. So `len(oof) < train_end` and the head-slice
   `prices[:len(oof)]` pairs prediction *k* with the price of bar *k*, not the price of the bar the
   prediction was actually made for. Every trade in the reported equity curve is attached to the
   wrong bar.

**Why deceptive:** produces a plausible-looking equity curve and Sharpe with no relationship to
either honest out-of-sample performance or to the model's actual signal. Direction of the bias is
not bounded — misalignment can inflate as easily as deflate, and the in-train population inflates
systematically.

**Fix:** compute financial metrics on the held-out **test** split via `Backtester`, joining
predictions to prices by timestamp/index (`PreparedData.test_indices` / DatetimeIndex), never by
positional head-slice. Keep the train-OOF curve only as a labelled diagnostic.

---

### C-2. Feature selection runs once, outside the CV that reports the score
**Severity: CRITICAL** · OBSERVED
`src/models/training/feature_selection.py:238-272` (`_run_feature_selection_on_train_data`),
`:274-446` (`_run_feature_selection_pipeline`).

The Phase-94 claim ("feature selection moved to train-only data") is **partly true**: line 264-265
does `train_end = int(n * self.config.train_ratio); df_train = df.iloc[:train_end]`, so the *test*
split is excluded. **VERIFIED.**

But the selection is then run **once on the whole training split** (MDA ranking at
`:286` → timeframe budget `:310` → correlation dedup `:370` → low-variance `:384` → per-model top-K
`:410-422`), and the OOF/CV that produces every reported number is run afterwards **on the
already-selected features** (`src/models/training/training_ops.py:400-410` →
`src/models/training/services/oof_generation.py:203`). Every fold's validation rows participated in
choosing the features that fold is scored on. This is textbook selection bias for the CV estimate,
and it is the single largest source of optimism in a 200+-feature financial pipeline.

`DECISIONS.md` acknowledges this ("C2 SKIPPED — feature selection inside CV folds, 2-week effort,
deferred"), but the pipeline still reports the contaminated CV numbers as if they were clean.

Two aggravating details:

- **No purge at the selection boundary.** `df.iloc[:train_end]` includes rows whose `label_h{h}`
  was computed from bars *inside* the test region (labels look forward `max_bars`). The last
  `max_bars` training rows carry test-period price information into the MDA ranking. (`purge_bars`
  is applied at `src/data/adapters/preparation.py:718` for the *model* split but not for this
  selection slice.)
- **Ranking is computed on temporally shuffled data** — see C-4.

**Fix:** nest selection inside each outer CV fold (train-fold only), or at minimum report a second,
honest estimate from a nested-CV run and label the fast number as biased. Apply
`purge_bars` when carving `df_train`.

---

### C-3. Hyperparameter search consumes the whole training set, then the same set is scored
**Severity: CRITICAL** · OBSERVED
`src/models/training/services/hyperparameter_tuning.py:51-107`,
`src/validation/cv/cv_tuner.py:195-256`.

Optuna correctly never touches `X_test` (`hyperparameter_tuning.py:106-107` uses
`prepared_data.X_train` only) — **that claim VERIFIED**. But:

- The same `X_train` is then used for OOF generation and for the reported metrics. Selecting
  hyperparameters by CV on a set and then reporting CV on that set is optimistically biased by
  construction, and no correction is applied by default (see below).
- `cv_tuner.py:198-218` does **no per-fold scaling** inside the objective — `X_train = X.iloc[train_idx].values`
  is used raw. The data was already globally scaled on the full train split (§H-2).
- `hyperparameter_tuning.py:91-94` builds `PurgedKFoldConfig(n_splits=..., embargo_bars=embargo)`
  and **omits `purge_bars`**, so the tuning CV silently uses the dataclass default `purge_bars=60`
  (`src/validation/cv/purged_kfold.py:61`). For `max(horizons) > 60` the tuning folds have
  insufficient purge and overlapping labels leak across the fold boundary.
- The DSR selection-bias correction is **inert in the default configuration**:
  `cv_tuner.py:271-286` only computes DSR when `is_sharpe_like_metric(self.metric)` is true, and
  the default metric is `f1_weighted` (`src/core/config.py:213`,
  `src/models/training/services/hyperparameter_tuning.py:23`). So Phase 66/87's "DSR gate
  enforcement" never fires on a default run. Even when it does fire in `cv_tuner`, the result is
  only written into a dict (`:297-306`) — nothing gates on it. The only real gate
  (`dsr_gate`, `enforce_dsr_gate=True`) lives in `src/optimization/five_dimension_objective.py:1172-1178`,
  which is not on the default training path.
- `compute_deflated_sharpe(..., num_tests: int = 1)` (`src/validation/deflated_sharpe.py:362-368`)
  defaults to no Bonferroni correction, so even the live path under-deflates when many models ×
  horizons are searched.

**Fix:** either nest tuning inside the outer CV, or hold out a tuning-free block. Pass
`purge_bars` explicitly. Make the default metric Sharpe-like (or add a bounded-metric selection-bias
correction) and make `should_deploy` actually block.

---

### C-4. MDA feature ranking is computed on randomly shuffled rows, defeating its own purged CV
**Severity: CRITICAL** · OBSERVED
`src/models/training/feature_selection.py:73-107`.

```python
from sklearn.model_selection import train_test_split
clean_df, _ = train_test_split(
    clean_df, train_size=mda_max_rows, stratify=clean_df[label_col], random_state=42,
)                                              # :81-88   shuffle=True by default
...
mda_cv = PurgedKFold(mda_cv_config)
cv_splits = list(mda_cv.split(X, y))           # :106-107  purge/embargo applied POSITIONALLY
```

`train_test_split` shuffles by default. After line 83 the rows are in **random temporal order**.
`PurgedKFold` then applies purge/embargo by *integer position* on that shuffled frame
(`src/validation/cv/purged_kfold.py:471-476`), so the purge window removes 60 arbitrary rows rather
than the 60 bars adjacent to the fold. Every fold's "training" set therefore contains rows
immediately adjacent in time to its validation rows, with fully overlapping label windows.

Consequence: MDA importance is measured under near-zero effective purging, which systematically
**over-ranks autocorrelated and slow-moving features** (levels, regime flags, session/vol state)
because the nearest-neighbour bar is in the training fold. Those features then become the selected
feature set for the whole pipeline.

`COMPLETION.md` Phase 103 claims "Strided temporal subsampling (replaces random stratified that
destroyed temporal structure)". That fix is **not present at this call site** — `train_test_split`
is still here. (Two more shuffled `train_test_split` calls exist in
`src/optimization/feature_selection/regime_selection.py:101` and
`src/optimization/feature_selection/economic_value.py:181`, and
`src/validation/ticker_portability.py:114`.)

**Fix:** replace with strided subsampling (`clean_df.iloc[::stride]`) or a contiguous block sample;
never shuffle before a positional purged split.

---

### C-5. Future-derived columns are not excluded from candidate features (latent, path-dependent)
**Severity: CRITICAL (conditional)** · OBSERVED for the gap; INFERRED for exploitability

The 12-stage data pipeline writes pure-future columns into the DataFrame:

| Column | Written at | Content |
|---|---|---|
| `bars_to_hit_h{h}`, `mae_h{h}`, `mfe_h{h}` | `src/data/pipeline/stages/labeling/run.py:329-332` | barrier outcome |
| `fwd_return_h{h}`, `fwd_return_log_h{h}` | `src/data/pipeline/stages/final_labels/core.py:234-239` (`close.shift(-horizon)`) | the forward return itself |
| `quality_h{h}`, `pain_to_gain`, `time_weighted_dd` | `src/data/pipeline/stages/final_labels/core.py:305-317` | MAE/MFE-derived |

The repo *has* a correct exclusion list — `identify_feature_columns()` in
`src/optimization/feature_selection/filtering.py:70-84` excludes `bars_to_hit_`, `mae_`, `mfe_`,
`fwd_return_`, `quality_`, `touch_type_`, `pain_to_gain_`, `time_weighted_dd_`.

**It is not used by the training path.** `src/models/training/feature_selection.py:18-21` imports
only `filter_correlated_features` and `filter_low_variance`, and hand-rolls its own exclusion at
`:251-257`:

```python
exclude_patterns = ["label", "sample_weight", "datetime", "date", "time"]
ohlcv_cols = ["open", "high", "low", "close", "volume"]
```

`mae_h20`, `mfe_h20`, `bars_to_hit_h20`, `fwd_return_h20` all survive this filter. The adapter's
auto-detect fallback `BaseAdapter._get_feature_columns` (`src/data/adapters/base.py:332-379`) has the
same gap — its `exclude_prefixes` covers `label_`, `target_`, `meta_`, `regime_`, `mtf_raw_` but
none of the barrier-outcome prefixes.

The default `MLFactory` path is currently safe: `src/factory.py:767-795` labels via
`TripleBarrierLabeler` and writes only `label_h{h}` and `label`, discarding the metadata dict. But
any run that ingests a parquet/CSV produced by the stages pipeline (supported —
`factory.py` auto-detects file extension) feeds these columns straight into MDA ranking, where
`fwd_return_h{h}` would rank first and give near-perfect scores.

**Fix:** make `identify_feature_columns()` the single source of truth; call it from
`feature_selection.py:251` and from `BaseAdapter._get_feature_columns`. Add an assertion that no
candidate feature name matches the label-prefix set.

---

## 2. HIGH findings

### H-1. Ensemble headline metric is a contaminated meta-learner validation score
**Severity: HIGH** · OBSERVED
`src/models/training/services/ensemble_service.py:389-434`.

```python
n_train = int(n_samples * 0.8)
X_train = X_stack.iloc[:n_train].values ; X_val = X_stack.iloc[n_train:].values   # :390-396
...
val_f1 = f1_score(y_val, output.class_predictions, average="macro", ...)          # :430-432
metrics = {"val_f1": val_f1, "val_accuracy": val_accuracy, ...}                   # :438-444
```

Three problems:

1. **Base-model fold structure violates the meta split.** Stacking rows 0–80 % are OOF predictions
   from `PurgedKFold` folds 0–3, and each of those fold models was trained on the *other* folds —
   including fold 4, which is exactly the 80–100 % meta-validation block. Information from the
   meta-val period is baked into the meta-train features.
2. **No purge/embargo at the 80/20 meta split.** Plain `iloc` cut; labels straddling the boundary
   overlap.
3. This `val_f1` is what becomes `ensemble_metrics` (`ensemble_service.py:210` →
   `unified_orchestrator.py:551-558`) and is reported as the ensemble result. There is no test-split
   ensemble evaluation.

**Fix:** hold out a contiguous, purged final block from the base-model CV entirely, generate OOF
only over the earlier region, and evaluate the meta-learner on the untouched block.

### H-2. Scaler is fit on the full training split, then CV folds are drawn inside it
**Severity: HIGH** · OBSERVED
`src/data/adapters/preparation.py:597-603` fits `AdapterScaler` once on `train_result.X`
(the entire train split) and returns `PreparedData.X_train` already scaled. OOF then re-scales
per-fold on that already-scaled array (`src/validation/cv/oof_core.py:246-249`,
`src/models/training/services/oof_generation.py:289-297`).

The per-fold `FoldAwareScaler` (`src/validation/cv/fold_scaling.py:84-163`) is itself **correct**
(train-only fit, fresh instance per fold) — claim VERIFIED. But it is applied on top of a global
transform whose median/IQR were computed over every fold's validation rows. The leak is
low-information (two robust location/scale statistics) but it is unambiguously a global
fit-then-split, and the Optuna objective (`cv_tuner.py:198-218`) skips per-fold scaling entirely,
inheriting only the global transform.

**Fix:** carry raw features into `PreparedData` (`apply_scaling=False` for the CV path) and let
`FoldAwareScaler` be the only scaler inside CV; keep the global scaler only for the final refit /
inference bundle.

### H-3. `_validate_lookahead` is validation theatre — it always passes
**Severity: HIGH** · OBSERVED
`src/models/training/feature_selection.py:539-563`:

```python
is_valid, issues = validate_resample_config(closed="left", label="left")   # :549-552
```

The arguments are **string literals**, not the run's configuration and not any feature. The
function is described in the codebase and logs as a "mandatory lookahead audit that cannot be
disabled" (`:183-184`, `:196-198`), and the pipeline prints "Lookahead audit passed" on every run.
It audits nothing. The real machinery — `LookaheadAuditor.audit_feature_function`,
`audit_feature_lookahead`, `scan_dependency_propagation`, `verify_resampling_parity` in
`src/validation/lookahead_audit.py:162-760` — is never invoked from the training path.

**Why deceptive:** gives operators a green "PASSED" that carries zero information, discouraging the
manual audit that would find C-5 or a future un-shifted feature.

### H-4. Leakage detector has near-zero statistical power and is non-blocking in practice
**Severity: HIGH** · OBSERVED
`src/models/training/feature_selection.py:492-537` →
`src/validation/leakage_detection.py:116-...`.

- Threshold is `|Spearman ρ| > 0.5` against a **discrete 3-class label**
  (`src/models/config/trainer_config.py:122: validation_correlation_threshold = 0.5`). A single
  feature would have to be a near-deterministic function of the label to trip it. `fwd_return_h20`
  (§C-5) would trip it; virtually nothing else — including a fully leaking 30-feature combination — would.
- It is run on the **full `df`** including the test split (`unified_orchestrator.py:399` →
  `_pre_training_validation(df)`), before the train-only slice is taken.
- The whole check is wrapped in `except Exception as e: warnings.append(...)` (`:536-537`), so any
  failure (dtype, NaN, shape) silently downgrades a blocking check to a warning.
- Multivariate leakage (the realistic case) is never tested — no train-vs-test adversarial
  classifier, no permutation-null comparison.

### H-5. No held-out test evaluation for 3D/4D models — 10 of 12 model families
**Severity: HIGH** · OBSERVED
`src/models/training/services/model_training.py:157-167` routes `data_rank >= 3` to
`Trainer.run_prepared()`, which returns `"test_metrics": None  # Test eval not supported in
run_prepared yet` (`src/models/training/trainer.py:1104`). Every RNN, CNN, and transformer family
(LSTM, GRU, TCN, InceptionTime, 1D ResNet, PatchTST, iTransformer, TFT, N-BEATS) therefore has **no
generalization estimate at all**; only the 2D boosting/classical path reaches
`_evaluate_test_set` (`trainer.py:871-872`).

Compounding: `ModelTrainingResult.metrics` is set to `training_results["evaluation_metrics"]`
(`model_training.py:173`), i.e. **validation-split** metrics — the same split used for early
stopping and for `Trainer`'s calibrator. `TrainingRunResult.best_model`
(`unified_orchestrator.py:150`) then selects the best model by `val_f1`. Selection and reporting
happen on the same split.

### H-6. No naive baseline anywhere in the evaluation stack
**Severity: HIGH** · OBSERVED (exhaustive grep)
`grep -rni "baseline" src/` returns hits only inside feature-governance utilities
(`label_perturbation.py`, `economic_value.py`) — none in `src/models/evaluation/`,
`src/validation/evaluation/`, `src/inference/backtesting/`, or `src/models/training/evaluation.py`.

There is no always-neutral / majority-class classifier, no buy-and-hold equity curve, no
random-signal null, no shuffled-label control. Combined with the default optimisation metric
`f1_weighted` (`src/core/config.py:213`) on a label distribution dominated by the `0` (neutral)
class, an "always predict neutral" model would score well and no reported number would reveal it.

**Fix:** emit majority-class and buy-and-hold rows alongside every metrics table and equity curve;
add a shuffled-label run as a smoke test in CI.

### H-7. Sample weights are derived from future trade outcomes, with global percentiles
**Severity: HIGH** · OBSERVED
`src/data/pipeline/stages/final_labels/core.py:110-227`.

`quality_scores` blends `speed_scores` (bars_to_hit), `mae_scores`, `mfe_scores`, `ptg_scores`
(pain-to-gain), `twdd_scores` — all functions of what happened *after* the entry bar — and each is
normalised with a **whole-dataset percentile** (`np.percentile(valid_twdd, 95)` at `:180`, and the
tier cuts `p20`/`p80` at `:214-216`). `assign_sample_weights` then upweights the top 20 % of
outcomes 1.5× and downweights the worst 20 % to 0.5×, and the result is written to
`sample_weight_h{h}` (`:314`) and passed to `model.fit(..., sample_weight=...)`.

Two distinct issues: (a) whole-dataset percentiles are a global statistic computed with test-period
data; (b) weighting training samples by how *cleanly* their future resolved makes the model learn an
easier problem than the one it faces at inference, and any weighted metric (`logloss_weighted`) is
correspondingly inflated. Lopez de Prado's sample weights are based on **label uniqueness/overlap**,
which is causal — this is outcome quality, which is not.

---

## 3. MEDIUM findings

### M-1. CPCV purge/embargo are dataset-size fractions, decoupled from the label horizon
`src/validation/cv/cpcv.py:61-62` (`purge_pct = embargo_pct = 0.01`), sizes computed at `:263-267`
as `int(n_samples * pct)`. `src/models/training/services/hyperparameter_tuning.py:171-177` hardcodes
`purge_pct=0.01, embargo_pct=0.01`. On a 6,000-row train split that is 60 bars; on a 2,000-row split,
20 bars — below `max(horizons)` for the documented horizon set `[5,10,20,60,120]`. Purge must be
`>= max_bars` of the barrier, not a fraction of dataset length.
Also `:306-313`: when `len(all_test_combos) > max_combinations`, combinations are sub-sampled by
`int(i * step)` — a deterministic non-uniform slice, not a random or exhaustive one.

### M-2. Transaction-cost calibration uses a whole-dataset ATR median
`src/data/labeling/triple_barrier.py:598-605`:
```python
median_atr = np.median(valid_atr)          # entire dataset, test period included
return float(cost_in_price / median_atr)
```
The inline comment explicitly defends the global median. This directly **contradicts** the
`COMPLETION.md` Phase 96 C13 claim — *"Triple barrier ATR uses expanding median for training-time
cost calibration"*. There is no expanding median in this file. The magnitude of the leak is small
(one scalar), but the barrier width — and therefore the entire label distribution — is set using
future volatility. **Claim DISPROVEN.**

### M-3. `PurgedKFold` label-aware purge is dead code on the main path
`src/validation/cv/purged_kfold.py:482-496` implements label-overlap purging correctly, but
`OOFGenerationService` never supplies `label_end_times`
(`src/models/training/services/oof_generation.py:203-208`, `:272`, and `_create_cv` at `:81-88`).
Only `stacking.py:739`, `manager.py:227`, `walk_forward.py:377` and `trainer.py:642` pass it.
So the primary OOF path relies purely on the fixed `purge_bars` count. That is adequate *iff*
`purge_bars >= max_bars`, which `src/config/experiment.py:222-230` enforces only as
`purge_bars = max_h` (exactly, no margin) and only in `ExperimentConfig` — `PipelineConfig` /
`TrainerConfig` carry an independent default of 60 with no such guard.

Also, the code comment at `:490-492` claims the mask handles "samples after embargo whose labels
started during test period", but the mask condition `index_values <= test_end_time` excludes all
post-test rows. Harmless (forward labels can't reach backwards) but the comment is wrong.

### M-4. Fold-leakage validator logs but never blocks
`src/models/training/services/oof_generation.py:213-223` and `:350-360` call
`OOFValidator.validate_fold_leakage` and, on failure, emit `logger.error(...)` and continue. The
validator itself (`src/validation/cv/oof_validation.py:28-120`) is genuine and correct — it verifies
that no train index sits in `[val_start - purge, val_start)` or `(val_end, val_end + embargo]`. It
should raise.

### M-5. OBV has no session reset — documented fix does not exist
`src/data/pipeline/stages/features/volume.py:48` and `:153`:
`(np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()` — a global cumulative sum over the
entire history, then `.shift(1)`. `grep -rn "session_cumsum" src/` returns **zero hits**, so the
`COMPLETION.md` Phase 94 claim — *"Cumulative features (OBV/VWAP/TWAP/cum_order_flow) reset at
session boundaries via `session_cumsum()`"* — is **DISPROVEN for OBV**. VWAP *is* session-reset
(`volume.py:106-108`, `groupby("date").cumsum()`) — that half of the claim holds. A global OBV is not
a lookahead but is a monotone-ish proxy for row index, letting models memorise temporal position.

### M-6. Duplicate, contradictory CV config classes
`src/config/cv.py:154-193` defines a `PurgedKFoldConfig` whose docstring says
*"This is the CANONICAL PurgedKFoldConfig … `src/validation/cv/purged_kfold.py` (deprecated)"* — but
every production call site imports the "deprecated" one
(`oof_generation.py:15`, `stacking.py:676`, `hyperparameter_tuning.py:66`, `cv_tuner`,
`unified_orchestrator.py:272`, `training_ops.py:1027`, `feature_selection.py:101`,
`cv_evaluator.py:105`, `cv_feature_selection.py:131`, `cv_orchestrator.py:284`). The "canonical"
class also carries `use_label_end_times: bool = True`, a flag the executing class does not have.
Config drift risk on the single most safety-critical parameter set.

### M-7. Walk-forward `gap_bars` defaults to 0 in the mode config
`src/models/training/modes/walk_forward.py:82-83`: `gap_bars: int = 0`, `embargo_bars: int = 0`.
`src/models/training/training_ops.py:439-441` does wire `gap_bars=self.config.purge_bars`, so the
orchestrated path is fine — but any direct construction of `WalkForwardTrainerConfig` (and the CLI:
`src/cli/commands/evaluate.py:415`, `--gap-bars` default `0`) yields **train ending on the bar
immediately before test**, with `max_bars` of label overlap. The `WalkForwardConfig` split logic
itself (`src/validation/cv/walk_forward.py:263-323`) is correct: `train_end = test_start - gap_bars`
(`:269`), expanding/rolling both strictly before test, embargo applied after every prior test window
(`:288-299`). **Walk-forward index arithmetic VERIFIED correct.**

### M-8. `set_all_seeds` is only called for neural models; determinism is off by default
`src/core/reproducibility.py:99-160` is correct and comprehensive (python `random`, numpy, torch CPU,
torch CUDA, optional `use_deterministic_algorithms`). It is invoked from exactly one place:
`src/models/neural/base_rnn.py:344-346`. There is no seeding at pipeline entry, so numpy global state
for feature selection, boosting, and ensembles depends on call order.
`deterministic_mode: bool = False` (`src/models/config/trainer_config.py:75`) means cuDNN autotune and
non-deterministic GPU reductions are active; combined with `torch.compile` max-autotune and
`n_jobs=-1` (both documented as Phase 68 defaults), GPU runs are **not bit-reproducible**. Individual
`random_state=42` values are passed widely, so results are *approximately* stable on CPU.

### M-9. Calibrator would be self-referential if enabled
`src/validation/cv/oof_core.py:322-396`: `calibrate_oof_predictions` fits `ProbabilityCalibrator` on
the OOF probabilities (`:379`) and writes the calibrated values back into the same frame
(`:384-389`), which then feeds stacking. The docstring at `:328-341` calls this "leakage-safe"; it is
not — the OOF probs are out-of-sample w.r.t. the *base model* but in-sample w.r.t. the *calibrator*.
The inline NOTE at `:368-373` shows partial awareness. **Currently dormant**: `calibrate` defaults to
`False` (`src/validation/cv/oof_generator.py:117`) and `OOFGenerationService` never sets it. Latent.

### M-10. `merge_asof` NaN fill maps early anchors to a future higher-TF bar
`src/data/adapters/multi_stream.py:660-669`: `merged["higher_pos"].fillna(0)` — anchor bars that
precede the first higher-TF bar are mapped to higher-TF index 0, which is *after* them. Affects only
the first few rows of a 4D stream. Should be dropped, not filled.

---

## 4. LOW findings

- **L-1** `src/models/metrics.py:366-369` — `compute_trading_metrics` builds a **per-trade** return
  series and annualises it with a hardcoded `periods_per_year=252`, i.e. treats one trade as one
  trading day. Arbitrary by orders of magnitude in either direction. (The `Backtester` path is
  correct: `src/inference/backtesting/backtest.py:1048-1062` derives `periods_per_year` from the
  actual timestamp span, and `equity_curve.py:247-248` re-derives when the caller left the default.
  **That annualization claim VERIFIED.**) `src/models/evaluation/financial_report.py:43` also
  hardcodes 252; `src/cli/commands/evaluate.py:579` and
  `src/data/pipeline/stages/evaluation/run.py:193` hardcode `np.sqrt(252)`.
- **L-2** Backtest entry-vs-label mismatch: the labeler enters at `close[t]`
  (`triple_barrier.py:162`) while the backtest enters at `open[t]` under the default
  `MARKET_ON_OPEN` (`backtest.py:83`, `:469`). Both are causal — features at `t` are shift(1)'d, so
  `open[t]` is knowable — so this is **not** lookahead, but the two are optimising different games
  and the reported hit-rate will not translate to the reported P&L. Exit ordering is correct: the
  exit check at `backtest.py:895-902` runs before the entry check, so a position cannot be closed on
  its own entry bar.
- **L-3** CPCV label-aware purge (`cpcv.py:343-358`) is an O(n) Python loop per test group, and
  `pbo.py:137,371` hardcodes `annualization=252.0`.
- **L-4** PBO / CPCV-PBO evaluation is dead: `CPCVPBOEvaluator` is exported from
  `src/validation/evaluation/__init__.py:17` but referenced nowhere else in `src/`. CPCV itself is
  reachable only via `cv_method="cpcv"`, default `"purged_kfold"` (`src/config/experiment.py:108`).
  So the pipeline ships an overfitting-probability measurement it never runs.
- **L-5** `ConformalPredictor` (`src/models/calibration/conformal.py:137-141`) documents itself as
  *"Available but not yet wired into the automated pipeline"* — accurate; grep confirms no
  construction outside its own docstrings. So the uncertainty-quantification claim is aspirational,
  not a leakage risk.

---

## 5. What the audit VERIFIED as correct

These are real and worth protecting:

- **`PurgedKFold` index arithmetic** — `src/validation/cv/purged_kfold.py:454-508`. Test block,
  purge `[test_start - purge, test_start)`, embargo `[test_end, test_end + embargo)`, mask-based so
  training data on both sides is handled. Correct. (Purge before / embargo after is the correct
  asymmetry per Lopez de Prado; "embargo on both sides" is not the standard and is not needed.)
- **`WalkForwardEvaluator`** — `walk_forward.py:263-323`. Train strictly precedes test with a
  `gap_bars` gap; embargo zones after *all* prior test windows removed from training.
- **MTF anti-lookahead** — `src/data/pipeline/stages/mtf/generator.py:189` (`closed="left",
  label="left"`) + `:334` (`df_mtf_idx.shift(1)`) + `:337` (`reindex(..., method="ffill")`).
  Semantically correct. The 4D path matches: `src/factory.py:608-615` and
  `src/inference/bundle.py:1134-1137` both resample-then-`shift(1)`, and
  `src/data/adapters/multi_stream.py:660-665` aligns with `merge_asof(direction="backward")`.
  **Training/inference parity VERIFIED.**
- **Triple-barrier causality** — `src/data/labeling/triple_barrier.py:160-231`: entry `close[i]`,
  entry ATR `atr[i]`, barrier walk `for j in range(1, min(max_bars+1, n-i))` starting at `i+1`,
  trailing `max_bars` rows marked `-99`. Correct.
- **ATR is Wilder's EMA (`alpha = 1/period`)** in the labeler
  (`triple_barrier.py:550-555`) — matches the Phase 94 parity claim.
- **Cost symmetry** — `triple_barrier.py:382-383`: `k_up_effective = k_up + cost_in_atr`,
  `k_down_effective = k_down + cost_in_atr`. Symmetric. **VERIFIED.**
- **`FoldAwareScaler`** — `fold_scaling.py:84-163`: fresh scaler per fold, fit on train only,
  float32 fast path uses train-derived median/IQR for both arrays. Correct in isolation.
- **Feature shift(1) discipline** — spot-checked across `moving_averages.py:50,94`,
  `trend.py:59-64,181-184` (ADX + Supertrend), `volume.py:49-65`, `entropy.py:302,548,828,1124,1381`,
  `regime.py:104,225,284`, `hmm.py:468-469`. Repo-wide grep for `.shift(-`, `center=True`, `bfill`,
  `backfill` returns exactly **one** hit — `final_labels/core.py:234`, which is deliberate target
  construction. This is unusually disciplined.
- **HMM regime detection is causal** — `src/data/pipeline/stages/regime/hmm.py:355-425` refits on an
  expanding or rolling window per bar; no whole-dataset fit. Plus `shift(1)` at `:468-469`.
- **Temporal features** (`temporal.py:38-60`) are known at bar `t`; correctly unshifted.
- **NaN handling** is `dropna` (`nan_handling.py:156`) — no global-statistic imputation anywhere.
- **Raw OHLCV and label columns are excluded** from adapter auto-detect
  (`base.py:350-372`).
- **Meta-labeling uses OOF primary predictions**, not in-sample —
  `training_ops.py:879-895`. **Phase 103 claim VERIFIED.**
- **Optuna never touches the test split** — `hyperparameter_tuning.py:106-107`.
- **Backtest annualization is data-derived** — `backtest.py:1048-1062`.
- **Feature selection excludes the test split** — `feature_selection.py:264-265`. Phase 94 claim
  **partially** verified (see C-2 for what it does not fix).

---

## 6. Ranking by "capable of producing deceptively strong results"

| # | Finding | Sev | File:line |
|---|---|---|---|
| 1 | Financial report built from train-OOF against misaligned prices | CRIT | `models/training/feature_selection.py:700,742` |
| 2 | Feature selection once, outside the reporting CV | CRIT | `models/training/feature_selection.py:238-446` |
| 3 | Optuna on the same set that is then scored; DSR gate inert; purge omitted | CRIT | `services/hyperparameter_tuning.py:91-94`; `cv/cv_tuner.py:271-286` |
| 4 | MDA ranking on shuffled rows defeats its purged CV | CRIT | `models/training/feature_selection.py:81-107` |
| 5 | Future-derived columns not excluded from candidate features | CRIT* | `models/training/feature_selection.py:251`; `adapters/base.py:332-379` |
| 6 | Ensemble metric = contaminated meta-learner val score | HIGH | `services/ensemble_service.py:389-434` |
| 7 | No naive baseline anywhere | HIGH | (absent) |
| 8 | No test evaluation for 3D/4D models (10 of 12 families) | HIGH | `models/training/trainer.py:1104` |
| 9 | Lookahead audit is a no-op that always passes | HIGH | `models/training/feature_selection.py:549-552` |
| 10 | Leakage detector has no power; non-blocking on exception | HIGH | `models/training/feature_selection.py:492-537` |
| 11 | Global scaler fit before CV folds are drawn | HIGH | `adapters/preparation.py:597-603` |
| 12 | Sample weights from future outcome quality + global percentiles | HIGH | `stages/final_labels/core.py:110-227` |
| 13 | CPCV purge/embargo are dataset fractions, not horizon-based | MED | `cv/cpcv.py:61-62,263-267` |
| 14 | Global ATR median for cost calibration (claim disproven) | MED | `labeling/triple_barrier.py:598-605` |
| 15 | `label_end_times` never supplied on the main OOF path | MED | `services/oof_generation.py:81-88,203` |
| 16 | Fold-leakage validator logs instead of raising | MED | `services/oof_generation.py:218-223` |
| 17 | OBV global cumsum; `session_cumsum()` does not exist | MED | `features/volume.py:48,153` |
| 18 | Duplicate/contradictory `PurgedKFoldConfig` | MED | `config/cv.py:154` vs `validation/cv/purged_kfold.py:31` |
| 19 | Walk-forward `gap_bars` default 0 outside the orchestrator | MED | `training/modes/walk_forward.py:82` |
| 20 | Seeds only for neural; `deterministic_mode=False` | MED | `core/reproducibility.py:99`; `config/trainer_config.py:75` |
| 21 | Calibrator self-referential (dormant) | MED | `cv/oof_core.py:379-389` |
| 22 | 4D `merge_asof` NaN → index 0 (future bar) | MED | `adapters/multi_stream.py:667` |
| 23 | Per-trade Sharpe annualised at 252 | LOW | `models/metrics.py:366-369` |
| 24 | Label enters at close[t], backtest at open[t] | LOW | `triple_barrier.py:162`; `backtest.py:469` |
| 25 | PBO / CPCV-PBO / conformal are dead code | LOW | `validation/evaluation/__init__.py:17` |

\* CRIT conditional on the input DataFrame originating from the 12-stage pipeline rather than
`MLFactory`'s own labeling.

---

## 7. Documentation claims checked against source

| Claim (`CLAUDE.md` / `COMPLETION.md`) | Status |
|---|---|
| Phase 94: "Feature selection moved to train-only data" | **PARTLY TRUE** — test excluded (`feature_selection.py:264`), but still outside CV and unpurged at the boundary |
| Phase 94: "Cumulative features reset at session boundaries via `session_cumsum()`" | **DISPROVEN** — symbol does not exist; OBV is a global cumsum. VWAP *is* session-reset. |
| Phase 96 C13: "Triple barrier ATR uses expanding median for cost calibration" | **DISPROVEN** — `np.median(valid_atr)` over the whole dataset (`triple_barrier.py:604`) |
| Phase 103: "Strided temporal subsampling replaces random stratified" | **DISPROVEN at the MDA site** — `train_test_split(..., stratify=...)` still at `feature_selection.py:83` |
| Phase 103: "Meta-label OOF predictions replace in-sample" | **VERIFIED** (`training_ops.py:884`) |
| Phase 103: "4D OHLCV anti-lookahead shift(1) in factory + bundle" | **VERIFIED** (`factory.py:615`, `bundle.py:1137`) |
| Phase 103: "HP tuning embargo wired from PipelineConfig" | **VERIFIED for embargo**; `purge_bars` still omitted (`hyperparameter_tuning.py:91-94`) |
| Phase 66/87: "DSR gate enforcement" | **INERT by default** — default metric `f1_weighted` skips DSR; nothing gates on `should_deploy` on the training path |
| Phase 95: "Sharpe annualization derived from data frequency" | **VERIFIED for the Backtester** (`backtest.py:1048-1062`); still hardcoded 252 in `models/metrics.py:366`, `financial_report.py:43`, `cli/commands/evaluate.py:579` |
| Phase 95: "shift(1) added to all 12 entropy compute functions" | **5 shifted `add_*` functions found** (`entropy.py:302,548,828,1124,1381`); count unverified — the ones present are correct |
| Phase 94: "ATR unified to Wilder's EMA in labeling + backtest" | **VERIFIED** (`triple_barrier.py:550-555`) |
| "Mandatory lookahead audit that cannot be disabled" | **DISPROVEN** — no-op with literal args (`feature_selection.py:549`) |
| "No data leakage — purge/embargo in all CV splits" (CLAUDE.md guarantee table) | **NOT SUPPORTED** — the splits themselves are correct, but feature selection, hyperparameter search, scaling, and the reported metrics all sit outside the purged boundary |

---

## 8. Recommended remediation order

1. **Move every reported number to the held-out test split**, and back the equity curve with
   timestamp-joined `Backtester` output. (C-1, H-5) — highest value, lowest effort.
2. **Add a naive-baseline row** (majority class, buy-and-hold, shuffled-label) to every metrics
   table. (H-6) — trivial effort, immediately exposes 1 and 2.
3. **Route all candidate-feature discovery through `identify_feature_columns()`** and assert on
   label-prefix collisions. (C-5) — small, closes a catastrophic landmine.
4. **Replace `train_test_split` with strided sampling** at `feature_selection.py:83`. (C-4) — one line.
5. **Pass `purge_bars` in `hyperparameter_tuning.py:91`**; make `OOFValidator` raise. (C-3, M-4).
6. **Delete or implement `_validate_lookahead`** — a no-op that prints PASSED is worse than nothing.
   (H-3)
7. **Make the CV path consume unscaled features**, with `FoldAwareScaler` as the only in-CV scaler.
   (H-2)
8. **Purge the meta-learner split** and hold out a base-model-free final block. (H-1)
9. **Nested CV for feature selection** — the acknowledged two-week item. Until it lands, label the
   current CV numbers as biased in every report. (C-2)
10. Consolidate `PurgedKFoldConfig` to one class; call `set_all_seeds` at pipeline entry; make
    horizon-relative purge mandatory in CPCV. (M-6, M-8, M-1)
