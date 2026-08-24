# Agent 3 — Ensemble Architecture Research

**Scope:** `/home/user/Research` ensemble subsystem + state-of-the-art research on heterogeneous model combination.
**Date:** 2026-08-24. **Commit at time of review:** `f559758` (post Phase 114).
**No code was changed.**

Every claim is tagged:
- **[OBSERVED]** — read directly in the repo, file:line given.
- **[INFERRED]** — logically follows from observed code, but not executed/proven at runtime.
- **[PROPOSED]** — my design recommendation.
- **[UNKNOWN]** — could not determine from static reading.

---

# PART A — WHAT EXISTS TODAY

## A0. Map of the ensemble surface

**[OBSERVED]** `src/models/ensemble/` — 16 modules, 8,604 lines:

| File | Lines | Role |
|---|---|---|
| `orchestrator.py` | 730 | `EnsembleOrchestrator`, `EnsembleResult` — declared "THE single entry point" |
| `stacking.py` | 1,398 | `StackingEnsemble` (a `BaseModel`, self-contained OOF + meta-learner) |
| `diversity.py` | 1,021 | Q-stat, disagreement, double-fault, entropy, KL, MCC, composite score, greedy selection |
| `voting.py` | 672 | `VotingEnsemble` — soft/hard vote |
| `second_level.py` | 715 | `SecondLevelStacker` — cluster models, per-cluster meta-learner, final meta-learner |
| `meta_factory.py` | 575 | `MetaLearnerFactory` + `register_meta_learner` decorator registry |
| `blending.py` | 516 | `BlendingEnsemble` — holdout blend |
| `meta_selection.py` | 517 | Optuna search over meta-learner type + hyperparams |
| `heterogeneous_stacking.py` | 474 | `HeterogeneousStackingBuilder` (**duplicate of** `src/validation/cv/oof_stacking.py:337`) |
| `validator.py` | 382 | Input-rank compatibility rules for ensembles |
| `xgboost_meta.py` / `mlp_meta.py` / `ridge_meta.py` / `calibrated_meta.py` | 356/325/295/341 | The 4 meta-learners |
| `meta_base.py` | 27 | just `softmax()` |
| `meta_learners/__init__.py` | 26 | pure re-export shim |

**[OBSERVED]** OOF machinery is **not** in `src/models/training/services/` as the brief assumed. It lives in `src/validation/cv/`:
`oof_core.py`, `oof_sequence.py`, `oof_generator.py`, `oof_stacking.py`, `oof_alignment.py`(→ actually `src/validation/cv/oof_*` plus `src/data/adapters/alignment.py`), `oof_cache.py`, `oof_io.py`, `oof_validation.py`.
`src/models/training/services/oof_generation.py` is a **thin service wrapper** (`OOFGenerationService`) plus its own private 4D path.

---

## A1. Ensemble strategies actually implemented

### A1.1 Soft / hard voting — `src/models/ensemble/voting.py`
**[OBSERVED]**
- `VotingEnsemble._soft_vote()` (`voting.py:478`): stacks `o.class_probabilities` from each base model into `(n_models, n, C)`, applies optional normalized weights (`voting.py:73`, weights set in `set_base_models`, `voting.py:254`), sums or means over models, `argmax`, then `map_classes_to_labels(idx, self._n_classes)`.
- `VotingEnsemble._hard_vote()` (`voting.py:517`): per-sample `np.unique(...counts)`, ties broken by `np.random.RandomState(seed=42 + i)` (deterministic per-sample), confidence = vote fraction. Probabilities returned are the *unweighted mean* even in hard-vote mode (`voting.py:551`) — an inconsistency, hard votes and reported probabilities can disagree.
- Weights are **static, user-supplied, never learned** (`voting.py:296-306`). **[OBSERVED]** grep for `class .*Weight|def .*weight` in `src/models/ensemble/*.py` and `src/inference/*.py` returns **nothing** — there is no weight-optimization module anywhere.

**[OBSERVED] HARD BLOCKER for heterogeneity:** `voting.py:281-294` raises `ValueError("Cannot mix base models with different input ranks in VotingEnsemble")` if the base models don't all share rank 2/3/4. Voting is **homogeneous-rank only, by construction.**

### A1.2 Blending — `src/models/ensemble/blending.py`
**[OBSERVED]** `BlendingEnsemble` (`blending.py:34`) — holdout-split blend, single `X` array input (`fit(X_train, y_train, X_val, y_val)`, `blending.py:155`). Same single-tensor input assumption ⇒ **[INFERRED]** cannot serve 2D+4D mixes.

### A1.3 Stacking as a model — `src/models/ensemble/stacking.py`
**[OBSERVED]** `StackingEnsemble` is a `BaseModel` that does its *own* internal OOF via PurgedKFold (`stacking.py:609 _generate_oof_predictions`), then trains a meta-learner and refits base models on full train.
- Defaults (`stacking.py:243`): `n_folds=5`, `use_probabilities=True`, `passthrough=False`, `purge_bars=60`, `embargo_bars=1440`, `use_default_configs_for_oof=True`, `analyze_diversity=True`, `reject_low_diversity=False`.
- **[OBSERVED]** Heterogeneity support is *partial and hand-wired*: `fit()` accepts extra `X_train_seq` / `X_val_seq` (`stacking.py:263`) so 3D sequence models can be mixed with 2D tabular models. There is **no** `X_train_4d` parameter ⇒ 4D transformers cannot participate.
- **[OBSERVED]** `src/models/ensemble/validator.py` + `tests/test_ensemble_input_ranks.py` codify this: `test_validator_allows_stacking_2d_plus_3d`, `test_validator_rejects_stacking_with_4d_mixed_in`.

### A1.4 Second-level (clustered) stacking — `second_level.py`
**[OBSERVED]** `SecondLevelStacker` (`second_level.py:124`): correlation-cluster the base models (`_cluster_models`, `:312`), one meta-learner per cluster (`:417`), then a final meta-learner over cluster outputs (`:497`). Save/load implemented.
**[OBSERVED] DEAD:** grep across `src/` and `tests/` finds `SecondLevelStacker` only in `src/models/ensemble/__init__.py:150,229` (the re-export). **Zero call sites.**

### A1.5 Meta-learner selection — `meta_selection.py`
**[OBSERVED]** `select_meta_learner_with_optuna` (`:202`) and `MetaLearnerSelector` (`:326`) with `_select_with_cv` / `_select_with_holdout`. **[OBSERVED] DEAD** — only referenced from `__init__.py:133,224`.

### A1.6 Diversity — `diversity.py`
**[OBSERVED]** Implemented and *live* (called from `EnsembleService._analyze_diversity`, `ensemble_service.py:453`):
- pairwise Pearson correlation of class predictions (`:99`, `:137`)
- Yule's **Q-statistic** (`:172`, `:221`) — needs `y_true`
- **disagreement** (`:251`, `:274`)
- **double-fault** (`:300`, `:332`) — needs `y_true`
- vote entropy (`:362`), KL divergence between prob distributions (`:401`, `:436`)
- composite `compute_diversity_score` (`:464`) — fixed weights 0.20/0.20/0.20/0.15/0.15/0.10, tanh-normalized entropy with a **hardcoded `1.1 ≈ ln 3`** (`:520`) ⇒ **[INFERRED]** mis-normalized in binary mode (max entropy is ln 2 ≈ 0.69).
- `compute_mcc_diversity_matrix` (`:807`), `select_diverse_models` (`:845`) — greedy accuracy-then-diversity forward selection, `filter_correlated_models` (`:943`), `compute_kl_diversity_penalty` (`:966`).

**[OBSERVED] `select_diverse_models` is broken and dead.** At `diversity.py:884` it does `np.asarray(oof.predictions)` — but `OOFPrediction.predictions` is a **pandas DataFrame** (`oof_core.py:65`), not a 1-D label vector. At `:894` it reads `oof.metrics`, an attribute `OOFPrediction` does not have. Only call site is `filter_correlated_models` (also dead). No production path calls either.

### A1.7 Inference-time voting — `src/inference/pipeline.py`
**[OBSERVED]** `InferencePipeline.predict_ensemble(method=...)` (`pipeline.py:268`) → `_combine_predictions` (`:330`) supports `"soft_vote"`, `"hard_vote"`, `"weighted"` (which just calls `_soft_vote` with weights, `:341-344`). Weights are caller-supplied.
**[OBSERVED]** `EnsembleResult.to_dataframe()` (`pipeline.py:100-112`) hardcodes `prob_short/prob_neutral/prob_long` — **[INFERRED]** IndexError in binary (2-class) mode.

---

## A2. OOF generation machinery: 2D vs 3D vs 4D

There are **three distinct OOF producers plus a fourth (walk-forward) that hand-rolls the struct.**

### A2.1 2D tabular — `src/validation/cv/oof_core.py:196 CoreOOFGenerator.generate_tabular_oof`
**[OBSERVED]**
- Allocates full-length `(n, n_classes)` NaN arrays (`:226-229`), loops `PurgedKFold.split(X, y, label_end_times)`.
- Per-fold: `FoldAwareScaler.fit_transform_fold` (train-only fit) → `ModelRegistry.create(model_name, config)` → `model.fit(...)` → `model.predict(X_val_scaled)` → writes `class_probabilities`, `class_predictions`, `confidence`, `fold_idx` at `val_idx` (`:273-276`).
- Output frame columns: `datetime`, `y_true`, `{model}_prob_*` (dynamic via `_get_prob_column_names`, `:27`), `{model}_pred`, `{model}_confidence`, `fold_id` (`:303-313`).
- **[OBSERVED] `datetime` is `X.index if isinstance(X.index, pd.DatetimeIndex) else range(len(X))`** (`:305`).
- Returns `OOFPrediction(..., coverage=...)` with **`original_indices=None`, `n_total_samples=None`** (`:315`).

### A2.2 3D sequence — `src/validation/cv/oof_sequence.py:55 SequenceOOFGenerator.generate_sequence_oof`
**[OBSERVED]**
- Storage arrays are again full-length over the *input row count* (`:103-107`).
- Uses `SequenceCVBuilder` with symbol/gap boundary awareness so no sequence straddles a segment boundary.
- Coverage < 100% is expected and documented (loses `seq_len` rows per segment); `strict_validation` can raise.
- Same `datetime` fallback at `:326`; returns `OOFPrediction(..., original_indices=valid_indices)` (`:336-342`).

### A2.3 4D multi-stream — `src/models/training/services/oof_generation.py:227 _generate_4d_oof`
**[OBSERVED]**
- Splits **by sample index** on the already-windowed 4D tensor (`X_4d[train_idx]`, `:278`) — no re-windowing.
- Per-fold scaling is **hand-rolled median/IQR** on a `reshape(-1, n_features)` view, in-place (`:286-300`) — deliberately *not* `FoldAwareScaler`, so 4D and 2D scaling semantics differ.
- Can reuse cached `request.fold_models` (`:308-312`).
- **[OBSERVED] `oof_data["datetime"] = range(n_samples)`** (`:373`) — a positional counter, *not* a timestamp, and *not* the anchor-bar index.
- Returns `original_indices=valid_indices` (`:390`) — indices into its **own** 0..M-1 sample axis.

### A2.4 Routing — `oof_generation.py:178 _generate_oof_inner`
**[OBSERVED]**
- `data_rank == 4` → `_generate_4d_oof`.
- Otherwise **3D is flattened to 2D**: `_flatten_to_2d` does `X.reshape(X.shape[0], -1)` (`:90-103`), then wraps in a DataFrame with **synthetic column names `f0..fN`** and a fresh `RangeIndex` (`:191-194`). It then calls `OOFGenerator.generate_oof_predictions`, which at `oof_generator.py:255` re-routes on `ModelRegistry.get_model_info(model_name)["requires_sequences"]` and hands the *flattened* frame to `SequenceOOFGenerator`, which **re-windows it**.
  **[INFERRED]** For a 3D model this is a window→flatten→re-window round trip: the flattened row `i` already contains `seq_len × n_features` values, and the sequence builder then stacks `seq_len` of *those* rows. Unless `SequenceCVBuilder` is specifically written to undo the flattening (it is not — it takes an arbitrary feature frame), the 3D OOF path is building sequences-of-sequences. This is a strong candidate for a latent correctness bug; **[UNKNOWN]** whether it is masked because `prepared.X_train` for 3D models is stored pre-window in some paths.
- **[OBSERVED]** Because `X_train_df` is constructed fresh (`:191`), the `DatetimeIndex` is destroyed → `oof_core.py:305` always takes the `range(len(X))` branch in the service path. **All production OOF frames carry positional integers in the `datetime` column, not timestamps.**

### A2.5 Walk-forward — `src/models/training/training_ops.py:576-640`
**[OBSERVED]** Hand-builds the OOF frame from `wf_result.predictions_df`: columns `{model}_pred`, `{model}_confidence`, `y_true`, and probability columns renamed from `{model}_prob_class{i}` → `{model}_prob_{class_name}` (`:610-615`). Constructs `OOFPrediction(..., original_indices=valid_indices, n_total_samples=n_all)`.
**[OBSERVED]** This producer emits **no `fold_id` column and no `datetime` column** — so downstream `EnsembleService._convert_to_oof_results` falls into the `fold_ids = np.zeros(...)` fallback (`ensemble_service.py:261-263`), silently erasing fold provenance for walk-forward ensembles.

### A2.6 Are OOF predictions aligned on a common index?
**[OBSERVED] No. They are aligned on per-model positional integers, and nothing maps those back to a shared bar/time axis.**

Evidence chain:
1. `PreparedData` **does** carry `train_indices: np.ndarray | None` documented as "Original DataFrame indices for training samples" (`src/data/adapters/preparation.py:179`), populated from the adapter (`preparation.py:626: train_indices=train_result.original_indices`) and correctly preserved through `filter_invalid_labels` (`preparation.py:354`).
2. The sequence adapter builds them properly: `sequence.py:237 original_indices = df_indices[label_positions]`; the multi-stream adapter likewise: `multi_stream.py:512 original_indices[:] = anchor_ends - 1`.
3. **`prepared.train_indices` is never read by any OOF producer.** grep for `train_indices` across `src/` returns hits only in `regime_trainer.py:709`, CV splitters, and pipeline stage scripts — **zero hits in `oof_*.py` or `oof_generation.py`.**
4. Consequently `OOFPrediction.original_indices` means *"positions within this model's own prepared array"*, and for a 4D model position 0 is the bar at `seq_len-1` of the source frame while for XGBoost position 0 is bar 0.
5. `OOFAligner.align` (`src/data/adapters/alignment.py:294`) intersects those integers as if they were a shared key (`:349 idx_map = {idx: i for i, idx in enumerate(common_indices)}`).

**[INFERRED] CRITICAL:** an `xgboost + patchtst` ensemble is aligned with a systematic **off-by-`seq_len-1`-bars shift** between the two models' rows, and the meta-learner's labels come from whichever model's `y_true` happened to be first in dict order (`ensemble_service.py:304-311`). This is a silent correctness failure, not a crash — the pipeline reports a healthy `n_common` and a plausible `val_f1`.

---

## A3. The prediction data contract today — there are FIVE overlapping types

**[OBSERVED]**

| Type | File:line | Carries |
|---|---|---|
| `PredictionResult` | `src/core/interfaces.py:125` | `class_predictions (n,)`, `class_probabilities (n,C)`, `confidence (n,)`, `metadata`, optional `indices`, `model_name`, `horizon`, `inference_time_ms`, `is_ensemble`, `individual_predictions`. Validates length agreement in `__post_init__`. |
| `OOFResult` | `src/core/interfaces.py:282` | `predictions`, `probabilities`, `indices`, `fold_ids`, `model_name`, `coverage`; has `align_to(target_indices)` |
| `OOFPrediction` | `src/validation/cv/oof_core.py:48` | `model_name`, `predictions: pd.DataFrame` (**wide, model-name-prefixed columns**), `fold_info: list[dict]`, `coverage`, `original_indices`, `sequence_length`, `n_total_samples`; `get_probabilities()`, `get_class_predictions()`, `alignment_offset`, `n_valid` |
| `OOFPredictionProtocol` | `src/core/interfaces.py:349` | structural mirror of the above, existing purely to break a circular import |
| `AlignedOOFResult` | `src/data/adapters/alignment.py:47` | `probabilities (n_common, n_models*C)` flattened, `predictions (n_common, n_models)` with `-999` sentinel, `common_indices`, `model_names`, `coverage: dict`, `n_common/n_models/n_classes`; `stacking_features` property adds 3 derived columns |

**[OBSERVED]** Plus `ModelPrediction` (`src/inference/batch.py:386`) and two different `EnsembleResult`s (`src/models/ensemble/orchestrator.py:75` and `src/inference/pipeline.py:84`).

**What the canonical prediction does NOT carry today [OBSERVED]:**
- **No timestamp.** `PredictionResult` has no time axis at all. `OOFPrediction` has a `datetime` column that in the production path is a positional range (§A2.4).
- **No horizon on the OOF struct** (it is a dict-key suffix `_h5` in `training_ops.py:151`).
- **No class-label vocabulary.** Column names hardcode `short/neutral/long` for `C==3` (`oof_core.py:34-39`), fall back to `prob_0..prob_{n-1}` otherwise. The mapping index→trading label lives in a separate function `map_classes_to_labels` (`src/models/common/label_mapping.py:90`).
- **No calibration state flag** (is this probability calibrated? by what?).
- **No feature-set / data-rank / scaler provenance.**
- **No coverage mask** distinct from NaN-in-probabilities.
- **Two different missing-value sentinels**: `NaN` in probabilities, `-999` in `AlignedOOFResult.predictions` (`alignment.py:77`), `-99` for invalid labels (`preparation.py:288`), `-1` for "no fold" (`oof_core.py:229`).

---

## A4. How a heterogeneous ensemble (xgboost + patchtst) is actually assembled — full trace

**[OBSERVED]** The live path is **`UnifiedTrainingOrchestrator` → `EnsembleService`**, *not* `EnsembleOrchestrator`.

1. `unified_orchestrator.py:444` — `if self.config.build_ensemble and len(self._model_results) > 1:` → `_build_ensemble(df)` (`:494`).
2. `_build_ensemble` filters `self._oof_predictions` to keys ending `_h{primary_horizon}` (`:511-517`) — mixing horizons is correctly forbidden. Falls back to *all* OOF if the filter empties (`:519-525`).
3. Builds `EnsembleRequest(oof_predictions=filtered, config, df)` → `EnsembleService.build_ensemble` (`ensemble_service.py:75`).
4. Guards: `len < 2` → error result (`:102`); all-zero probs → error (`:116`); NaN probs → warning only (`:123`).
5. `_convert_to_oof_results` (`:216`) — enforces the full-length NaN-padded schema (`:232-238`), slices `probs[indices]`/`preds[indices]` when `original_indices` is set (`:242-248`), pulls `fold_id` if present else zeros (`:255-263`), and `np.where(np.isnan(preds), 0, preds).astype(int)` (`:266`) — **[INFERRED]** NaN predictions silently become class `0`, which in 3-class trading labels means *neutral*, not *missing*.
6. `OOFAligner.align(oof_results, strategy="intersection")` (`:134`) — intersects the positional index sets (§A2.6).
7. `_extract_aligned_labels` (`:280`): **Strategy 1** takes `y_true` from the *first* OOF whose length exceeds `common_indices.max()`; **Strategy 2** falls back to `df["label_h{horizon}"]` indexed positionally by `common_indices`. Both assume common_indices index the *source frame*.
8. `_analyze_diversity` (`:453`) → `DiversityAnalyzer.analyze`.
9. Stacking design matrix = `aligned.stacking_features` (`alignment.py:80`): `[all model probs | mean_confidence | prediction_agreement | prediction_entropy]`, i.e. `n_models*C + 3` columns. Column names from `get_feature_names()` (`alignment.py:143`) — **[OBSERVED]** hardcoded to `LABEL_CLASSES.values()` = `["short","neutral","long"]`, so binary mode produces wrong names.
10. Length-mismatch guard: if `len(y_aligned) != len(stacking_df)` it **truncates both to the minimum** with only a warning (`ensemble_service.py:179-186`) — **[INFERRED]** this converts a real misalignment into a silently-wrong dataset.
11. `_train_meta_learner` (`:346`): drops NaN rows, requires ≥10 rows, **plain 80/20 chronological split with no purge/embargo** (`:391-397`), instantiates one of 4 meta-learners from a hardcoded dict (`:400-405`) — note this dict duplicates `MetaLearnerFactory` (`meta_factory.py:197`) and `EnsembleOrchestrator`'s own copy (`orchestrator.py:275`). Wrapped in a blanket `except Exception` returning `(None, {"error": ...})` (`:449`).

**Serving side [OBSERVED]:** `EnsembleBundle._stack_predictions` (`src/inference/ensemble_bundle.py:763`) — if all base arrays have equal length it calls `_simple_stack` (`:816`), which reassembles columns **in `self.metadata.base_model_names` order** and warns-and-skips any name it can't find (`:842`). If lengths differ it rebuilds `OOFResult`s with `indices = np.arange(n_samples)` (`:784`) — i.e. **re-introduces the positional-origin assumption at inference time**, where a sequence model's row 0 is again not the tabular model's row 0.

---

## A5. Meta-learners

**[OBSERVED]** Four, all `BaseModel` subclasses taking a plain 2-D stacking matrix:
- `RidgeMetaLearner` (`ridge_meta.py:34`) — linear, `softmax` from `meta_base.py:12`.
- `MLPMetaLearner` (`mlp_meta.py:33`).
- `XGBoostMeta` (`xgboost_meta.py:28`), with `_check_cuda_available` (`:335`).
- `CalibratedMetaLearner` (`calibrated_meta.py:35`) — wraps a base estimator with isotonic/Platt; Phase 110 switched it to `TimeSeriesSplit`.

**[OBSERVED]** `meta_factory.py` provides a decorator registry (`register_meta_learner`, `:133`) and `MetaLearnerFactory.create` (`:263`) — but the two live code paths (`ensemble_service.py:400`, `orchestrator.py:275`) each hardcode their own name→class dict instead of using it. Three independent registries for four classes.

**[OBSERVED]** `src/models/ensemble/meta_learners/` is a 26-line re-export shim with no content.

---

## A6. Calibration integration

**[OBSERVED]**
- Base-model calibration: `training_ops.py:_calibrate_model` attaches `result.calibrator` when `config.auto_calibrate`.
- OOF calibration: `CoreOOFGenerator.calibrate_oof_predictions` (`oof_core.py:322`) fits `ProbabilityCalibrator` on the OOF probs themselves, updates the frame in place, and recomputes `confidence` as `calibrated.max(axis=1)`. It **explicitly declines to report before/after improvement** because that would be self-referential (`:368-374`) — good discipline, Phase 110.
- **[OBSERVED]** `calibrate=False` is the default in `generate_oof_predictions` (`oof_generator.py:117`) and the `OOFGenerationService` never passes `calibrate=True` — so **in the production path OOF probabilities feeding the meta-learner are uncalibrated**, and models with different native calibration (XGBoost logloss vs a softmax transformer) enter the stacking matrix on incomparable probability scales.
- **[OBSERVED]** `EnsembleBundle.predict(..., calibrate=True)` takes the argument and **ignores it** — "reserved for future use" (`ensemble_bundle.py:611`).

---

## A7. FAILURE MODES — brutally specific

### F1. A base model fails to train → the entire run dies
**[OBSERVED]** `training_ops.py:_train_standard` (`:36`) loops `for model_name in sequential_models: self._train_model_sequential(...)` with **no try/except** (`:69-70`, `:176`). `_train_boosting_parallel` uses `zip(..., strict=True)` (`:127`) which raises on any length mismatch. There is no per-model quarantine. One broken model = zero ensemble, zero artifacts for the models that already succeeded.
**Contrast [OBSERVED]:** OOF generation *is* defended — `OOFGenerationService.generate_oof` (`oof_generation.py:105`) catches `RuntimeError`, retries after `release_gpu_memory()`, then falls back to CPU by setting `CUDA_VISIBLE_DEVICES=""`, and finally returns `None`. A model that trains but whose OOF fails is **silently dropped from the ensemble while still appearing in `_model_results`** — the ensemble quietly shrinks with only a `logger.warning`.

### F2. Predictions of different lengths
**[OBSERVED]** Three different behaviours in three places:
- `OOFAligner.align` — intersection (default) or union, then the derived-feature math runs over NaNs with `nanmax`/`nanmean` (`alignment.py:99-101`). A model with zero overlap raises `"No common indices found"` (`:337`).
- `ensemble_service.py:179-186` — length mismatch between labels and features → **truncate to min, warn, continue**.
- `EnsembleBundle._stack_predictions` (`ensemble_bundle.py:776`) — `len(set(lengths)) == 1` decides between simple concat and aligner. Equal lengths from misaligned sources are concatenated **without any index check at all**.
**[INFERRED]** The dangerous case is not "different lengths" (that at least routes into the aligner) — it is **equal lengths with different origins**, which is exactly the 2D-vs-4D case after `filter_invalid_labels` and is handled by silent positional concatenation.

### F3. One model emits 2 classes, another 3
**[OBSERVED]** `OOFAligner` is constructed with **no `n_classes` argument** at three of four sites: `ensemble_service.py:73`, `orchestrator.py:197`, `heterogeneous_stacking.py:128` — all default to `N_CLASSES = 3` (`src/core/constants.py:255`). Only `ensemble_bundle.py:808` threads it through.
**[INFERRED]** In binary mode: `aligned_probs` is allocated `(n, n_models*3)` (`alignment.py:342`) and the write `aligned_probs[j, start_col:end_col] = oof.probabilities[i]` (`alignment.py:360`) assigns a length-2 RHS into a length-3 slice → **`ValueError: could not broadcast`**. So `config.n_classes=2` crashes at ensemble alignment despite Phase 114's claim of end-to-end binary support (which was verified through `StackingEnsemble`, `tests/test_d5_binary_mode.py`, not through the `EnsembleService` path).
**[OBSERVED]** A *mixed* 2-vs-3 class ensemble is never detected: `validate_oof_results` (`alignment.py:546`) does check `{r.n_classes for r in oof_results}` and warns on inconsistency — but **it is never called** (grep: only its own definition and `__all__`).
**[OBSERVED]** `AlignedOOFResult.get_feature_names` (`:143`) always emits `short/neutral/long`, and `diversity.py:520` normalizes entropy by `ln 3`.

### F4. Silent index misalignment (the big one)
Covered in §A2.6/§A4. **[INFERRED]** No crash, no warning, plausible metrics. This is the single most damaging failure mode because it is invisible.

### F5. `y_true` provenance is ambiguous
**[OBSERVED]** `_extract_aligned_labels` picks the first OOF whose `y_true` is long enough (`ensemble_service.py:304-311`) — dict-insertion-order dependent, i.e. dependent on `config.models` ordering. Different model orders can produce different labels for the same ensemble.

### F6. Meta-learner validation has no purge/embargo
**[OBSERVED]** `ensemble_service.py:391-397` — bare `n_train = int(n*0.8)`, `iloc[:n_train]` / `iloc[n_train:]`. With triple-barrier labels of horizon `h`, the last `h` training rows overlap the first validation rows. Every other CV surface in this repo purges; this one does not. Reported `val_f1` for the ensemble is therefore optimistically biased relative to the base models' purged OOF metrics — **[INFERRED]** the ensemble will *look* better than its constituents partly for methodological reasons.

### F7. Blanket exception swallowing
**[OBSERVED]** `ensemble_service.py:449` (`_train_meta_learner`), `:531` (`_analyze_diversity`), `oof_generation.py:174` — all `except Exception` → log + return None/error dict. The run completes "successfully" with no ensemble.

### F8. NaN → class 0 coercion
**[OBSERVED]** `ensemble_service.py:266`. Missing predictions become *neutral* votes in the diversity matrix and in `AlignedOOFResult.predictions`.

### F9. `EnsembleOrchestrator._convert_to_oof_results` does not slice probabilities
**[OBSERVED]** `orchestrator.py:560-586` sets `indices = oof_pred.original_indices` but leaves `probs`/`preds` **full-length**. `OOFAligner.align` then does `for i, idx in enumerate(oof.indices): aligned_probs[j] = oof.probabilities[i]` (`alignment.py:355-361`) — pairing the *i-th valid index* with the *i-th row of the full frame*, which for a sequence model with offset `k` maps prediction row `i` to bar `i+k`. That is a hard off-by-`k` bug, and it is precisely the code `EnsembleService._convert_to_oof_results` was later written to fix (`ensemble_service.py:242-248`) — the fix was never back-ported. Two copies, one correct.

### F10. Serving chain is not actually wired
**[OBSERVED]** `EnsembleBundle.from_ensemble_result` (`ensemble_bundle.py:258`) has **zero call sites in `src/`** (grep confirms only docstring mentions in `ensemble_service.py:543,552`). It tries `hasattr(ensemble_result, "_ensemble")` / `"ensemble"` on an `EnsembleResult` **dataclass that has neither field** (`orchestrator.py:92-101`) → `meta_learner = None` + a warning, always, unless `from_orchestrator` overrides it afterwards (`:391`). `to_ensemble_result` (`ensemble_service.py:536`) likewise never attaches the trained meta-learner. `builder.build_ensemble_bundle` (`builder.py:397`) exists but is only called from `builder.py:600` inside `build_all`. **[INFERRED]** The trained ensemble from the live pipeline is saved as metrics JSON + stacking parquet (`orchestrator.py:590 _save_results`) but the deployable ensemble artifact path is effectively orphaned. This matches DECISIONS.md's open item "serving chain".

### F11. Duplicate / dead modules
**[OBSERVED]** `HeterogeneousStackingBuilder` exists twice: `src/models/ensemble/heterogeneous_stacking.py:81` and `src/validation/cv/oof_stacking.py:337`. Neither is called from a live path. `SecondLevelStacker`, `MetaLearnerSelector`, `select_diverse_models`, `filter_correlated_models`, `compute_kl_diversity_penalty`, `build_stacking_features`, `meta_learners/` shim — all reachable only through `__init__` re-exports. **[INFERRED]** roughly 2,300 of the 8,604 ensemble lines are unreachable from `MLFactory`.

---

## A8. Where heterogeneous combination is hard-coded / fragile — ranked

| # | Location | Nature |
|---|---|---|
| 1 | OOF index origin — no producer reads `PreparedData.train_indices` (`oof_core.py:305`, `oof_generation.py:373`, `oof_sequence.py:326`) | **Silent misalignment.** The whole heterogeneous story rests on positional integers that mean different things per rank. |
| 2 | `OOFAligner()` default `n_classes=3` at 3 of 4 sites | Binary mode crashes; mixed-class ensembles undetected |
| 3 | `LABEL_CLASSES` short/neutral/long hardcoded in `alignment.py:151`, `oof_core.py:34`, `pipeline.py:102`, `ensemble_bundle.py` | 3-class assumption baked into names |
| 4 | `stacking_features` fixed at `probs + [confidence, agreement, entropy]` (`alignment.py:80`) | Not extensible; no per-model coverage flag, no regime/vol context, no rank/time features |
| 5 | `VotingEnsemble` rank check (`voting.py:281`) and `validator.py` 4D rejection | Voting/blending structurally cannot be heterogeneous |
| 6 | `StackingEnsemble.fit(X_train_seq=...)` (`stacking.py:263`) | Heterogeneity by adding one extra tensor argument per rank — does not scale to 4D or to a "marketplace" of arbitrary input shapes |
| 7 | Three hardcoded meta-learner name→class dicts | Extension requires editing 3 files |
| 8 | `_simple_stack` column order from `metadata.base_model_names` with warn-and-skip (`ensemble_bundle.py:842`) | Missing model → silently narrower matrix → meta-learner shape error or wrong answer |
| 9 | Meta-learner 80/20 split without purge (`ensemble_service.py:391`) | Optimistic ensemble metrics |
| 10 | 4D scaling hand-rolled (`oof_generation.py:289-297`) vs `FoldAwareScaler` for 2D/3D | Different preprocessing semantics per rank inside one ensemble |

---

# PART B — RESEARCH

## B1. Combination strategies, and what each needs from the prediction contract

| Family | Method | Minimum contract requirement |
|---|---|---|
| Static pooling | simple average, trimmed/median average | aligned prob matrix |
| | weighted average (inverse-error, inverse-variance) | + per-model OOF loss |
| Voting | hard/soft/weighted vote | + class vocabulary, tie policy |
| Rank aggregation | Borda / rank-average of scores | + per-model *scores* (calibration-free — robust to scale mismatch) |
| Learned | linear/ridge stacking, nonlinear stacking (GBDT/MLP), passthrough stacking | + honest OOF matrix + fold ids |
| Greedy | **Caruana ensemble selection with replacement** | + OOF matrix + a scalar metric; the AutoGluon default (ensemble size 100, forward selection with replacement, nonneg weights normalized to 1) |
| Bayesian | BMA; **stacking of predictive distributions** (Yao/Vehtari/Gelman 2018); pseudo-BMA(+) | + per-model *predictive density* on held-out points (log score) |
| Adaptive | Hedge / exponentiated-gradient / multiplicative weights; sliding-window performance weighting | + per-model *sequential* loss stream, i.e. time-ordered predictions with timestamps |
| Conditional | mixture-of-experts with a learned gate; regime-conditioned weights | + gate features per row (volatility, ADX regime, session) aligned to the same index |
| Confidence-aware | abstention/reject option, confidence-weighted pooling | + per-model confidence *and* a validity/coverage flag |
| Calibration-aware | pre-combination calibration vs post-combination calibration | + a `calibrated: bool` + method tag per prediction |
| Selection/pruning | diversity-based pruning (Q, disagreement, double-fault), MCS-based pruning | + per-model prediction vectors + labels |

**Key findings:**

**[OBSERVED-WEB]** *Multi-layer Stack Ensembles for Time Series Forecasting* (Bosch et al., AutoML Conf 2025; arXiv 2511.15350; also Amazon Science) benchmarks **33 forecast-combination methods across 50 datasets** for point *and* probabilistic forecasting. Headline results: (a) ensembling consistently improves accuracy; (b) **learning-based ensembles significantly beat simple aggregation**; (c) **no single stacker wins everywhere**; (d) a **multi-layer stack** (stack the stackers) consistently outperforms any individual ensemble. Practical read-through: the right architecture is a *registry of combiners* selected per-run, not one hardcoded meta-learner.

**[OBSERVED-WEB]** *The cost of ensembling: is it always worth combining?* (arXiv 2506.04677) — combination is not free; ensemble size and pruning matter. **[INFERRED]** For this repo, where each base model can cost hours on a 1.6M-row dataset, greedy selection over an already-trained pool is far cheaper than growing the pool.

**[OBSERVED-WEB]** Caruana ensemble selection (Caruana et al. 2004) as implemented in AutoGluon: greedy **forward selection with replacement** on OOF/validation predictions, default ensemble size 100, weights = selection counts normalized. It is the strongest accuracy/complexity trade-off in AutoML practice and needs *nothing* but an OOF matrix and a metric. **[PROPOSED]** this should be the default combiner in a marketplace design — it degrades gracefully to "pick the best single model" when the ensemble doesn't help.

**[OBSERVED-WEB]** CMA-ES for post-hoc ensembling (arXiv 2307.00286) shows gradient-free weight search over the simplex can beat greedy selection when the metric is non-decomposable — relevant here because the real objective is *Sharpe after costs*, not F1.

**[OBSERVED-WEB]** Yao, Vehtari, Simpson & Gelman (2018), *Using stacking to average Bayesian predictive distributions*: BMA is inconsistent in the **M-open** setting (true DGP not in the candidate set) — which is always the case in finance. They recommend **stacking of predictive distributions** (maximize leave-future-out log score of the *combination*), with bootstrapped pseudo-BMA as a cheap approximation. **[INFERRED]** For this repo the analogue is: optimize combination weights against **out-of-fold log-loss / expected log predictive density**, not against argmax accuracy — the repo already computes `logloss_unweighted`/`logloss_weighted` (Phase 84), so the ingredient exists.

**[OBSERVED-WEB]** Online combination: Hedge / multiplicative weights (a.k.a. normalized exponentiated gradient) is minimax-optimal for the experts problem; recent work (OneNet, arXiv 2309.12659; sector-rotation online ensembles, arXiv 2304.09947) applies it to concept drift in time series, dynamically reweighting heterogeneous models by *recent* loss. **[PROPOSED]** Requires only a time-ordered per-model loss stream — which the repo cannot currently produce because OOF frames have no real timestamps.

**[OBSERVED-WEB]** Regime-conditioned / MoE: RAVEN (arXiv 2606.24062), Regime-Gated Residual MoE (arXiv 2608.12251), and volatility-sensitive MoE (arXiv 2508.02686) all use a **learned softmax gate over regime features**. **[INFERRED]** This repo already has per-symbol ADX regime thresholds (Phase 93) and a regime-aware training mode — a gate is a small addition *if* gate features can be joined to predictions on a shared time index.

**[OBSERVED-WEB]** Diversity: Kuncheva & Whitaker (2003) catalogued 10 measures (Q, correlation, disagreement, double-fault + 6 non-pairwise) and — importantly — found the **relationship between diversity measures and ensemble accuracy is weak and inconsistent on real problems**. **[INFERRED]** The repo's `min_diversity_threshold=0.3` gate and composite diversity score should be treated as **diagnostics, never as an accept/reject gate**; the current code is right to default `reject_low_diversity=False` (`stacking.py:259`).

**[OBSERVED-WEB]** Calibration: "Should Ensemble Members Be Calibrated?" (arXiv 2101.05397) distinguishes **pre-combination** (calibrate members, then combine) from **post-combination** (combine, then calibrate); histogram binning is invalid pre-combination for multiclass because outputs may not be a valid PMF; temperature scaling is the safe pre-combination mapping. Other work (arXiv 2212.00881) shows pre-combination temperature scaling can *hurt* vs plain averaging on some datasets. **[INFERRED]** Calibration mode must be an explicit, recorded, benchmarkable choice — exactly what the repo's `calibrate` flag currently is *not* (§A6).

## B2. The canonical-interface question (how do heterogeneous families share one contract?)

**[OBSERVED-WEB]** The two mature answers in the Python ecosystem:
- **sktime / skpro**: one estimator interface with a *family* of prediction methods — `predict` (point), `predict_var`, `predict_quantiles(alpha)`, `predict_interval(coverage)`, `predict_proba` (full distribution) — with documented **conversion relationships** between them (intervals↔quantiles↔variance↔distribution). Crucially the *return type is indexed* (pandas with a forecasting horizon index), so combination is a join, not a positional concat.
- **AutoGluon-TimeSeries**: models are heterogeneous internally but every model emits predictions on the **same `(item_id, timestamp)` MultiIndex** with the same quantile columns; the weighted ensemble then operates purely on that shared frame.

**[INFERRED] The lesson for this repo is single and blunt: the canonical representation must be *keyed*, not *positional*.** Every method in §B1 is expressible over a keyed prediction table; almost none of them are safely expressible over the repo's current positional arrays. The rank difference (2D/3D/4D) is *irrelevant* once every model publishes `(key → probability vector)` — the windowing loss simply shows up as absent keys.

## B3. Proving an ensemble adds value (temporal data)

**[OBSERVED-WEB]** The accepted ladder:

1. **Diebold–Mariano** — pairwise test of equal predictive accuracy on a loss differential series, with HAC/Newey-West variance for autocorrelated losses (essential at horizon `h>1`; use the Harvey-Leybourne-Newbold small-sample correction). Limitation: **strictly pairwise**; with many models it suffers multiple-comparison and data-snooping bias. Modified DM variants exist for clustered dependence.
2. **Model Confidence Set (Hansen, Lunde & Nason 2011)** — the multi-model generalization. Returns the subset of models whose predictive ability is statistically indistinguishable from the best at level α. **[PROPOSED] The right acceptance criterion for a model marketplace:** "the ensemble is in the 90% MCS and the best single model is not" is a far stronger, honest claim than "ensemble F1 > best single F1". Available in Python via `arch.bootstrap.MCS`.
3. **Block / stationary bootstrap** (moving-block, circular-block, Politis-Romano stationary bootstrap) — required for CIs on *any* path-dependent statistic (Sharpe, max drawdown, turnover) because daily/bar returns are serially dependent. Choose block length ≈ the dependence horizon (here: the barrier horizon).
4. **Deflated Sharpe Ratio (Bailey & López de Prado)** — corrects the observed Sharpe for (a) the number of trials, (b) skew/kurtosis of returns, (c) sample length. **[OBSERVED]** the repo already has this: `compute_deflated_sharpe` with a `num_tests` Bonferroni argument (Phase 96 C1) and `is_sharpe_like_metric()` gating (Phase 87). **[INFERRED]** The `num_tests` for an ensemble must include *the combination search itself* — every candidate weight vector / meta-learner tried is a trial. This is currently not accounted for.
5. **Reality Check / SPA** (White 2000; Hansen 2005) — the "is the best of N strategies better than the benchmark?" test; SPA is the studentized, more powerful version. Complements MCS.
6. **Combinatorially Purged CV (CPCV)** — **[OBSERVED]** already present in the repo (`src/validation/cv/cpcv.py`, wired into HP tuning in Phase 66) and gives ~15 backtest paths, which is exactly the sample you need to bootstrap a Sharpe distribution rather than a point estimate.

**[PROPOSED] Minimum evidence bar for "this ensemble adds value":**
- (a) a **fixed, pre-registered baseline set**: best single model by OOF metric, equal-weight soft vote, and buy-and-hold / always-neutral;
- (b) **DM test with HAC variance** on the ensemble-vs-best-single loss differential, on a *held-out* period the combiner never saw;
- (c) **MCS at α=0.10** over {all base models, all combiners, baselines};
- (d) **stationary-bootstrap CI** on ΔSharpe-after-costs with block length = barrier horizon;
- (e) **DSR** with `num_tests` = models tried × combiners tried × weight configurations tried;
- (f) all of the above computed on the **same aligned key set**, so the comparison is genuinely apples-to-apples.

---

# PART C — [PROPOSED] Canonical prediction contract

## C1. Design principles

1. **Keyed, not positional.** The primary key is a *bar identity*, not an array offset. This alone eliminates failure modes F4, F9 and half of F2.
2. **Full-length with an explicit validity mask.** No compact frames, no sentinels-as-data. NaN means "not predicted"; validity is a first-class boolean array, not inferred from NaN.
3. **Self-describing.** Class vocabulary, calibration state, horizon, data rank, and feature-set fingerprint travel *with* the predictions.
4. **Rank-agnostic.** A 2D, 3D and 4D model emit the identical struct; the rank difference manifests only as missing keys at the head of the series.
5. **One struct, many views.** `PredictionSet` (per model) and `PredictionPanel` (aligned, many models) — nothing else. Delete `OOFResult`, `OOFPrediction`, `AlignedOOFResult`, `OOFPredictionProtocol`, `ModelPrediction`, and the two `EnsembleResult`s in favour of these plus a thin `CombinationResult`.
6. **Every combiner is a plugin over the panel.** `Combiner.fit(panel, y, sample_weight) -> Combiner`; `Combiner.combine(panel) -> PredictionSet`. Voting, greedy Caruana, ridge/GBDT stacking, Hedge, MoE gate — all the same signature.

## C2. The struct

```python
@dataclass(frozen=True)
class PredictionSchema:
    n_classes: int                    # 2 or 3 (or K)
    class_labels: tuple[int, ...]     # e.g. (-1, 0, 1) or (0, 1) — the TRADING labels
    class_names: tuple[str, ...]      # ("short","neutral","long") | ("flat","move")
    horizon: int                      # bars
    task: str = "classification"      # future: "regression" | "quantile"

@dataclass(frozen=True)
class PredictionProvenance:
    model_name: str
    model_family: str                 # boosting | rnn | cnn | transformer | mlp | meta
    data_rank: int                    # 2 | 3 | 4
    sequence_length: int | None
    feature_fingerprint: str          # hash of the selected feature list
    config_fingerprint: str           # hash of resolved hyperparameters
    calibrated: bool
    calibration_method: str | None    # "isotonic" | "sigmoid" | "temperature" | None
    source: str                       # "oof" | "insample" | "test" | "live" | "walkforward"
    cv_fingerprint: str | None        # n_splits/purge/embargo/cv_method — MUST match across a panel

@dataclass(frozen=True)
class PredictionSet:
    # ---- IDENTITY (the fix for F4/F9) ----
    keys: np.ndarray                  # (N,) int64 — canonical bar index into the run's reference frame
    timestamps: np.ndarray            # (N,) datetime64[ns] — REQUIRED, not optional
    # ---- PAYLOAD, all length N, NaN where invalid ----
    probabilities: np.ndarray         # (N, n_classes) float32, rows sum to 1 where valid
    valid: np.ndarray                 # (N,) bool — the ONLY missingness signal
    fold_id: np.ndarray               # (N,) int16, -1 = not assigned
    # ---- OPTIONAL ----
    y_true: np.ndarray | None         # (N,) trading labels, NaN/sentinel-free; carried once per panel
    sample_weight: np.ndarray | None
    schema: PredictionSchema
    provenance: PredictionProvenance

    # derived, never stored:
    #   class_predictions -> class_labels[argmax(probabilities)] where valid
    #   confidence        -> max(probabilities, axis=1)
    #   margin            -> top1 - top2 probability   (better than confidence for abstention)
    #   logloss_per_row   -> needed by Hedge / pseudo-BMA / DM tests
    #   coverage          -> valid.mean()
```

**Invariants enforced in `__post_init__` (fail loud, at the producer):**
- `keys` strictly increasing and unique; `len(timestamps) == len(keys) == N`.
- `probabilities.shape == (N, schema.n_classes)`; rows where `valid` sum to 1 ± 1e-5; rows where `~valid` are all-NaN.
- `valid.sum() > 0`.
- No sentinel integers anywhere. `-99`, `-999`, `-1`-as-missing are abolished except `fold_id == -1`.

```python
@dataclass(frozen=True)
class PredictionPanel:
    keys: np.ndarray                  # (M,) the union or intersection key set
    timestamps: np.ndarray            # (M,)
    y_true: np.ndarray                # (M,) — resolved ONCE, from the reference frame, never from a model
    members: dict[str, PredictionSet] # each already reindexed onto `keys`
    schema: PredictionSchema          # single source of truth; construction REJECTS mismatched member schemas
    join: str                         # "intersection" | "union"

    # views
    def probability_tensor(self) -> np.ndarray: ...   # (M, n_models, n_classes)
    def valid_matrix(self) -> np.ndarray: ...         # (M, n_models) bool
    def loss_matrix(self) -> np.ndarray: ...          # (M, n_models) per-row logloss — powers Hedge, DM, MCS
    def design_matrix(self, extras: Sequence[str] = ()) -> tuple[np.ndarray, list[str]]: ...
        # base columns = flattened probabilities
        # opt-in extras: "confidence", "margin", "agreement", "entropy",
        #                "coverage_flags", "rank_scores", "gate_features"
```

**Why this exact shape makes every §B1 strategy expressible:**
- averaging / weighted averaging / voting → `probability_tensor` + a weight vector.
- rank aggregation → argsort over `probability_tensor` (immune to calibration mismatch).
- stacking (linear/nonlinear/passthrough) → `design_matrix`.
- greedy Caruana / CMA-ES → `probability_tensor` + any metric callable; weights are just a vector.
- Bayesian stacking / pseudo-BMA → `loss_matrix` (log score is exactly what those objectives optimize).
- Hedge / exponentiated gradient → `loss_matrix` **in timestamp order** — which is only possible because `timestamps` is mandatory.
- MoE / regime gate → `design_matrix(extras=["gate_features"])` joined on `keys`.
- confidence-aware / abstention → `margin` + `valid_matrix`.
- diversity & pruning (Q, disagreement, double-fault, MCC) → `argmax(probability_tensor)` + `y_true`.
- DM / MCS / block bootstrap → `loss_matrix` columns are exactly the loss series those tests consume.

## C3. What must change in the repo to honour it (ordered, minimal)

1. **[PROPOSED]** Every adapter already computes `original_indices`; make `PreparedData.train_indices` **non-optional** and have all four OOF producers write `keys = prepared.train_indices[...]` and `timestamps = reference_frame.index[keys]` instead of `range(n)`. This is the single highest-value change in the entire ensemble subsystem — it converts a silent-wrong system into a correct one. (Fixes F4, F9, most of F2.)
2. **[PROPOSED]** Thread `schema` (n_classes + class_labels) from `PipelineConfig` into every producer and into `OOFAligner`; make panel construction **reject** heterogeneous schemas rather than broadcast-crash. (Fixes F3.)
3. **[PROPOSED]** Wrap per-model training in a `ModelOutcome{ok|failed(reason)}` so one failure quarantines one model. (Fixes F1.)
4. **[PROPOSED]** Replace the ad-hoc 80/20 meta split with `PurgedKFold`/CPCV over the panel, using the panel's own `fold_id` to avoid re-using a base model's own fold as meta-validation. (Fixes F6.)
5. **[PROPOSED]** Collapse the three meta-learner dicts into `MetaLearnerFactory`, and re-express `Voting/Blending/Stacking/SecondLevel` as `Combiner` plugins over `PredictionPanel` — at which point `voting.py`'s rank check becomes unnecessary and voting becomes heterogeneous for free.
6. **[PROPOSED]** Make calibration an explicit panel-level mode: `none | pre_combination(temperature|isotonic) | post_combination`, recorded in provenance and benchmarked, per arXiv 2101.05397.
7. **[PROPOSED]** Delete or resurrect: `second_level.py`, `meta_selection.py`, `heterogeneous_stacking.py` (×2), `select_diverse_models`/`filter_correlated_models`, `meta_learners/` shim. As `Combiner` plugins, `SecondLevelStacker` and `MetaLearnerSelector` are genuinely valuable; as orphans they are 1,700 lines of risk.
8. **[PROPOSED]** Add an `EnsembleEvidence` report emitted with every ensemble: baselines, DM (HAC), MCS at α=0.10, stationary-bootstrap ΔSharpe CI, DSR with an honest `num_tests`. Refuse to mark an ensemble `primary_model` in the deploy manifest unless it clears the bar.

## C4. Open questions [UNKNOWN]

- Whether the 3D OOF path (`_flatten_to_2d` → `SequenceOOFGenerator` re-window, §A2.4) is actually double-windowing at runtime, or whether `PreparedData.X_train` for 3D models is stored pre-window in the paths that matter. Needs a runtime probe of `prepared.X_train.shape` vs `prepared.sequence_length` for an LSTM run.
- Whether any current production run has actually built a 2D+4D ensemble whose metrics were trusted — i.e. the blast radius of F4.
- Whether `EnsembleBundle` artifacts produced by `builder.build_all` have ever been round-tripped through `predict_from_raw` with a genuinely heterogeneous member set.

---

## Appendix — key file:line index

| Concern | Location |
|---|---|
| `PredictionResult` (canonical model output) | `src/core/interfaces.py:125` |
| `OOFResult` | `src/core/interfaces.py:282` |
| `OOFPredictionProtocol` | `src/core/interfaces.py:349` |
| `OOFPrediction` | `src/validation/cv/oof_core.py:48` |
| 2D OOF producer | `src/validation/cv/oof_core.py:196` |
| 3D OOF producer | `src/validation/cv/oof_sequence.py:55` |
| 4D OOF producer | `src/models/training/services/oof_generation.py:227` |
| Walk-forward OOF producer | `src/models/training/training_ops.py:576` |
| OOF routing (flatten 3D→2D) | `src/models/training/services/oof_generation.py:90,178` |
| Model-type routing | `src/validation/cv/oof_generator.py:237` |
| `AlignedOOFResult` / `OOFAligner` | `src/data/adapters/alignment.py:47,264` |
| `stacking_features` (the design matrix) | `src/data/adapters/alignment.py:80` |
| Live ensemble entry | `src/models/training/unified_orchestrator.py:494` |
| `EnsembleService` | `src/models/training/services/ensemble_service.py:51` |
| OOF→OOFResult (correct copy) | `src/models/training/services/ensemble_service.py:216` |
| OOF→OOFResult (buggy copy) | `src/models/ensemble/orchestrator.py:542` |
| Label extraction | `src/models/training/services/ensemble_service.py:280` |
| Meta-learner training | `src/models/training/services/ensemble_service.py:346` |
| Diversity analysis | `src/models/ensemble/diversity.py:535,574` |
| Rank compatibility rules | `src/models/ensemble/validator.py:33,227,307` |
| Serving-time stacking | `src/inference/ensemble_bundle.py:763,816` |
| Inference voting | `src/inference/pipeline.py:330,348,379` |
| `PreparedData.train_indices` (unused key source) | `src/data/adapters/preparation.py:179,626` |
| Label mapping | `src/models/common/label_mapping.py:90` |

## Sources (Part B)

- [Multi-layer Stack Ensembles for Time Series Forecasting (AutoML 2025 / arXiv 2511.15350)](https://arxiv.org/pdf/2511.15350) · [PMLR](https://proceedings.mlr.press/v293/bosch25a.html) · [Amazon Science](https://www.amazon.science/publications/multi-layer-stack-ensembles-for-time-series-forecasting)
- [The cost of ensembling: is it always worth combining? (arXiv 2506.04677)](https://arxiv.org/pdf/2506.04677)
- [AutoGluon-TimeSeries — Forecasting Ensembles](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-ensembles.html) · [AutoGluon weighted_ensemble_model source](https://auto.gluon.ai/stable/_modules/autogluon/core/models/ensemble/weighted_ensemble_model.html) · [How AutoGluon computes ensemble weights (discussion #3529)](https://github.com/autogluon/autogluon/discussions/3529)
- [Caruana et al., Ensemble Selection from Libraries of Models](https://www.researchgate.net/publication/221345642_Ensemble_Selection_from_Libraries_of_Models)
- [CMA-ES for Post Hoc Ensembling in AutoML (arXiv 2307.00286)](https://arxiv.org/pdf/2307.00286)
- [Yao, Vehtari, Simpson, Gelman — Using stacking to average Bayesian predictive distributions](https://sites.stat.columbia.edu/gelman/research/published/stacking.pdf) · [Bayesian Analysis](https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-3/Using-Stacking-to-Average-Bayesian-Predictive-Distributions-with-Discussion/10.1214/17-BA1091.full) · [loo package: stacking and pseudo-BMA weights](https://mc-stan.org/loo/articles/loo2-weights.html)
- [OneNet: online ensembling under concept drift (arXiv 2309.12659)](https://arxiv.org/pdf/2309.12659) · [Online Ensemble Learning for Sector Rotation (arXiv 2304.09947)](https://arxiv.org/html/2304.09947) · [The Many Faces of Exponential Weights in Online Learning](http://proceedings.mlr.press/v75/hoeven18a/hoeven18a.pdf)
- [RAVEN: Regime-Aware Variable-context Expert Network (arXiv 2606.24062)](https://arxiv.org/pdf/2606.24062) · [Regime-Gated Residual Mixture-of-Experts (arXiv 2608.12251)](https://arxiv.org/html/2608.12251v1) · [Adaptive Market Intelligence: MoE for Volatility-Sensitive Stock Forecasting (arXiv 2508.02686)](https://arxiv.org/pdf/2508.02686)
- [Kuncheva & Whitaker — Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy](http://machine-learning.martinsewell.com/ensembles/KunchevaWhitaker2003.pdf) · [Springer](https://link.springer.com/article/10.1023/A:1022859003006)
- [Should Ensemble Members Be Calibrated? (arXiv 2101.05397)](https://arxiv.org/pdf/2101.05397) · [Beyond temperature scaling: Dirichlet calibration (NeurIPS 2019)](https://papers.neurips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf) · [Investigating Deep Learning Model Calibration (arXiv 2212.00881)](https://arxiv.org/pdf/2212.00881)
- [Diebold-Mariano test overview](https://www.emergentmind.com/topics/diebold-mariano-test) · [A modified Diebold–Mariano test with clustered dependence](https://www.researchgate.net/publication/353708791_A_modified_Diebold-Mariano_test_for_equal_forecast_accuracy_with_clustered_dependence) · [Forecast evaluation tests and negative long-run variance estimates in small samples](https://www.sciencedirect.com/science/article/abs/pii/S0169207017300559)
- [sktime — Probabilistic Forecasting interface](https://www.sktime.net/en/stable/examples/01b_forecasting_proba.html) · [sktime issue #984: probabilistic prediction interface redesign](https://github.com/sktime/sktime/issues/984) · [skpro — unified probabilistic regression framework](https://github.com/sktime/skpro)
