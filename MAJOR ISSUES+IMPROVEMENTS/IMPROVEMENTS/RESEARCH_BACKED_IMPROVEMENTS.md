# Research‑Backed Improvements for This OHLCV ML Factory

This document consolidates (1) static analysis findings from this repo and (2) current, reputable online guidance for time‑series ML, OHLCV pipelines, leakage prevention, calibration, drift monitoring, and model selection. It is written as a practical backlog: what to change, where to change it, and why.

## Executive Summary (What to do next)

Highest ROI improvements (in order):

1. **Eliminate CV leakage from global scaling** by making scaling fold‑aware inside cross‑validation/OOF generation.
2. **Add probability calibration** (isotonic/Platt) and probability‑quality metrics (Brier/ECE) before stacking/position sizing.
3. **Make purging label‑aware for triple‑barrier events** by passing label end times into `PurgedKFold` (you already support it, but don’t compute/plumb it).
4. **Add walk‑forward / rolling‑origin evaluation** alongside purged k‑fold so selection reflects temporal degradation.
5. **Implement Phase‑5 inference + monitoring** (documented, not implemented) including live drift/performance monitors and retrain triggers.
6. **Quantify backtest overfitting risk** (e.g., CSCV/PBO) as a first‑class report to reduce “winner’s curse” selection.

## Online References (sources used)

- scikit‑learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- scikit‑learn time‑series split notes/leakage context: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- Rolling‑origin / time series cross‑validation (forecasting best practice): https://otexts.com/fpp3/tscv.html
- Probability of Backtest Overfitting (PBO) + CSCV concept (abstract): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- River ADWIN drift detector (online drift monitoring): https://riverml.xyz/latest/api/drift/ADWIN/
- Modern sequence models for time series:
  - TCN: https://arxiv.org/abs/1803.01271
  - Informer: https://arxiv.org/abs/2012.07436
  - PatchTST: https://arxiv.org/abs/2211.14730
  - Temporal Fusion Transformer (TFT): https://arxiv.org/abs/1912.09363

## Repo Static Analysis: Current Design (what exists today)

### Data pipeline (Phase 1)

- Scaled split artifacts exist and are loaded by training/CV:
  - `src/phase1/stages/scaling/*` (scaler logic; “fit on train only” at the dataset‑split level)
  - `src/phase1/stages/scaled_validation/run.py` (offline PSI drift checks on scaled splits)
  - `src/phase1/stages/datasets/container.py` (`TimeSeriesDataContainer` loads `train_scaled.parquet`, etc.)

### Training (Phase 2)

- Unified trainer orchestrates model training and emits classification metrics:
  - `src/models/trainer.py`
- Models produce class predictions + class probabilities (`PredictionOutput`).
- Trading metrics are placeholders; primary evaluation is classification metrics.

### Cross‑validation (Phase 3)

- Purged k‑fold + embargo is implemented:
  - `src/cross_validation/purged_kfold.py`
- Cross‑validation runner generates OOF predictions for stacking:
  - `src/cross_validation/cv_runner.py`
  - `src/cross_validation/oof_generator.py`
  - CLI: `scripts/run_cv.py`
- Walk‑forward feature selection exists:
  - `src/cross_validation/feature_selector.py`

### Ensembles (Phase 4)

- Stacking dataset builder exists (OOF predictions → meta features):
  - `src/cross_validation/oof_generator.py`

### Inference/Production (Phase 5)

- Documented but not implemented:
  - `docs/phases/PHASE_5.md` indicates inference pipeline not implemented.

## Critical Issues (what is currently “wrong” or risky)

### 1) Leakage in CV from “global” scaling

**Observed pattern:** CV/OOF uses `TimeSeriesDataContainer.get_sklearn_arrays("train")`, which loads already‑scaled `train_scaled.parquet` and then splits it into folds (`src/cross_validation/cv_runner.py`, `src/cross_validation/oof_generator.py`).

**Why it matters:** Even if scalers were fit on the global training split only, during CV each fold’s validation portion has been transformed using statistics computed with information from *other folds’ future periods*. This violates time‑series CV assumptions and can inflate metrics.

**Fix:** Make scaling fold‑aware during CV (see “Implementation Backlog” below).

Relevant files:
- `src/cross_validation/cv_runner.py`
- `src/cross_validation/oof_generator.py`
- (reuse scaler logic) `src/phase1/stages/scaling/scaler.py`

### 2) Probabilities are used “raw” without calibration

**Observed pattern:** Models return class probabilities, which are then used for OOF, stacking features (entropy/confidence), and potentially future sizing decisions, without probability calibration.

**Why it matters:** Many classifiers—especially boosting trees—are poorly calibrated out‑of‑the‑box. If you use probabilities downstream (thresholding, ranking, sizing, ensembles), calibration is a direct risk/return lever.

**Fix:** Add calibration on a held‑out validation slice (or via CV) and report Brier score + ECE.

Relevant files:
- `src/models/trainer.py`
- `src/cross_validation/oof_generator.py`

### 3) Purging is not label‑aware for event labels

**Observed pattern:** `PurgedKFold.split(...)` supports `label_end_times` but the repo does not compute or pass this from triple‑barrier labeling into CV.

**Why it matters:** Triple‑barrier labels are resolved at different horizons (event end times can vary). Fixed “purge_bars” is a blunt approximation; label end times enable correct overlap‑aware purging.

**Fix:** Persist label end times (e.g., `label_end_time_h{h}`) during labeling and pass them into `PurgedKFold` during CV.

Relevant files:
- `src/phase1/stages/labeling/triple_barrier.py`
- `src/cross_validation/purged_kfold.py`
- `src/cross_validation/cv_runner.py` / `src/cross_validation/oof_generator.py`

### 4) Evaluation is mostly classification‑centric; trading evaluation is placeholder

**Observed pattern:** `Trainer` adds “basic stats only” trading metrics.

**Why it matters:** For OHLCV trading, classification metrics alone can select models that are untradeable (high turnover, poor cost‑adjusted returns, unstable regimes).

**Fix:** Add a minimal but honest backtest metric set for validation/test (PnL, turnover, max DD, cost model), and report it consistently for single models and ensembles.

Relevant files:
- `src/models/trainer.py`
- (future) Phase‑5 runtime evaluator

### 5) Inference + monitoring gap is real (documented as missing)

**Observed pattern:** Phase‑5 doc exists but code is absent.

**Why it matters:** Without a serialized inference bundle and monitoring, “production” behavior diverges from research and models silently decay.

**Fix:** Implement inference pipeline and connect to drift monitors + retrain triggers.

Relevant files:
- `docs/phases/PHASE_5.md`
- (new) `src/inference/*` (proposed)

### 6) Reproducibility issues in scripts (hard‑coded absolute paths)

**Observed pattern:** Some scripts reference `/Users/...` paths.

**Fix:** Convert to config/CLI args, and ensure scripts work relative to repo root.

Relevant files:
- `scripts/diagnose_label_distribution.py`
- `scripts/verify_pipeline_final.py`

## Implementation Backlog (concrete changes)

### A) Make CV fold‑aware (no leakage)

Goal: each fold’s scaler is fit only on that fold’s training indices and applied to that fold’s validation indices.

Practical approaches:

1. **Fold‑local scaling inside OOF generation**
   - In `src/cross_validation/oof_generator.py`, before training each fold:
     - Fit a scaler on `X_train` only.
     - Transform both `X_train` and `X_val`.
   - Reuse the repo’s own scaling logic if possible (or keep it minimal if you don’t want to import Phase‑1 stage classes into CV).

2. **Use an sklearn `Pipeline` in CV**
   - Wrap scaler + model so every fold refits properly.
   - This is clean for sklearn‑like models, but your models are custom; still possible by inserting scaling before calling `.fit()`.

Deliverables:
- CV/OOF results that can be trusted as out‑of‑sample within the CV design.
- A report note indicating fold scaling is enabled.

### B) Add probability calibration + probability quality metrics

Goal: all downstream consumers (ensembles, sizing, thresholding) use calibrated probabilities.

Implementation options:

1. **Post‑fit calibration in Trainer**
   - After fitting the base model, fit `CalibratedClassifierCV(cv="prefit")` on the validation set.
   - Persist the calibrator alongside the model artifact.
   - Report: Brier score, log loss, ECE, and reliability curve bins.

2. **Calibrated OOF**
   - Fit calibrator on each fold’s validation (or a sub‑split of training) and apply consistently.
   - For stacking, store calibrated probabilities in OOF predictions.

Deliverables:
- `metrics/calibration.json` per run (or equivalent).
- Stacking features computed from calibrated probabilities.

### C) Add label end times and wire into `PurgedKFold`

Goal: purge overlaps based on event completion time rather than only fixed bars.

Concrete steps:

1. In triple barrier labeling (`src/phase1/stages/labeling/triple_barrier.py`), you already compute `bars_to_hit`.
2. Create a per‑row end timestamp:
   - `label_end_time = datetime + bars_to_hit * bar_interval`
   - Store per horizon, e.g., `label_end_time_h20`.
3. In `TimeSeriesDataContainer`, expose a way to retrieve this series (or keep it in the dataframe and access it during CV).
4. In CV, pass `label_end_times` into `PurgedKFold.split(...)`.

Deliverables:
- CV that correctly purges event overlap for triple‑barrier labels.

### D) Add walk‑forward / rolling‑origin evaluation

Goal: replicate “real usage” evaluation where training window rolls forward and you score the next chunk (optionally purged/embargoed).

Concrete steps:
- Add a new evaluation mode (separate from k‑fold) in `src/cross_validation/*`:
  - expanding window or rolling window
  - fixed forecast horizon and evaluation step
  - report per‑window degradation (stability over time)

Deliverables:
- A report table by window (accuracy/F1 + trading metrics).

### E) Implement inference pipeline + monitoring (Phase 5)

Goal: one serialized artifact bundle and a runtime that:
- loads scaler + feature list + model/ensemble
- runs feature engineering consistently (or asserts precomputed features)
- produces predictions + calibrated probabilities
- logs and monitors drift + performance

Concrete components:
- `InferencePipeline` object (code, not just docs)
- Drift monitors:
  - Feature drift (PSI or similar) already exists offline; reuse.
  - Online performance drift via ADWIN‑style detector on prediction correctness.
- Retrain triggers:
  - drift threshold, performance degradation threshold, or schedule.

Deliverables:
- `src/inference/` module (proposed)
- CLI entrypoint for batch inference + monitoring

### F) Add PBO/CSCV backtest‑overfitting reporting

Goal: avoid selecting models/hyperparams that “won by chance” across many trials.

Concrete steps:
- Add a CSCV/PBO evaluator module and run it after hyperparameter tuning / model selection.
- Store PBO estimates and warn on high PBO.

Deliverables:
- `reports/pbo_*.json` (or similar) and summary in `cv_results.json`.

### G) Sequence‑model CV correctness

Goal: when evaluating LSTM/GRU/Transformer/TCN, CV must operate on sequences without leaking across boundaries.

Concrete steps:
- Implement a sequence‑aware fold builder or restrict CV runner to non‑sequence models until it exists.
- Ensure sequence construction never crosses symbol boundaries (already supported in `TimeSeriesDataContainer.get_pytorch_sequences(..., symbol_isolated=True)`).

Deliverables:
- CV runner that supports both tabular and sequence models correctly.

## Quality Gates (what “done” looks like)

- CV metrics drop (likely) but become more trustworthy after fold‑aware scaling and label‑aware purging.
- Probability calibration improves Brier/ECE; reliability curves show reduced overconfidence.
- Walk‑forward evaluation shows stability profile across time (not just fold averages).
- Inference artifacts include everything needed for production parity (features, scaler, model, calibration, metadata).
- Drift monitors produce actionable alerts and retrain triggers (not just offline diagnostics).

## Notes / Repo Hygiene

- Fix scripts with absolute paths to be config/CLI driven:
  - `scripts/diagnose_label_distribution.py`
  - `scripts/verify_pipeline_final.py`
- Consider consolidating dependency and packaging metadata separately; not covered here beyond pipeline correctness.

