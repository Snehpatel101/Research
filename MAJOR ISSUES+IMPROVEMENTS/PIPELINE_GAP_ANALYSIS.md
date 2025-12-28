# Pipeline Gap Analysis (Root Improvements vs. Current Code vs. Online Guidance)

This document converts the recommendations in `RESEARCH_BACKED_IMPROVEMENTS.md` into a concrete crosswalk:

- **What the root improvements propose**
- **What this repo’s pipeline actually does today (static analysis)**
- **What online best-practice guidance implies**
- **Where to change code**

## Sources (online guidance referenced)

- Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- Time-series split / leakage context: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- Rolling-origin time-series CV: https://otexts.com/fpp3/tscv.html
- Probability of Backtest Overfitting (PBO) / CSCV (abstract): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Online drift detection (ADWIN): https://riverml.xyz/latest/api/drift/ADWIN/
- Sequence models (context for model choices):
  - TCN: https://arxiv.org/abs/1803.01271
  - Informer: https://arxiv.org/abs/2012.07436
  - PatchTST: https://arxiv.org/abs/2211.14730
  - TFT: https://arxiv.org/abs/1912.09363

## Crosswalk Table (by root improvement item)

| Root improvement item | What the pipeline does today | Why it conflicts / aligns with guidance | Concrete gap + where to change |
|---|---|---|---|
| 1) Fold-aware scaling (remove CV leakage) | Phase-1 scaling fits once on the global **train split** and writes `data/splits/scaled/train_scaled.parquet` etc. (`src/phase1/stages/scaling/run.py`). Cross-validation loads these **already-scaled** splits and then creates folds within them (`src/cross_validation/cv_runner.py`, `src/cross_validation/oof_generator.py`). | In time-series CV, preprocessing must be fit on the training data for that fold; otherwise fold validation sees transforms informed by “future” samples from other folds (`scikit-learn` time-series split guidance). | **Gap:** CV/OOF metrics are optimistic because fold validation data is scaled using statistics computed on data outside the fold’s training window. **Fix:** move scaling into each fold inside `src/cross_validation/oof_generator.py` (fit scaler on `X_train`, transform `X_train` + `X_val`), optionally conditional on `model.requires_scaling`. |
| 2) Probability calibration + Brier/ECE | No `CalibratedClassifierCV`/Brier/ECE/reliability code exists in `src` (static search). OOF stacking features (entropy/confidence) are derived from raw probabilities (`src/cross_validation/oof_generator.py`). | If probabilities are consumed downstream (stacking, thresholds, sizing), calibration is a standard step; scikit-learn documents isotonic/sigmoid calibration and how to evaluate calibration quality. | **Gap:** raw probabilities (esp. boosting) can be systematically miscalibrated → wrong thresholds/weights/meta-features. **Fix:** add calibration (post-fit on validation, or per-fold) in `src/models/trainer.py` and/or `src/cross_validation/oof_generator.py`. Add Brier/ECE reporting alongside existing metrics. |
| 3) Label-aware purging via label end times | `PurgedKFold.split(...)` supports `label_end_times` but nothing passes it (`src/cross_validation/purged_kfold.py`). Triple-barrier labeling computes `bars_to_hit` but doesn’t persist event end timestamps (`src/phase1/stages/labeling/triple_barrier.py`). Also, CV features `X` typically do not have a DatetimeIndex (container keeps `datetime` as a column), so label-aware purging wouldn’t trigger even if passed (PurgedKFold checks index type). | Event-based labels are exactly where overlap-aware purging matters; fixed “purge_bars” is a coarse approximation. Your CV already anticipates the correct approach via `label_end_times`. | **Gap:** label-aware purge path is effectively dead. **Fix:** persist `label_end_time_h{h}` during labeling, ensure CV uses a DatetimeIndex (or modify `PurgedKFold` to accept timestamps separately), and pass `label_end_times` from the container into `PurgedKFold.split` in `src/cross_validation/cv_runner.py` / `src/cross_validation/oof_generator.py`. |
| 4) Walk-forward / rolling-origin evaluation | Repo contains walk-forward **feature selection** (`src/cross_validation/feature_selector.py`), but not walk-forward **evaluation**. Model evaluation is based on purged k-fold averages on the training split. | Rolling-origin evaluation is a standard way to measure temporal degradation and “train-then-predict-next-block” behavior (`FPP3`). | **Gap:** fold averages can hide regime/recency effects; selection may not reflect live behavior. **Fix:** add a rolling-origin evaluator alongside `CrossValidationRunner` (expanding/rolling window, purged/embargoed), and report metrics per window + degradation over time. |
| 5) Phase-5 inference + monitoring | Phase 5 is documented but not implemented: `docs/phases/PHASE_5.md` describes `InferencePipeline`, serialization, and monitoring, but there’s no corresponding `src/inference/` runtime module. Offline drift checks exist post-scale via PSI (`src/phase1/stages/scaled_validation/run.py`). | Production best practice is training/inference parity plus continuous monitoring; offline-only drift checks don’t address post-deploy decay. ADWIN is a canonical online drift detector for streaming accuracy shifts. | **Gap:** no executable inference bundle; no online monitors/retrain triggers. **Fix:** implement `InferencePipeline` as code; serialize scaler+feature list+model(s)+calibrator(s); add monitors (feature drift PSI + online performance drift ADWIN) and retrain triggers. |
| 6) Backtest overfitting risk (PBO/CSCV) | No PBO/CSCV code exists (static search). Optuna tuning exists in `src/cross_validation/cv_runner.py`, but the risk of over-selection across models/horizons/trials isn’t quantified. | Bailey et al. show standard selection can be misleading in backtests; CSCV/PBO estimates probability the selection is overfit. | **Gap:** you can “optimize to noise” with no guardrail. **Fix:** add a PBO report stage after tuning/selection and warn/block when PBO is high; integrate into `scripts/run_cv.py` and `src/cross_validation/cv_runner.py` reporting outputs. |
| (Bonus) Sequence-model CV correctness | CV runner uses `container.get_sklearn_arrays("train")` for everything (`src/cross_validation/cv_runner.py`). OOF training uses `X_train.values` (2D). Sequence models in this repo expect sequences created by `TimeSeriesDataContainer.get_pytorch_sequences` (`src/phase1/stages/datasets/container.py`). | If CV doesn’t match the data contract of sequence models, results are invalid regardless of model sophistication. | **Gap:** neural/transformer/TCN CV is likely structurally wrong (or silently not what you think). **Fix:** implement sequence-aware CV path (or restrict CV to tabular models until sequence CV exists). Ensure sequences never cross symbol boundaries (already supported). |

## Two “pipeline reality” issues to fix alongside the above

1. **Datetime/index mismatch blocks label-aware purge and weakens diagnostics**
   - OOF currently sets `datetime` to either `X.index` or `range(len(X))` (`src/cross_validation/oof_generator.py`), but `X` commonly has a non-datetime index because `datetime` is a column in the dataset artifacts.
   - Fix by consistently using the `datetime` column (and/or setting it as the index) in CV/OOF code paths.

2. **Label naming/reporting inconsistency**
   - Data contract labels are `{-1, 0, 1}` (`src/phase1/stages/validation/data_contract.py`).
   - Some metric reporting uses `{0: short, 1: neutral, 2: long}` labels (`src/models/trainer.py`) even though predictions are typically `{-1, 0, 1}`.
   - Fix by standardizing label encoding/decoding and class naming in metrics output.

## Suggested execution order (fastest path to more truthful results)

1. Fold-aware scaling in OOF/CV.
2. Datetime indexing + `label_end_times` plumbing for label-aware purge.
3. Probability calibration + Brier/ECE reporting (and stack on calibrated probs).
4. Rolling-origin evaluation runner + temporal degradation reporting.
5. Phase-5 inference pipeline implementation + online monitoring hooks.
6. PBO/CSCV risk report for selection guardrails.

