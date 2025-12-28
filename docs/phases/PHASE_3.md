# Phase 3: Cross-Validation (OOF Predictions, Stability, PBO)

## Status: COMPLETE

Phase 3 provides time-series-aware evaluation utilities (purged CV, walk-forward, CPCV/PBO) and produces out-of-fold (OOF) predictions for ensembles.

**Primary entrypoints**
- Purged CV + OOF: `python scripts/run_cv.py ...` → outputs under `data/stacking/`
- Walk-forward: `python scripts/run_walk_forward.py ...` → outputs under `data/walk_forward/`
- CPCV + PBO: `python scripts/run_cpcv_pbo.py ...` → outputs under `data/cpcv_pbo/`

---

## What Exists (Source of Truth)

- Purged K-Fold: `src/cross_validation/purged_kfold.py`
- CV runner + tuning: `src/cross_validation/cv_runner.py`
- OOF predictions + stacking dataset: `src/cross_validation/oof_generator.py`
- Walk-forward evaluation: `src/cross_validation/walk_forward.py`
- CPCV + PBO: `src/cross_validation/cpcv.py`, `src/cross_validation/pbo.py`

All scripts use Phase 1’s split parquets via `TimeSeriesDataContainer` (`src/phase1/stages/datasets/container.py`).

---

## Purged CV (OOF predictions)

```bash
# Run CV for specific models/horizons
python scripts/run_cv.py --models xgboost,lightgbm --horizons 5,10,15,20

# Run with Optuna tuning
python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 100

# Disable walk-forward feature selection
python scripts/run_cv.py --models xgboost --horizons 20 --no-feature-selection
```

**Outputs (default)**
- `data/stacking/cv_results.json`
- `data/stacking/tuned_params/`
- `data/stacking/stacking/` (OOF datasets for Phase 4 ensembles)

---

## Walk-Forward Evaluation

```bash
python scripts/run_walk_forward.py --models xgboost --horizons 20
python scripts/run_walk_forward.py --models all --horizons all
```

**Outputs (default)**
- `data/walk_forward/` (per-model/horizon reports + predictions)

---

## CPCV + PBO (Overfitting Risk Gating)

```bash
python scripts/run_cpcv_pbo.py --models xgboost,lightgbm --horizons 20
python scripts/run_cpcv_pbo.py --models all --horizons all --n-groups 8 --n-test-groups 2
```

**Outputs (default)**
- `data/cpcv_pbo/` (CPCV path results + PBO summary + gating output)

---

## Notes / Known Gaps

- Phase 3 evaluates models on Phase 1 features/labels; it does not backtest an executable trading strategy end-to-end (fills/slippage/live execution are out of scope here).
