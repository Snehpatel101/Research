# Mix-and-Match Verification Matrix

Generated 2026-09-29 by `python scripts/mix_match.py report` from the
results of `python scripts/mix_match.py {solo,pairs,meta,modes,modes-solo,binary,all-in}`.

**Overall: 195/202 runs pass.**

Every run is a full `MLFactory.run()` on synthetic 5-minute OHLCV (4,000 bars,
1 epoch, CPU): features -> labels -> per-model feature selection -> training ->
OOF -> stacking ensemble -> backtest -> bundles -> deploy artifact -> reload ->
`predict_from_raw`. A run passes only if **all** of these hold:

- training succeeds and every model reports metrics (plus ensemble metrics for >1 model)
- stacking rows pair every model's OOF prediction with the label of the *same* bar
- the deployed bundle reproduces the trained model's validation probabilities
  (re-prepared validation split vs `predict_from_raw` on raw bars)
- features recomputed from raw OHLCV equal the training features
- the backtest produces metrics and the deploy artifact reloads and predicts

## Building blocks

- **Base models (16):** `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic`, `svm`, `lstm`, `gru`, `tcn`, `transformer`, `inceptiontime`, `resnet1d`, `nbeats`, `patchtst`, `itransformer`, `tft`
- **Meta-learners (5):** `ridge_meta`, `xgboost_meta`, `mlp_meta`, `calibrated_meta`, `voting_meta`
- **Training modes (4):** `standard`, `walk_forward`, `regime_aware`, `meta_labeling`

## Every base model alone — 16/16 pass

| Run | Result | Seconds |
|---|---|---|
| `solo_xgboost` | PASS | 52.1 |
| `solo_lightgbm` | PASS | 53.4 |
| `solo_catboost` | PASS | 54.2 |
| `solo_random_forest` | PASS | 51.2 |
| `solo_logistic` | PASS | 48.4 |
| `solo_svm` | PASS | 54.5 |
| `solo_lstm` | PASS | 113.6 |
| `solo_gru` | PASS | 108.3 |
| `solo_tcn` | PASS | 96.4 |
| `solo_transformer` | PASS | 457.1 |
| `solo_inceptiontime` | PASS | 284.4 |
| `solo_resnet1d` | PASS | 111.9 |
| `solo_nbeats` | PASS | 51.6 |
| `solo_patchtst` | PASS | 46.6 |
| `solo_itransformer` | PASS | 51.0 |
| `solo_tft` | PASS | 1061.4 |

## Every pair of base models (stacking ensemble) — 117/120 pass

| | xgboost | lightgbm | catboost | random_forest | logistic | svm | lstm | gru | tcn | transformer | inceptiontime | resnet1d | nbeats | patchtst | itransformer | tft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **xgboost** | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **lightgbm** | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **catboost** | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **random_forest** | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **logistic** | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **svm** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **lstm** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **gru** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **tcn** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **transformer** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **inceptiontime** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| **resnet1d** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ❌ |
| **nbeats** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ❌ |
| **patchtst** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| **itransformer** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ❌ |
| **tft** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | — |

## Every meta-learner on a cross-family ensemble (xgboost + lstm + patchtst) — 5/5 pass

| Run | Result | Seconds |
|---|---|---|
| `meta_ridge_meta` | PASS | 395.4 |
| `meta_xgboost_meta` | PASS | 399.1 |
| `meta_mlp_meta` | PASS | 395.3 |
| `meta_calibrated_meta` | PASS | 180.4 |
| `meta_voting_meta` | PASS | 177.9 |

## Every training mode on a cross-family ensemble (xgboost + lstm + patchtst) — 4/4 pass

| Run | Result | Seconds |
|---|---|---|
| `mode_standard` | PASS | 142.9 |
| `mode_walk_forward` | PASS | 156.6 |
| `mode_regime_aware` | PASS | 163.5 |
| `mode_meta_labeling` | PASS | 52.1 |

## Every base model in every non-standard training mode — 44/48 pass

| Run | Result | Seconds |
|---|---|---|
| `mode_walk_forward_xgboost` | PASS | 79.8 |
| `mode_walk_forward_lightgbm` | PASS | 77.6 |
| `mode_walk_forward_catboost` | PASS | 88.9 |
| `mode_walk_forward_random_forest` | PASS | 70.6 |
| `mode_walk_forward_logistic` | PASS | 68.0 |
| `mode_walk_forward_svm` | PASS | 82.2 |
| `mode_walk_forward_lstm` | PASS | 201.2 |
| `mode_walk_forward_gru` | PASS | 192.3 |
| `mode_walk_forward_tcn` | PASS | 177.5 |
| `mode_walk_forward_transformer` | PASS | 714.6 |
| `mode_walk_forward_inceptiontime` | PASS | 457.3 |
| `mode_walk_forward_resnet1d` | PASS | 163.0 |
| `mode_walk_forward_nbeats` | PASS | 97.5 |
| `mode_walk_forward_patchtst` | PASS | 70.8 |
| `mode_walk_forward_itransformer` | PASS | 82.3 |
| `mode_walk_forward_tft` | CRASH (no result) | 1926.9 |
| `mode_regime_aware_xgboost` | PASS | 68.4 |
| `mode_regime_aware_lightgbm` | PASS | 75.7 |
| `mode_regime_aware_catboost` | PASS | 97.2 |
| `mode_regime_aware_random_forest` | PASS | 69.0 |
| `mode_regime_aware_logistic` | PASS | 65.2 |
| `mode_regime_aware_svm` | PASS | 66.4 |
| `mode_regime_aware_lstm` | PASS | 125.6 |
| `mode_regime_aware_gru` | PASS | 112.2 |
| `mode_regime_aware_tcn` | PASS | 101.5 |
| `mode_regime_aware_transformer` | PASS | 599.0 |
| `mode_regime_aware_inceptiontime` | PASS | 367.0 |
| `mode_regime_aware_resnet1d` | PASS | 181.5 |
| `mode_regime_aware_nbeats` | PASS | 83.4 |
| `mode_regime_aware_patchtst` | PASS | 79.1 |
| `mode_regime_aware_itransformer` | PASS | 86.0 |
| `mode_regime_aware_tft` | PASS | 1554.3 |
| `mode_meta_labeling_xgboost` | PASS | 66.9 |
| `mode_meta_labeling_lightgbm` | PASS | 69.5 |
| `mode_meta_labeling_catboost` | PASS | 75.6 |
| `mode_meta_labeling_random_forest` | PASS | 71.1 |
| `mode_meta_labeling_logistic` | PASS | 68.9 |
| `mode_meta_labeling_svm` | PASS | 74.5 |
| `mode_meta_labeling_lstm` | PASS | 185.1 |
| `mode_meta_labeling_gru` | PASS | 153.4 |
| `mode_meta_labeling_tcn` | PASS | 137.8 |
| `mode_meta_labeling_transformer` | PASS | 575.2 |
| `mode_meta_labeling_inceptiontime` | PASS | 363.7 |
| `mode_meta_labeling_resnet1d` | PASS | 143.4 |
| `mode_meta_labeling_nbeats` | PASS | 75.6 |
| `mode_meta_labeling_patchtst` | ValueError: Model 'patchtst' requires multi-stream adapter (4D data). Either provide additional_dfs or set symbol and sp | 68.9 |
| `mode_meta_labeling_itransformer` | ValueError: Model 'itransformer' requires multi-stream adapter (4D data). Either provide additional_dfs or set symbol an | 71.1 |
| `mode_meta_labeling_tft` | ValueError: Meta-labeling needs primary bets that both win and lose: the primary took 0 sided OOF bets with win rate nan | 1005.6 |

## Binary labels: every meta-learner and every non-standard mode — 8/8 pass

| Run | Result | Seconds |
|---|---|---|
| `binary_ridge_meta` | PASS | 225.7 |
| `binary_xgboost_meta` | PASS | 223.8 |
| `binary_mlp_meta` | PASS | 226.4 |
| `binary_calibrated_meta` | PASS | 358.3 |
| `binary_voting_meta` | PASS | 353.5 |
| `binary_walk_forward` | PASS | 418.3 |
| `binary_regime_aware` | PASS | 222.1 |
| `binary_meta_labeling` | PASS | 90.7 |

## All base models in one ensemble — 1/1 pass

| Run | Result | Seconds |
|---|---|---|
| `all_in` | PASS | 2953.5 |
