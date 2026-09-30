# ML Factory

**Config-driven factory for financial time-series models.** Put OHLCV bars in;
get a leakage-free, cost-aware, backtested model or ensemble out, packaged so
that production inference replays training exactly.

```text
Raw OHLCV bars (parquet / csv, any bar size)
   │  sanitize (naive UTC, sorted, de-duplicated) → optional resample (data.bar_timeframe)
   ▼
FeatureEngineer ─────────────── 200+ features: momentum, volatility, volume, microstructure,
   │                            entropy, wavelets, regime; MTF features from closed
   │                            higher-timeframe bars only (shift(1))
   ▼
Triple-barrier labels ───────── per horizon: k_up / k_down × ATR + round-trip cost,
   │                            time barrier max_bars; every label records the bar it resolves at
   ▼
Chronological split ─────────── train │purge+embargo│ val │purge+embargo│ test
   │
   ▼
Per-model feature selection ─── train rows only: clustered MDA, capped by the model contract
   │
   ▼
Any mix of 16 models ────────── 2D tabular  (xgboost, lightgbm, catboost, random_forest, logistic, svm)
   │                            3D sequence (lstm, gru, tcn, transformer, inceptiontime, resnet1d, nbeats, tft)
   │                            4D multi-timeframe streams (patchtst, itransformer)
   ▼
Purged k-fold OOF predictions ─ label-span purging, embargo, uniqueness weights,
   │                            early stopping on a purged train tail
   ▼
Stacking meta-learner ───────── ridge_meta │ xgboost_meta │ mlp_meta │ calibrated_meta │ voting_meta
   │                            aligned on the bar each prediction belongs to
   ▼
Backtest ────────────────────── signal at bar i → fill at bar i+1; stop / take-profit on the
   │                            label's own barriers; commission + slippage per symbol
   ▼
Bundles + deploy manifest ───── model, scaler, calibrator, feature spec, preprocessing graph
   │
   ▼
load_deploy_artifact(...).predict_from_raw(raw_bars)   # same cleaning, features, scaling, routing
```

## Why it exists

Financial ML fails quietly. A model that peeks one bar into the future, a CV
fold whose training labels resolve inside the test period, a backtest that
fills at the signal bar's close, a serving path that computes features slightly
differently from training — each of these produces a beautiful research result
and a strategy that loses money. ML Factory makes the honest path the default
one:

| Guarantee | How |
|---|---|
| No lookahead in features | Higher-timeframe features use completed bars only (`shift(1)` before alignment); entropy, regime and cumulative features are lagged or reset per session |
| No label leakage in CV | Purge derived from the label span, embargo of one trading day, plus purging on every label's actual end bar ([concepts](concepts.md#purge-and-embargo)) |
| Honest out-of-fold predictions | Fold models early-stop on a purged tail of their own training rows, never on the fold they predict |
| Labels and backtest play the same game | One barrier resolution and one cost term feed both the labeler and the backtester |
| No same-bar fills | A signal known at bar *i*'s close is filled at bar *i + 1* ([execution timing](concepts.md#backtest-execution-timing)) |
| Train/serve parity | Bundles replay the recorded bar timeframe, `FeatureEngineer` spec, scaler and calibrator; verified per run by the [mix-and-match harness](MIX_AND_MATCH.md) |
| Overfitting is measured | PSR/DSR for selection bias, CSCV PBO and CPCV paths across model configurations |
| Reproducible | One `ExperimentConfig` (saved with every run), seeded training, checkpoint/resume |

## Where to go next

- **[Getting started](getting-started.md)** — install with `uv`, train on the bundled data from the CLI and from Python, predict from raw bars.
- **[Concepts](concepts.md)** — the methodology and *why* each piece is there: triple-barrier labels, purge/embargo, uniqueness weights, OOF stacking, meta-labeling, regimes, walk-forward, execution timing, costs, DSR/PSR/PBO/CPCV.
- **[Mix and match](mix-and-match.md)** — every model × meta-learner × training mode, and how to add a model.
- **[Configuration](configuration.md)** — every `ExperimentConfig` field with its default (generated from the code).
- **[CLI](cli.md)** — every `ml` command and option (generated from the code).
- **[Deploy and serve](deploy-and-serve.md)** — bundles, the deploy manifest, `predict_from_raw`, warmup.
- **[Examples](examples.md)** — three scripts that run in a few minutes on a laptop CPU.
- **[API reference](reference/index.md)** — `MLFactory`, `ExperimentConfig`, bundles, inference, backtesting.
