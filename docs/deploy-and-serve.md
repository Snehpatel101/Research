# Deploy and serve

Every `MLFactory.run()` (and `ml run`, unless `--no-deploy`) ends by packaging
what it trained into **bundles** and indexing them in a **deploy manifest**.
Serving is one call: load the artifact for a horizon, pass raw OHLCV bars.

```python
from src.inference import load_deploy_artifact

artifact = load_deploy_artifact("experiments/runs/<run_id>/deploy", horizon=5)
pred = artifact.predict_from_raw(raw_ohlcv_df)

pred.class_predictions        # (n,)   -1 short, 0 neutral, +1 long  (binary: 0 / 1)
pred.class_probabilities      # (n, 3) columns in class order: short, neutral, long
pred.confidence               # (n,)   max probability
pred.metadata["timestamps"]   # DatetimeIndex: the bar each row belongs to
```

## Why bundles replay training

The most common silent failure in deployed trading models is **train/serve
skew**: inference computes features with slightly different code, a different
resampling rule, another scaler fit, or no calibrator — and the model sees
inputs it was never trained on. Nothing crashes; it just stops working.

A bundle therefore stores the *recipe*, not a re-implementation:

1. **Bar timeframe** — raw bars finer than the training bars are resampled to
   it (1-minute in, 5-minute model: fine); coarser bars are rejected.
2. **Cleaning** — the same `sanitize_bars` step as training (naive UTC,
   sorted, de-duplicated).
3. **Features** — the `FeatureEngineer` spec captured at training time is
   rebuilt with `FeatureEngineer.from_spec` and runs *the same code* training
   ran. There is no second feature implementation to drift.
4. **Columns** — exactly the model's selected features, in training order.
5. **Scaling** — the scaler fitted on the training split, applied once.
6. **Model input** — the model's rank: the 2D table, 3D windows of
   `seq_len` bars, or 4D multi-timeframe OHLCV streams.
7. **Calibration** — the calibrator fitted on the validation split
   (`calibrate=False` returns the raw model probabilities).

The mix-and-match harness verifies this for every model and combination: the
deployed bundle must reproduce the trained model's validation probabilities
from raw bars, and features recomputed from raw OHLCV must equal the training
features ([verification matrix](MIX_AND_MATCH.md)).

## What a run writes

```text
<output_dir>/<run_id>/
├── experiment_config.yaml
├── bundles/
│   ├── xgboost_h5/                 ModelBundle
│   │   ├── model/                  fitted model (native format per family)
│   │   ├── scaler.pkl              training-split scaler
│   │   ├── calibrator.pkl          validation-split calibrator
│   │   ├── features.json           selected feature columns, in order
│   │   ├── feature_spec.json       FeatureEngineer spec
│   │   ├── preprocessing_graph.json  bar timeframe + spec + columns
│   │   ├── metadata.json           model, horizon, input rank, sequence length, hashes, metrics
│   │   └── manifest.json           file list + checksums, bundle version
│   ├── lstm_h5/  patchtst_h5/      one ModelBundle per model and horizon
│   └── ensemble/                   EnsembleBundle
│       ├── meta_learner/           fitted meta-learner
│       ├── base_bundles.json       which ModelBundles it stacks
│       ├── stacking_features.json  stacking column layout
│       └── alignment_config.json   how base predictions are aligned
└── deploy/
    └── manifest.json               DeployManifest
```

Training modes add their own bundle kinds:

| Kind | Written by | Layout | `predict_from_raw` |
|---|---|---|---|
| `ModelBundle` | every mode | as above | one model's predictions |
| `EnsembleBundle` | `build_ensemble=True` with 2+ models | meta-learner + references to base bundles | runs every base bundle, aligns on timestamps, applies the meta-learner |
| `RegimeBundle` | `regime_aware` | `regime_bundle_metadata.json`, `regimes/<regime>/` (one ModelBundle each) | detects each bar's regime with the training detector config and routes it; `metadata["regimes"]` |
| `MetaLabelingBundle` | `meta_labeling` | `meta_labeling_metadata.json`, `primary_bundle/`, `meta_model.pkl` | primary predictions set to neutral where the meta-model rejects the bet |

## The deploy manifest

`deploy/manifest.json` is pure JSON (no pickles). Per horizon it lists every
bundle with its validation metrics and names a **primary** model: the ensemble
when one was built, otherwise the model with the best validation macro-F1.

```json
{
  "version": "1.0.0",
  "symbol": "MES",
  "horizons": {
    "5": {
      "horizon": 5,
      "primary_model": "voting_meta",
      "entries": [
        {"model_name": "voting_meta", "bundle_path": "bundles/ensemble", "is_ensemble": true,  "metrics": {"...": 0}},
        {"model_name": "lstm",        "bundle_path": "bundles/lstm_h5",  "is_ensemble": false, "metrics": {"...": 0}},
        {"model_name": "xgboost",     "bundle_path": "bundles/xgboost_h5", "is_ensemble": false, "metrics": {"...": 0}}
      ]
    }
  }
}
```

Bundle paths are relative, so the whole run directory can be copied or
archived and served from anywhere.

## Loading

| Function | Use |
|---|---|
| `load_deploy_artifact(deploy_dir, horizon, model_name=None)` | The primary artifact for a horizon, or a named one (`model_name="xgboost"`). Returns whatever kind the manifest points at. |
| `select_deploy_artifact(deploy_dir, horizon, model_name=None)` | The bundle *path* that would be loaded. |
| `validate_deploy_artifact(deploy_dir)` | `{"valid": bool, "issues": [...]}` — manifest parses and every bundle exists. Run it in CI / before promoting a run. |
| `load_bundle(path)` | Any bundle directory, kind auto-detected from its metadata file. |
| `describe_bundle(path)` | Kind, model name, horizon and metrics of a bundle directory without loading it. |

All four bundle kinds share the serving interface: `predict_from_raw(raw_df,
calibrate=True, ...)` for raw bars and `predict(X)` for already-prepared input.
Bundles contain pickles (scaler, calibrator, sklearn-style models), and
unpickling can execute code: only load bundles you produced or trust. The
manifest's checksums let you verify a bundle was not modified after it was
written.

### Meta-labeling output

```python
bundle = load_deploy_artifact(deploy_dir, horizon=5)     # MetaLabelingBundle
meta = bundle.predict_meta(raw_df)
meta.directions            # primary side per bar
meta.meta_probabilities    # P(the primary's bet pays off)
meta.trade_mask            # side taken AND P(win) >= threshold
meta.positions             # side x P(win) on traded bars, 0 elsewhere
meta.n_trades
```

### Several bundles at once

`UniversalInferencePipeline` serves a set of bundles (mixed ranks) plus an
optional ensemble, e.g. to compare base models on live bars:

```python
from src.inference import UniversalInferencePipeline

pipe = UniversalInferencePipeline.from_bundles(
    ["run/bundles/xgboost_h5", "run/bundles/lstm_h5"],
    ensemble_path="run/bundles/ensemble",
)
one = pipe.predict_from_raw(raw_df, bundle_index=1)    # the LSTM
every = pipe.predict_all(raw_df)                        # one result per bundle
combined = pipe.predict_ensemble(raw_df)                # the stacked prediction
spread = pipe.predict_with_uncertainty(raw_df)          # metadata["prob_std"] across models
print(pipe.summary())
```

## Warmup

Features are rolling statistics, so the first bars of any input window have no
valid features and are dropped (the returned `timestamps` say which bars were
predicted). How much history to pass:

| Needed by | Bars of history (at the training bar timeframe) |
|---|---|
| Most rolling features | ~200 |
| MTF features (`data.mtf.enabled`, default on) | **≥ 500** — below that the MTF columns cannot be reproduced and prediction raises |
| Wavelet features | ≥ 64 |
| 3D / 4D models | plus `seq_len − 1` bars for the first window (60–128) |
| Ensembles | the largest requirement among the base models (predictions are the bars every base model covers) |

EMA-type and session-cumulative features converge rather than switch on, so
more history than the minimum brings served features closer to the training
values. In practice pass **1,000+ bars** and use the last row(s). If there is
too little history, prediction fails loudly with the number of bars each
feature family needs rather than returning skewed values.

For live use, keep a rolling buffer of recent raw bars and call
`predict_from_raw` on it when a bar closes; the last row is the signal for that
bar, to be acted on at the next bar (as the [backtest](concepts.md#backtest-execution-timing)
assumed).
