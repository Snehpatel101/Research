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

**Warmup.** Features look back over a bounded window (session-reset OBV/VWAP,
rolling windows) or an exponentially fading one (EMA-based indicators).
`FeatureEngineer.warmup_bars()` derives from the feature definitions how many
bars that takes (SMA-200, EWMs until their start weighs less than the spec's
`ewm_settle_tolerance` = 1e-3, the wavelet windows, each MTF timeframe's
indicators in base bars — 320 5-minute bars without MTF, 1,476 with 15/60-minute
MTF). Training and serving apply the same rule: a bar is scored only once that
many bars precede it and, for the session-reset features, once the bars they
read (the scored bar and the 21 before it at 5 minutes) all lie past the
input's first session. A session is a **calendar date of the bar timestamps**
(naive UTC after cleaning), not an exchange session open. So every row a bundle
returns from a short window equals, to within 1e-3 of each feature's spread,
the row it returns from the full history. The exception is
`metadata["event_flags"]` with CUSUM event sampling: the filter is path
dependent, so its flags depend on where the history starts (see
`PreprocessingGraph.event_flags`). The warmup is recorded in the bundle's
`preprocessing_graph.json`; passing fewer bars raises a `ValueError` naming the
raw bars needed (after resampling to the training bar timeframe), and training
refuses data shorter than the warmup before computing features.

**Feature engine version.** Bundles record the `FEATURE_ENGINE_VERSION` that
computed their training features. Loading a bundle built by another version (or
one that recorded none) raises, because the model would see differently computed
inputs; retrain it, or pass `allow_engine_mismatch=True` to `load_bundle`,
`load_deploy_artifact`, `ModelBundle.load` or `UniversalInferencePipeline.from_*`
to serve it anyway (logged as an error). `validate_deploy_artifact` reports such
bundles as invalid.

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

The primary inside the bundle is refit on all training rows, so on bars from
the training range its probabilities are in-sample (more confident than the
out-of-fold ones the meta-model learned from); judge the filter on bars after
the training range.

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

A bar is scored only once its features no longer depend on where the input
starts — the same rule training applied to its own first bars (see *Warmup*
above). The returned `timestamps` say which bars were scored. How much history
to pass (at the training bar timeframe):

| Needed by | Bars of history |
|---|---|
| Default features, no MTF, 5-minute bars | `warmup_bars` = 320 |
| MTF features (`data.mtf.enabled`, default on; 15/60-minute) | 1,476 5-minute bars — the hourly MACD needs 123 hourly bars to forget its start |
| 1-minute bars | periods scale ×5: 1,114 without MTF, 7,380 with 60-minute MTF |
| Session-reset features (OBV, VWAP) | the scored bar and the `volume_sma` + 1 bars before it (21 at 5 minutes) must lie past the input's first calendar date |
| 3D models | plus `seq_len − 1` bars for the first window |
| Ensembles | the largest requirement among the base models |

The exact number is `preprocessing_graph.json`'s `warmup_bars` (or
`FeatureEngineer.warmup_bars()`). With too little history `predict_from_raw`
raises a `ValueError` that names the minimum number of raw bars (e.g. five
times as many 1-minute bars for a 5-minute model).

For live use, keep a rolling buffer of recent raw bars and call
`predict_from_raw` on it when a bar closes; the last row is the signal for that
bar, to be acted on at the next bar (as the [backtest](concepts.md#backtest-execution-timing)
assumed).
