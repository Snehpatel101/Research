# Lopez de Prado options (opt-in)

Three techniques from *Advances in Financial Machine Learning* are available as
config; every default keeps the current behavior.

```python
# cfg = ExperimentConfig(...) as in Getting started
cfg.data.labeling.event_sampling = "cusum"        # AFML ch. 2: label CUSUM event bars only
cfg.data.labeling.cusum_threshold = "auto"        # or a log-return number; auto = multiple x TRAIN vol
cfg.data.labeling.cusum_vol_multiple = 3.0        # ~one event per 9 bars for i.i.d. returns
cfg.data.features.frac_diff.enabled = True        # AFML ch. 5: ffd_log_{close,open,high,low}
cfg.data.features.frac_diff.d = "auto"            # smallest d passing ADF on TRAIN rows (needs the `stats` extra)
cfg.evaluation.position_sizing = "probability"    # AFML ch. 10: size from the predicted probability
cfg.evaluation.bet_max_contracts = 5              # contracts at full size
```

- **Event sampling.** Features are computed on every bar; only bars where a
  symmetric CUSUM filter on the log returns fires carry a label (the others are
  invalid, -99, and dropped by training, CV, feature selection and stacking).
  Label spans stay in bar coordinates, so purging and uniqueness weights follow
  the events. The auto threshold is fitted on the leading training bars only
  (walk-forward: before the first test window) and frozen into the bundle: the
  validation/test holdout never influences it, while purged-CV folds inside the
  training split see a value fitted on all of it. The CV embargo counts event
  samples and is sized to cover the bar embargo even where events cluster; the
  chronological val/test gap keeps the embargo in bars. The backtest opens
  positions on event bars only. `predict_from_raw`
  still scores every bar — the model was trained on event bars, so act where
  `pred.metadata["is_event"]` is true (the CUSUM sums reset at every event: pass
  the same history start as training for identical events).
- **Fractional differentiation.** `ffd_log_<col>` are fixed-window FFD of the log
  price (window `data.features.frac_diff.window`, lagged one bar like every
  feature), so training and serving agree exactly. `d="auto"` is fitted on the
  leading training bars (same prefix as the CUSUM threshold; at least 0.05) and
  frozen into the recorded `FeatureEngineer` spec: inference replays the same `d`.
- **Probability bet sizing.** Size = `2 * Phi((p - 1/K) / sqrt(p (1 - p))) - 1`
  of `bet_max_contracts`, with `p` the predicted probability of the chosen side
  and `K` the class count: zero at `p = 1/K`, growing with `p`.
  `evaluation.bet_step_size` discretizes the size. In the backtest `p` is the
  prediction's `confidence`: the uncalibrated maximum class probability (a vote
  share for a hard `voting_meta`); a missing probability means no bet. At serve time call
  `afml_bet_size(p, n_classes)` (`src.inference.backtesting`) with the winning
  class probability, or with a meta-labeling bundle's
  `metadata["meta_probability"]` and `n_classes=2`.

Run them through the harness with `python scripts/mix_match.py custom xgboost,lstm
--set data.labeling.event_sampling=cusum --set data.features.frac_diff.enabled=true
--set evaluation.position_sizing=probability` (`--set` takes any config field).
