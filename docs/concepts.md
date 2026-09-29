# Concepts

This page explains the methodology behind ML Factory and, for each piece, the
failure it prevents. Most of it follows Marcos López de Prado, *Advances in
Financial Machine Learning* (AFML, 2018), and the backtest-overfitting papers
by Bailey, Borwein, López de Prado and Zhu. Module paths point at the
implementation.

## The problem with fixed-horizon labels

The textbook label — "is the close in *h* bars higher than now?" — ignores the
path. A trade that is stopped out at −2 ATR before recovering to +0.1 is
labeled a winner; a label that ignores volatility treats a 1-point move in a
quiet market the same as in a crash. Models trained on such labels learn
something no strategy can trade.

## Triple-barrier labels

`src/data/labeling/triple_barrier.py`

Every bar *i* opens a hypothetical trade at `close[i]` with three barriers:

- **upper** at `close[i] + (k_up + cost) × ATR[i]` → label **+1** (long wins)
- **lower** at `close[i] − (k_down + cost) × ATR[i]` → label **−1** (short wins)
- **time** after `max_bars` bars → label **0** (neutral: no barrier touched)

The first barrier touched decides the label; bars too close to the end of the
data get `−99` (invalid) and are dropped. ATR is Wilder's ATR(14) computed from
OHLCV — the same ATR the backtester uses.

Why this shape:

- **Volatility-scaled.** Barriers in ATR units mean the same thing in calm and
  volatile markets.
- **Path-dependent.** The label is what a stop-loss/take-profit trade would
  actually have experienced.
- **Asymmetric where needed.** Equity index futures drift up, so the MES table
  uses `k_up > k_down` (h5: 1.5 / 1.0) to keep long and short labels balanced;
  MGC is symmetric.
- **Cost-aware.** Both barriers are widened by the round-trip trading cost in
  ATR units (see [Costs](#transaction-costs)), so a +1 label means *profitable
  after costs*, not merely "went up".

The parameters come from one place, `ExperimentConfig.resolve_barrier_params`:
an explicit `data.labeling.upper_mult` / `lower_mult` / `max_holding_bars`
override if set, else the per-symbol, per-horizon table in
`src/data/pipeline/config/barriers_config.py`. The **labeler, the backtester and
the CV purge all read that one resolution**, so they always play the same game.

`data.labeling.binary_mode` remaps the labels to {0: time-out, 1: a barrier was
hit} — a "will it move?" classifier. Binary labels carry no direction, so the
directional backtest is skipped for them.

## Label spans

`src/core/label_spans.py`

A label decided at bar *i* is only *known* once a barrier is touched,
`bars_to_hit` bars later. The closed interval `[i, i + bars_to_hit]` is the
label's **span**; the factory stores its end as a `label_end_h<horizon>` column
(integer bar positions, so it survives row filtering, splitting, windowing and
subsampling). Two things depend on spans: purging and uniqueness weights.

## Purge and embargo

`src/validation/cv/purged_kfold.py`, `ExperimentConfig.resolve_cv_gaps`

Ordinary k-fold on time series leaks in two ways:

1. **Overlapping labels.** A training bar just before the test block has a
   label that resolves *inside* the test block. The model trains on an outcome
   from the test period.
2. **Serial correlation.** Features right after the test block are strongly
   correlated with those at its end, so training on them is training on
   near-copies of test rows.

ML Factory's answers:

- **Purge**: drop training rows within `purge_bars` before each test block.
  Default `purge_bars` = the **longest label span** = `max_bars` over the
  configured horizons (MES h5 → 12 bars; horizons 5–20 → 50). An explicit
  smaller value is raised to it with a warning — anything shorter leaks.
- **Span purge**: on top of the fixed gap, drop *every* training row whose
  actual span `[i, label_end_i]` overlaps the test block's span. The fixed gap
  covers the worst case; the span purge covers the exact case.
- **Embargo**: drop `embargo_bars` training rows *after* each test block.
  Default = **one trading day** of bars at the bar timeframe (1,440 minutes:
  CME equity and metal futures trade ~23 h a day, so one day ≈ one session),
  capped at 25% of a CV fold so small datasets still have training rows. At
  5-minute bars that is 288 bars.

The chronological train/validation/test split uses the same gaps: `purge_bars`
between train and validation, `max(purge_bars, embargo_bars)` between
validation and test.

Override with `training.purge_bars` / `training.embargo_bars` (or
`--purge-bars` / `--embargo-bars`). The run logs the derived values:

```text
Derived embargo_bars=26 (1440 min of 5min bars, capped at 25% of a 107-bar fold)
CV gaps: purge_bars=12 (longest label span 12 bars), embargo_bars=26 (bar timeframe 5min)
```

## Sample uniqueness weights

`src/core/label_spans.py` (`average_uniqueness`, `uniqueness_sample_weights`)

Triple-barrier labels overlap: ten consecutive bars in a slow trend may all
resolve on the same future bar. They are not ten independent observations, and
treating them as such lets long, overlapping stretches dominate the fit.

For each bar *t*, the **concurrency** *c<sub>t</sub>* is the number of training
labels whose span contains *t*. A label's **average uniqueness** is the mean of
1/*c<sub>t</sub>* over its span (AFML §4.4): 1.0 for a label that overlaps
nothing, 1/*k* for *k* labels on identical spans. Training samples are weighted
by it, rescaled to mean 1 so loss magnitudes and weight-based regularizers
(e.g. XGBoost `min_child_weight`) stay comparable. Concurrency is computed from
the training split only, so evaluation labels never shape the weights.
`training.sample_weighting="none"` turns it off.

## Feature engineering without lookahead

`src/data/pipeline/stages/features/`, `src/data/pipeline/stages/mtf/`

`FeatureEngineer` computes one fixed feature set (momentum, trend, volatility,
volume, microstructure proxies, entropy, wavelets, regime, temporal). The rules
that keep it honest:

- A feature at bar *i* uses information up to bar *i*'s close at most; features
  built from statistics that include the current bar in a way a live system
  could not (entropy, regime, microstructure, wavelets) are shifted by one bar.
- **Multi-timeframe (MTF) features** resample to the higher timeframe with
  left-closed, left-labeled bars and apply `shift(1)` *before* forward-filling
  to the base timeframe, so a 5-minute bar only sees *completed* 15- and
  60-minute bars.
- Cumulative features (OBV, VWAP, cumulative order flow) reset at session
  boundaries, so their level does not encode position in the file.

**Feature selection** (`src/models/training/feature_selection.py`) runs per
model on the **training rows only**: low-variance and correlation pre-filters,
then clustered, target-aware MDA (permutation importance with log-loss scoring,
on a temporal stride subsample for large data), capped by the model's contract
(`max_features`). Selecting on all rows would leak the future into the choice of
inputs — the classic "textbook" leak.

## Out-of-fold (OOF) stacking

`src/validation/cv/oof_*.py`, `src/models/training/services/ensemble_service.py`

To stack models, the meta-learner needs predictions for every training bar made
by a model that did **not** see that bar. For each base model:

1. Purged k-fold over the training split (`training.n_splits`, default 5).
2. A fresh model per fold, fit on the fold's training rows. When it early-stops
   (boosting rounds, neural best epoch), it does so on a **purged tail of its
   own training rows** — never on the fold it predicts. Stopping on the held-out
   fold would let the held-out rows choose the model and make the OOF
   predictions optimistic.
3. Its predictions on the held-out fold fill that fold's OOF rows.

Models of different input ranks produce predictions for different bars: a 3D
sequence model with window 60 has no prediction for the first 59 bars. OOF rows
are therefore **aligned on the source bar each prediction belongs to**, and the
ensemble uses the bars every model covers. Stacking features are each model's
class probabilities plus three derived columns (mean confidence, agreement,
entropy).

The meta-learner is evaluated honestly and then refit:

1. The most recent 20% of aligned OOF rows are a **holdout**; meta-train rows
   within `purge_bars` of it are dropped.
2. An evaluation fit on the remaining rows (early stopping on a purged tail of
   those rows) is scored on the holdout, and so is every base model — one
   metric, one set of rows, a fair comparison. These are `ensemble_metrics`.
3. The deployed meta-learner is refit on **all** aligned OOF rows.

| Meta-learner | What it is | When to use |
|---|---|---|
| `ridge_meta` (default) | L2-regularized multinomial logistic regression | Robust default; learns per-model weights |
| `xgboost_meta` | Gradient-boosted trees | Many base models, non-linear interactions |
| `mlp_meta` | Small neural network | Non-linear combinations, enough OOF rows |
| `calibrated_meta` | Ridge classifier with isotonic/Platt calibration (time-series CV) | You need well-calibrated ensemble probabilities |
| `voting_meta` | Average of the base models' probabilities; no fit | Baseline that cannot overfit the OOF rows |

**Probability calibration** (`training.calibration`) is fitted per model on its
validation split and shipped in the bundle. `method="auto"` uses isotonic
regression only when every class has at least 1,000 validation samples (it
overfits below that) and Platt scaling otherwise.

## Training modes

### Standard

Chronological train/validation/test split with purge/embargo gaps; OOF
predictions from purged k-fold on the training split; early stopping and
calibration on validation; one **one-shot** evaluation on the test split
(logged as such — iterate on validation metrics, not test).

### Walk-forward

`src/models/training/modes/walk_forward.py`

`training_mode="walk_forward"` re-fits each model on successive windows and
predicts the window after each training cutoff, with the purge gap in between
— the closest offline analogue of running the model live. `expanding` windows
grow the training set; `rolling` windows keep its length fixed (useful when old
regimes are no longer relevant). Each window **selects features and fits its
scaler on its own training rows**, because anything fitted on the whole
training split would hand early windows statistics from their future.

The window predictions are the out-of-sample signals used for stacking and the
backtest; the metrics are means over windows. The *deployed* model is trained
exactly as in standard mode (walk-forward is the evaluation protocol, not a
different model).

### Regime-aware

`src/models/training/regime_trainer.py`, `src/inference/regime_bundle.py`

One model per market regime. Each bar is assigned a regime from trailing data
only — the percentile rank of rolling volatility over `training.regime.lookback`
bars (`volatility_percentile`), ADX trend strength with a per-symbol threshold
(`trend_adx`), or both (`combined`) — with 2 (low/high) or 3 (low/medium/high)
regimes. A model is trained on each regime's bars; a regime with fewer than 100
training samples is served by a default regime's model. The `RegimeBundle`
detects the regime of every incoming bar and routes it to that regime's model.

Why: the relationship between features and outcomes differs between quiet and
volatile markets; a single model averages them.

### Meta-labeling

`src/models/training/training_ops.py`, `src/inference/meta_labeling_bundle.py`

AFML ch. 3 splits the decision in two: a **primary** model picks the side, a
**meta** model decides whether to take the bet.

1. The primary model (`training.models[0]`, any rank) trains normally; its
   **OOF** predictions give an honest side for every training bar.
2. Meta-labels exist only where the primary takes a side: 1 if the bet paid off
   (label == side), else 0.
3. Meta features are the primary's model input plus its OOF class
   probabilities and confidence — built by the same function at training and at
   serving time.
4. The meta-model (`training.meta_labeling.meta_model`: logistic, random forest
   or a boosting model) learns P(bet pays off), cross-validated with purging.
   A bet is taken only when that probability is at least
   `training.meta_labeling.threshold`.

Why: the primary can be tuned for recall (catch opportunities) while the meta
model buys precision (skip the bad ones), and P(win) doubles as a bet size
(`MetaLabelingBundle.predict_meta` returns `positions = side × P(win)` on traded
bars). Using OOF — not in-sample — primary predictions matters: in-sample the
primary is nearly always right, so the meta model would learn to trust it
blindly.

## Backtest execution timing

`src/inference/backtesting/backtest.py`

A prediction at row *i* uses features that include bar *i*'s close, and its
label is anchored at `close[i]`. The signal is therefore only known once bar *i*
has closed, and it is acted on at bar `j = i + signal_delay_bars`:

| Execution model | Fill price at bar *j* | Minimum delay | Barriers watched from |
|---|---|---|---|
| `MARKET_ON_OPEN` (default) | `open[j]` | 1 | bar *j* |
| `MIDPOINT` | `(high[j] + low[j]) / 2` | 1 | bar *j + 1* |
| `MARKET_ON_CLOSE` | `close[j]` | 0 | bar *j + 1* |
| `FILL_AT_SIGNAL` | `close[j]` | 0 | bar *j + 1* |

By default a signal at bar *i* **fills at the open of bar i + 1**. Filling at
`close[i]` would assume zero decision and routing latency — the optimistic limit
— and filling at any price inside bar *i* would be lookahead.

Once in a trade, the backtest applies **the label's own barriers**: stop and
take-profit at `(k + cost_in_atr) × ATR[i]` from the fill, a time exit at the
label's time barrier (`close[i + max_bars]`), all with the same `k_up`,
`k_down`, `max_bars` and cost term the labeler used. Stops and targets fill at
the barrier price (not the bar close); when both are touched in one bar the stop
wins (conservative). Stops pay slippage. The remaining difference between label
and backtest — the close-to-next-open gap — is real P&L the label never sees.

Circuit breakers (daily loss, consecutive losses, drawdown) **pause** trading
and flatten positions rather than ending the simulation; halts are recorded in
the result. Positions are force-closed at session end. Equity is marked to
market at every bar's close. Sharpe and volatility are annualized from the
data's bar frequency, not a hard-coded 252.

## Transaction costs

`src/inference/backtesting/costs.py`, `src/data/pipeline/config/barriers_config.py`

Per round trip: commission ($2.50) + exchange fee ($0.52) + NFA fee ($0.02) =
$3.04 per contract, plus slippage of one tick per fill (entry and exit); the
backtest passes the entry and exit volatility (ATR / price) to its slippage
model so it can charge more in fast markets. In ticks that is 2.43 (MES), 3.04 (MGC) and
6.08 (MNQ) for the fixed part.

The **same cost** enters the labels: the round-trip cost in price units
(`ticks × tick_size`) divided by the **median ATR of the training rows** gives
`cost_in_atr`, which widens both barriers. It is calibrated on training rows
only, so validation/test volatility cannot shape training labels, and the
backtester reuses that exact scalar. Override the backtest costs with
`evaluation.commission_per_contract` / `evaluation.slippage_ticks`.

## Measuring overfitting

Trying many configurations and keeping the best one inflates its measured
performance even when none of them has an edge. Three tools quantify that.

### PSR and DSR

`src/validation/deflated_sharpe.py`

The **Probabilistic Sharpe Ratio** is the probability that the true Sharpe
ratio exceeds a benchmark SR\*, given the sample length *T*, skewness
\(\gamma_3\) and kurtosis \(\gamma_4\) of the returns (all Sharpe ratios
per-period, not annualized):

\[
\mathrm{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{T-1}}
{\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}}\right)
\]

The **Deflated Sharpe Ratio** sets the benchmark to the Sharpe ratio you would
expect from the best of *N* trials with *no* skill:

\[
SR_0 = \sqrt{V[\{SR_n\}]}\left((1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N}\right)
+ \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right),
\qquad \mathrm{DSR} = \mathrm{PSR}(SR_0)
\]

where \(V[\{SR_n\}]\) is the variance of the trial Sharpe ratios and \(\gamma\)
the Euler–Mascheroni constant. DSR is a probability; the gate deploys at ≥ 0.95.
Optuna studies optimizing a Sharpe-like metric report the DSR of the best trial
(it is skipped for bounded metrics such as F1, where the math does not apply).

### CPCV

`src/validation/cv/cpcv.py`

Walk-forward gives one backtest path — one draw from the distribution of
outcomes. **Combinatorial Purged Cross-Validation** partitions the data into *N*
groups and tests on every combination of *k* groups (purged and embargoed around
each), then reassembles the out-of-sample predictions into
\(\binom{N-1}{k-1}\) complete backtest paths, each covering every bar once
(AFML §12.4). The default `ml cpcv-pbo` setting (N=6, k=2) gives 15 splits and 5
paths: a distribution of Sharpe ratios instead of a single number.
`training.cv_method="cpcv"` makes the Optuna tuner score trials on CPCV.

### PBO

`src/validation/cv/pbo.py`

The **Probability of Backtest Overfitting** (Bailey et al., 2017) asks: when you
pick the best configuration in-sample, how often is it below the median
out-of-sample? With a *T × N* matrix of per-period returns for *N*
configurations, CSCV splits the rows into *S* blocks and, for every half/half
combination, ranks the in-sample winner out-of-sample. PBO is the share of
combinations where it lands at or below the median. `ml cpcv-pbo` computes it
across models (warn at 0.5, block at 0.8 by default) from next-bar strategy
returns net of per-symbol costs.

## Reproducibility

- One `ExperimentConfig` drives a run and is saved as
  `<run_dir>/experiment_config.yaml`; it reloads to an identical config.
- `random_seed` seeds training.
- Each stage checkpoints (`checkpoints/`); `MLFactory(cfg).run(resume=True)` or
  `ml run --resume <run_dir>` continues after a crash, and a changed config is
  detected by hash.
- Bundles record everything needed to replay the transform at inference; see
  [Deploy and serve](deploy-and-serve.md).

## Further reading

- M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018 —
  ch. 3 (triple barrier, meta-labeling), ch. 4 (uniqueness), ch. 7 (purged CV),
  ch. 12 (CPCV).
- D. Bailey, M. López de Prado, "The Deflated Sharpe Ratio", *Journal of
  Portfolio Management*, 2014.
- D. Bailey, J. Borwein, M. López de Prado, Q. Zhu, "The Probability of Backtest
  Overfitting", *Journal of Computational Finance*, 2017.
