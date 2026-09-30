# Getting started

## Install

ML Factory targets Python 3.11+ and uses [uv](https://docs.astral.sh/uv/) for
environments.

```bash
git clone https://github.com/Snehpatel101/Research.git ml-factory && cd ml-factory
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
uv pip install -e ".[dev,stats]"
```

`make install-dev` does the same (plus the git hooks). Optional extras:
`stats` (statsmodels, for ADF tests), `hmm` (hmmlearn), `docs` (this site).

Check the install:

```bash
python -m src.cli models          # every registered model, by family
```

## Your data

Any OHLCV bar file works, parquet or csv:

| Requirement | Detail |
|---|---|
| Columns | `open`, `high`, `low`, `close`, `volume` (case-insensitive) |
| Time | a `DatetimeIndex`, or a `datetime` / `date` column |
| Bar size | any whole number of minutes, detected from the median spacing; `data.bar_timeframe` resamples to a coarser size (e.g. 1-minute input, 5-minute training bars) |
| Cleaning | automatic and shared with inference: timestamps made naive UTC, sorted, duplicates dropped |

`data.symbol` selects the contract presets: triple-barrier parameters, tick
size and value, commission and slippage, session hours and the ADX regime
threshold. `MES`, `MGC` and `MNQ` have contract and cost presets, `MES` and
`MGC` also tuned barrier tables; any other symbol gets the defaults.

The repository ships sample bars in `data/raw/` — `MES_1m_1week.parquet`
(one week of 1-minute MES) is enough for a first run.

## First run: CLI

```bash
python -m src.cli run -d data/raw/MES_1m_1week.parquet --bar-timeframe 5min \
    -m xgboost -h 5 --backtest
```

This takes well under a minute on a laptop. It:

1. resamples the 1-minute bars to 5 minutes and computes the features;
2. labels every bar with the MES horizon-5 triple barrier
   (`k_up=1.5`, `k_down=1.0` × ATR, time barrier 12 bars, costs included);
3. derives the CV gaps from the data — `purge_bars=12` (the label span) and
   `embargo_bars` = one trading day of 5-minute bars, capped at 25% of a fold;
4. selects features on the training rows, trains XGBoost, produces purged
   out-of-fold predictions and calibrates probabilities on the validation split;
5. backtests the out-of-sample signals (fills at the next bar's open,
   stop/take-profit on the label's barriers, commission + slippage);
6. writes a model bundle and a deploy manifest.

Everything lands in `experiments/<run_id>/`:

```text
experiments/<run_id>/
├── experiment_config.yaml   # the full config; reload with ExperimentConfig.from_yaml
├── checkpoints/             # per-stage checkpoints (`ml status`, `ml run --resume`)
├── cache/                   # labeled feature frame, feature spec, label costs
├── run_standard_<ts>/       # trained models, OOF predictions, per-model reports
├── bundles/xgboost_h5/      # model + scaler + calibrator + preprocessing graph
└── deploy/manifest.json     # which bundle serves each horizon
```

Add models, an ensemble and a training mode with flags:

```bash
python -m src.cli run -d data/raw/MES_1m_1month.parquet --bar-timeframe 5min \
    -m xgboost,lstm,patchtst --build-ensemble --meta-learner voting_meta -h 5 --backtest
```

See the [CLI reference](cli.md) for every command (`ml data`, `ml cv`,
`ml walk-forward`, `ml cpcv-pbo`, ...).

## First run: Python

```python
from src.config.experiment import ExperimentConfig
from src.factory import MLFactory

cfg = ExperimentConfig(name="mes_first_run")
cfg.data.symbol = "MES"
cfg.data.data_path = "data/raw/MES_1m_1week.parquet"
cfg.data.bar_timeframe = "5min"            # resample 1-minute input to 5-minute bars
cfg.training.models = ["xgboost"]
cfg.training.horizons = [5]
cfg.training.optuna.n_trials = 0           # default is 100 Optuna trials per model
cfg.evaluation.run_backtest = True

result = MLFactory(cfg).run()
print(result.summary())
result.metrics["xgboost_h5"]["macro_f1"]   # validation metrics, keyed <model>_h<horizon>
result.backtest_metrics["sharpe_ratio"]    # net of costs
```

Python defaults differ from the CLI in two places: `ExperimentConfig` trains
four horizons (`[5, 10, 15, 20]`) and runs 100 Optuna trials per model, while
the CLI defaults to one horizon and no tuning. Set them explicitly for a quick
first run. `ExperimentConfig` writes to `experiments/runs/<run_id>/`.

## Predict from raw bars

```python
import pandas as pd
from src.inference import load_deploy_artifact

artifact = load_deploy_artifact(result.deploy_path, horizon=5)
raw = pd.read_parquet("data/raw/MES_1m_1week.parquet").iloc[-3000:]   # raw 1-minute bars
pred = artifact.predict_from_raw(raw)       # resampled, featured, scaled, calibrated

pred.class_predictions          # -1 short, 0 neutral, +1 long
pred.class_probabilities        # (n, 3) in the order short, neutral, long
pred.metadata["timestamps"]     # the bar each prediction belongs to
```

Give `predict_from_raw` enough history for the features to warm up (see
[Deploy and serve](deploy-and-serve.md#warmup)).

## Try it without data

The [examples](examples.md) generate synthetic bars in-script and run in a few
minutes on a CPU:

```bash
OMP_NUM_THREADS=1 python examples/01_quickstart.py
```

## Development checks

```bash
make check        # uv lock --check, ruff, black --check, pyright (0 errors), vulture, fast tests
make test         # full suite incl. slow end-to-end tests
make docs         # build this site with --strict
make matrix       # the full mix-and-match verification matrix (hours)
```

## Next

- Understand the methodology: [Concepts](concepts.md).
- Combine models: [Mix and match](mix-and-match.md).
- Tune every knob: [Configuration](configuration.md).
- Run the Colab notebook on a GPU: [Colab notebook guide](USER_GUIDE.md).
