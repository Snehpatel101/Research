# ML Factory

Config-driven factory for financial time-series models: put OHLCV bars in, get
a leakage-free, cost-aware, backtested model or ensemble out — packaged so that
production inference replays training exactly.

```text
Raw OHLCV ─► features + triple-barrier labels ─► per-model feature selection (train rows only)
          ─► any mix of 16 models (2D tabular / 3D sequence / 4D multi-timeframe)
          ─► purged-CV out-of-fold predictions ─► stacking meta-learner
          ─► cost-aware backtest (fills at bar i+1) ─► bundles + deploy manifest
          ─► load_deploy_artifact(...).predict_from_raw(raw_bars)
```

## 30-second quickstart

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,stats]"

# One week of bundled MES 1-minute bars → 5-minute XGBoost model, backtested and deployed (<1 min).
# --no-mtf: multi-timeframe features need ~1,500 5-minute bars of warmup, more than a week holds
python -m src.cli run -d data/raw/MES_1m_1week.parquet --bar-timeframe 5min --no-mtf -m xgboost -h 5 --backtest
```

```python
import pandas as pd
from src.config.experiment import ExperimentConfig
from src.factory import MLFactory
from src.inference import load_deploy_artifact

cfg = ExperimentConfig(name="mes_mix")
cfg.data.data_path = "data/raw/MES_1m_1month.parquet"
cfg.data.bar_timeframe = "5min"                           # resample 1-minute bars
cfg.training.models = ["xgboost", "lstm", "patchtst"]    # 2D + 3D + 4D
cfg.training.meta_learner = "voting_meta"
cfg.training.horizons = [5]
cfg.training.optuna.n_trials = 0                          # skip tuning on a first run
cfg.training.max_epochs = 5                               # quick look on a CPU
cfg.evaluation.run_backtest = True

result = MLFactory(cfg).run()
print(result.summary())

artifact = load_deploy_artifact(result.deploy_path, horizon=5)   # the ensemble
# ~2,000 5-minute bars: the MTF features' ~1,500-bar warmup, then ~500 scored bars
raw = pd.read_parquet("data/raw/MES_1m_1month.parquet").iloc[-10000:]
pred = artifact.predict_from_raw(raw)   # same resampling, features, scaling, routing
pred.class_predictions, pred.class_probabilities, pred.metadata["timestamps"]
```

No data at hand? `python examples/01_quickstart.py` generates synthetic bars
and runs end to end in about a minute ([examples](examples/README.md)).

## Mix and match

| Building block | Choices |
|---|---|
| Base models (any subset) | `xgboost` `lightgbm` `catboost` `random_forest` `logistic` `svm` · `lstm` `gru` `tcn` `transformer` `inceptiontime` `resnet1d` `nbeats` `tft` · `patchtst` `itransformer` |
| Meta-learner | `ridge_meta` `xgboost_meta` `mlp_meta` `calibrated_meta` `voting_meta` |
| Training mode | `standard` · `walk_forward` · `regime_aware` (one model per regime, routed per bar) · `meta_labeling` (primary `models[0]` + meta bet filter) |

Models of different input ranks are aligned on the bar each prediction belongs
to, so a boosted tree, an LSTM and a multi-timeframe PatchTST stack cleanly.
Every combination runs end to end through deploy and `predict_from_raw` in the
[verification matrix](docs/MIX_AND_MATCH.md) (`scripts/mix_match.py`).
[Adding a model](docs/mix-and-match.md#adding-a-model) takes a `BaseModel`
subclass and a contract.

## Guarantees

| Guarantee | How |
|---|---|
| No lookahead in features | Higher-timeframe features from completed bars only (`shift(1)`); lagged entropy/regime/microstructure features; session-reset cumulative features |
| No label leakage in CV | Purge = longest label span, embargo = one trading day, plus purging on every label's actual end bar |
| Honest out-of-fold predictions | Fold models early-stop on a purged tail of their own training rows |
| Labels and backtest agree | One barrier resolution and one cost term feed the labeler and the backtester |
| No same-bar fills | Signal at bar *i* fills at bar *i + 1* (open); stops/targets at the barrier price |
| Train/serve parity | Bundles replay bar timeframe, `FeatureEngineer` spec, scaler and calibrator; checked per combination |
| Overfitting is measured | PSR/DSR, CPCV paths, CSCV PBO (`ml cpcv-pbo`) |
| Reproducible | One saved `ExperimentConfig`, seeded runs, checkpoint/resume |

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, data format, first run (CLI and Python), predicting |
| [Concepts](docs/concepts.md) | Triple-barrier labels, purge/embargo, uniqueness weights, OOF stacking, meta-labeling, regimes, walk-forward, execution timing, costs, DSR/PSR/PBO/CPCV — and why each exists |
| [Mix and match](docs/mix-and-match.md) | Models × meta-learners × modes; how to add a model |
| [Lopez de Prado options](docs/afml-options.md) | Opt-in CUSUM event sampling, fractional differentiation, probability bet sizing |
| [Configuration](docs/configuration.md) | Every `ExperimentConfig` field and default (generated) |
| [CLI](docs/cli.md) | Every `ml` command and option (generated) |
| [Deploy and serve](docs/deploy-and-serve.md) | Bundles, deploy manifest, `predict_from_raw`, warmup |
| [Examples](examples/README.md) | Three scripts, a few minutes each on a CPU |
| [Colab notebook](docs/USER_GUIDE.md) | GPU runs on large datasets |

Browse it as a site with `make docs-serve` (`uv pip install -e ".[docs]"`), or
build it with `make docs` (strict: any warning fails).

## Development

```bash
make check     # uv lock --check, ruff, black --check, pyright (0 errors), vulture, fast tests
make test      # full suite incl. slow end-to-end tests
make docs      # regenerate-check + strict docs build
make matrix    # full mix-and-match matrix (hours)
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Project process docs live at the root:
[CLAUDE.md](CLAUDE.md) (status, conventions), [DIRECTION.md](DIRECTION.md)
(architecture), [CLEANUP_PLAN.md](CLEANUP_PLAN.md) /
[CLEANUP_TASKS.md](CLEANUP_TASKS.md) (phases), [COMPLETION.md](COMPLETION.md)
(history), [DECISIONS.md](DECISIONS.md) (open decisions),
[COMMANDS.md](COMMANDS.md) (agent command system),
[CHANGELOG.md](CHANGELOG.md). Historical audits and phase reports are archived
in [docs/archive/](docs/archive/README.md).

MIT licensed — see [LICENSE](LICENSE).
