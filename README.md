# ML Factory

Config-driven factory for financial time-series models: put OHLCV bars in, get a
leakage-free, backtested, deployable model or ensemble out.

```
Raw OHLCV ─► features + triple-barrier labels ─► per-model feature selection
          ─► any mix of 16 models (2D tabular / 3D sequence / 4D multi-stream)
          ─► purged-CV out-of-fold predictions ─► stacking meta-learner
          ─► cost-aware backtest ─► bundles + deploy artifact
          ─► predict_from_raw(raw_bars)   # same features, same scaling, same routing
```

## Mix and match

| Building block | Choices |
|---|---|
| Base models | `xgboost` `lightgbm` `catboost` `random_forest` `logistic` `svm` · `lstm` `gru` `tcn` `transformer` `inceptiontime` `resnet1d` `nbeats` `tft` · `patchtst` `itransformer` |
| Meta-learner (stacking) | `ridge_meta` `xgboost_meta` `mlp_meta` `calibrated_meta` `voting_meta` |
| Training mode | `standard` · `walk_forward` · `regime_aware` (one model per market regime, routed per bar) · `meta_labeling` (primary = `models[0]` + bet filter) |

Any subset of base models can be combined, in any mode, with any meta-learner.
Models of different input ranks are aligned on the bar each prediction belongs
to, so a gradient-boosted tree, an LSTM and a multi-timeframe PatchTST stack
cleanly. Every combination is verified end to end — see
[`docs/MIX_AND_MATCH.md`](docs/MIX_AND_MATCH.md) and `scripts/mix_match.py`.

## Install

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
uv pip install -e ".[dev,stats]"
```

## Quick start (Python)

```python
from src.config.experiment import ExperimentConfig
from src.factory import MLFactory

cfg = ExperimentConfig(name="mes_mix")
cfg.data.symbol = "MES"
cfg.data.data_path = "data/raw/MES_1m_1month.parquet"
cfg.data.bar_timeframe = "5min"            # resample 1-minute input to 5-minute bars
cfg.training.models = ["xgboost", "lstm", "patchtst"]
cfg.training.meta_learner = "voting_meta"
cfg.training.training_mode = "standard"    # or walk_forward / regime_aware / meta_labeling
cfg.training.horizons = [5]
# purge/embargo are derived: purge = longest label span (triple-barrier max_bars),
# embargo = one trading day of bars at the bar timeframe (set training.purge_bars /
# training.embargo_bars to override). CV also purges on every label's actual end bar,
# and training samples are weighted by label uniqueness (training.sample_weighting).
cfg.training.optuna.n_trials = 0           # first run: skip Optuna (default 100 trials per model)
cfg.evaluation.run_backtest = True

result = MLFactory(cfg).run()
print(result.summary())
```

## Quick start (CLI)

```bash
python -m src.cli run -d data/raw/MES_1m_1month.parquet --bar-timeframe 5min \
    -m xgboost,lstm,patchtst --build-ensemble --meta-learner voting_meta -h 5
```

## Serve

```python
from src.inference import load_deploy_artifact

artifact = load_deploy_artifact(result.deploy_path, horizon=5)   # ensemble if one was built
pred = artifact.predict_from_raw(raw_ohlcv_df)                   # raw bars in, same as training
pred.class_predictions, pred.class_probabilities, pred.metadata["timestamps"]
```

Bundles replay training exactly: the recorded bar timeframe and
`FeatureEngineer` spec rebuild the features, the training scaler and calibrator
are applied, regime bundles route each bar to its regime's model, and
meta-labeling bundles neutralize bars the meta-model rejects. Give
`predict_from_raw` enough history for warmup (MTF features need ≥ 500 bars;
EMA-type and cumulative features converge with more).

## Verify

```bash
ruff check src/ && black --check src/ && pyright   # lint, format, types (0 errors)
pytest                                             # unit + end-to-end tests
python scripts/mix_match.py pairs                   # every pair of models, full pipeline
python scripts/mix_match.py report                  # regenerate docs/MIX_AND_MATCH.md
```

## Project docs

`CLAUDE.md` (status and conventions) · `DIRECTION.md` (architecture) ·
`CLEANUP_PLAN.md` / `CLEANUP_TASKS.md` (phases) · `COMPLETION.md` (history) ·
`DECISIONS.md` (open decisions) · `docs/USER_GUIDE.md` (Colab notebook guide)
