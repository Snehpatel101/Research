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

One pipeline sits behind every command: raw OHLCV bars -> `FeatureEngineer` ->
triple-barrier labels (with label spans) -> models. There are no intermediate
stage directories; every command takes the raw bars (`-d`, parquet or csv).

```bash
python -m src.cli run -d data/raw/MES_1m_1month.parquet --bar-timeframe 5min \
    -m xgboost,lstm,patchtst --build-ensemble --meta-learner voting_meta -h 5
```

`ml` below is shorthand for `python -m src.cli`.

| Command | Purpose |
|---------|---------|
| `ml run` | Full pipeline: features, labels, training (one model, many, or an ensemble; `--training-mode standard\|walk_forward\|regime_aware\|meta_labeling`), optional `--backtest`, bundle + deploy artifact. `--resume <run_dir>` continues a run from its last checkpoint. |
| `ml data` | Features and labels to parquet only (`<run_dir>/features_labels.parquet`), no training. |
| `ml status <run_dir>` | Checkpoint progress of a run. |
| `ml models [name]` | Registered models, or one model's default configuration. |
| `ml cv` | Purged k-fold CV of tabular models: fold stability, prediction correlation, out-of-fold stacking datasets. |
| `ml walk-forward` | Expanding/rolling walk-forward evaluation of tabular models (sequence models: `ml run --training-mode walk_forward`). |
| `ml cpcv-pbo` | CPCV backtest paths (next-bar returns net of per-symbol costs) and the PBO overfitting gate over 2+ models. |

The evaluation commands (`cv`, `walk-forward`, `cpcv-pbo`) use the same data step
as `ml run`, evaluate the chronological train split only (validation and test
stay untouched), scale per fold, and purge on each label's actual end bar with
derived purge/embargo (`--purge-bars` / `--embargo-bars` override). Results go
to `<output_dir>/<run_id>/{cv,walk-forward,cpcv-pbo}/`.

Outputs of `ml run` live in `<output_dir>/<run_id>/` (default `experiments/`):
`experiment_config.yaml`, `checkpoints/`, per-model artifacts, `bundles/` and
`deploy/manifest.json`. Run `ml <command> --help` for every option.

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
