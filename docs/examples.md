# Examples

The [`examples/`](https://github.com/Snehpatel101/Research/tree/main/examples)
directory holds three runnable scripts. Each generates 4,000 synthetic 5-minute
bars in-script, runs the full pipeline on a CPU in a few minutes and serves the
result from raw bars.

```bash
OMP_NUM_THREADS=1 python examples/01_quickstart.py
OMP_NUM_THREADS=1 python examples/02_mix_and_match_ensemble.py
OMP_NUM_THREADS=1 python examples/03_walk_forward_and_meta_labeling.py
```

| Script | Shows | CPU time |
|---|---|---|
| `01_quickstart.py` | One model (XGBoost): features → triple-barrier labels → purged CV → backtest → bundle + deploy manifest → `predict_from_raw` | ~1 min |
| `02_mix_and_match_ensemble.py` | XGBoost (2D) + LSTM (3D) + PatchTST (4D) stacked by `voting_meta`; the deployed `EnsembleBundle` aligns the three on bar timestamps | ~2.5 min |
| `03_walk_forward_and_meta_labeling.py` | `walk_forward` evaluation and `meta_labeling` (primary side + meta bet filter) with `predict_meta` | ~1.5 min |

Outputs land in `experiments/examples/<name>/<run_id>/`. The data is a random
walk, so metrics are at chance level — the scripts show the workflow.

## 01 — quickstart

```python
cfg = ExperimentConfig(name="quickstart", output_dir=OUTPUT_ROOT / "quickstart")
cfg.data.symbol = "MES"
cfg.data.data_path = data_path
cfg.data.mtf.enabled = False          # keep the demo fast
cfg.training.models = ["xgboost"]
cfg.training.horizons = [5]
cfg.training.n_splits = 3
cfg.training.optuna.n_trials = 0
cfg.evaluation.run_backtest = True

result = MLFactory(cfg, verbose=0).run()

artifact = load_deploy_artifact(result.deploy_path, horizon=5)
pred = artifact.predict_from_raw(pd.read_parquet(data_path).iloc[-1000:])
```

```text
validation    : accuracy 0.432, macro F1 0.300, log loss 1.053
backtest      : 95 trades, win rate 48.4%, Sharpe -5.07, net P&L $-153.43 (after costs)

last 5 of 917 predictions from raw bars:
                     p_short  p_neutral  p_long  signal
datetime
2024-01-16 06:45:00    0.495      0.138   0.367      -1
```

## 02 — mix and match

```python
cfg.training.models = ["xgboost", "lstm", "patchtst"]   # 2D + 3D + 4D
cfg.training.meta_learner = "voting_meta"
cfg.training.build_ensemble = True
cfg.training.max_epochs = 1                              # demo budget
```

```text
  xgboost_h5     macro F1 0.298
  lstm_h5        macro F1 0.284
  patchtst_h5    macro F1 0.302
  ensemble       macro F1 0.311 (scored on a purged holdout of the OOF rows)

EnsembleBundle: 858 predictions from 1000 raw bars
```

The ensemble returns fewer rows than the raw input: features and the 60-bar
sequence windows need warmup, and the ensemble keeps the bars every base model
predicted ([Deploy and serve](deploy-and-serve.md#warmup)).

## 03 — walk-forward and meta-labeling

```python
cfg.training.training_mode = "walk_forward"
cfg.training.walk_forward.n_windows = 3
...
cfg.training.training_mode = "meta_labeling"
cfg.training.models = ["xgboost"]                # models[0] is the primary
cfg.training.meta_labeling.meta_model = "logistic"
cfg.training.meta_labeling.threshold = 0.5

meta = load_deploy_artifact(result.deploy_path, horizon=5).predict_meta(raw)
meta.directions, meta.meta_probabilities, meta.trade_mask, meta.positions
```

See [Concepts → Training modes](concepts.md#training-modes) for what each mode
does and why.
