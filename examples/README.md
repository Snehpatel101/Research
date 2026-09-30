# Examples

Three self-contained scripts. Each generates 4,000 synthetic 5-minute OHLCV
bars in-script (a seeded random walk, so no data download), runs the full
pipeline and serves the result from raw bars. Run them from the repository
root after installing (`make install-dev`, or see the
[getting started guide](../docs/getting-started.md)):

```bash
OMP_NUM_THREADS=1 python examples/01_quickstart.py
OMP_NUM_THREADS=1 python examples/02_mix_and_match_ensemble.py
OMP_NUM_THREADS=1 python examples/03_walk_forward_and_meta_labeling.py
```

| Script | Shows | CPU time* |
|---|---|---|
| [`01_quickstart.py`](01_quickstart.py) | One model (XGBoost): features → triple-barrier labels → purged CV → backtest → bundle + deploy manifest → `load_deploy_artifact(...).predict_from_raw(raw_bars)` | ~1 min |
| [`02_mix_and_match_ensemble.py`](02_mix_and_match_ensemble.py) | A 2D + 3D + 4D ensemble (XGBoost + LSTM + PatchTST) stacked by `voting_meta`, deployed as an `EnsembleBundle` and served from raw bars | ~2.5 min |
| [`03_walk_forward_and_meta_labeling.py`](03_walk_forward_and_meta_labeling.py) | `walk_forward` training (LightGBM, 3 expanding windows) and `meta_labeling` (XGBoost primary + logistic meta-model), with the `MetaLabelingBundle`'s directions, P(win), trade mask and positions | ~1.5 min |

\*Measured single-threaded on a laptop-class CPU; neural models run one epoch.

Outputs go to `experiments/examples/<name>/<run_id>/` (gitignored): the saved
config, checkpoints, bundles and `deploy/manifest.json`.

**Expect chance-level metrics.** A random walk has no edge; the examples
demonstrate the workflow, not a strategy. Point `cfg.data.data_path` at real
bars (parquet or csv with `open, high, low, close, volume` and a datetime index
or column) to train on your own data, and raise `training.max_epochs` /
`training.optuna.n_trials` for real runs.

`_synthetic.py` holds the shared helpers (synthetic bars, quiet logging, and a
`sys.path` entry so the scripts import `src` from this checkout).
