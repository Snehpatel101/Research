# Remote Full-Runtime Validation (Google Colab)

The local dev machine is memory-constrained: the full application (neural /
transformer training, 1M+ row datasets, walk-forward with 3D/4D tensors) must
NOT be run locally. Local verification stops at the layered strategy below;
anything above Layer 4 runs in Colab.

## Validation layers

| Layer | What | Where | Command |
|-------|------|-------|---------|
| 1 | Static (ruff, black, pyright) | local | `ruff check src/ tests/ && black --check src/ tests/ && pyright src/` |
| 2 | Unit + behavioral tests (~600) | local | `python3 -m pytest tests/ -q` (~8 min, ~3.6 GB peak) |
| 3 | Quick behavioral tier | local | `make test-quick` |
| 4 | Mini E2E (tiny synthetic, boosting only) | local | `python3 -m pytest tests/test_factory_e2e.py -q` |
| 5 | Full runtime (all 12 models, real data) | **Colab** | below |

## Layer 5 — Colab procedure

1. **Runtime**: GPU runtime (T4 minimum; A100/H100 for the 1M+ row datasets —
   the memory work in Phases 72–79 targets ~59 GB peak for TCN walk-forward).

2. **Setup** (first cell):
   ```bash
   !git clone <repo-url> research && cd research
   %cd research
   !pip install -q -r requirements-colab.txt
   ```
   `requirements-colab.txt` deliberately pins `pandas==2.2.2` to match Colab's
   pre-installed stack and omits `river` (pandas conflict — install with
   `pip install river --no-deps` only if drift detection is needed).

3. **Data**: mount Drive and point `config.data.data_path` at the 1-min OHLCV
   parquet (`from google.colab import drive; drive.mount('/content/drive')`).

4. **Run**: use `notebooks/ml_factory_colab.ipynb` (the maintained notebook) or:
   ```python
   from src.config.experiment import ExperimentConfig
   from src.factory import MLFactory

   cfg = ExperimentConfig()
   cfg.data.symbol = "MES"                # or MGC / MNQ
   cfg.data.data_path = "/content/drive/MyDrive/data/mes_1min.parquet"
   cfg.training.models = ["xgboost", "lightgbm", "catboost", "lstm", "tcn", "patchtst"]
   cfg.training.horizons = [5]
   result = MLFactory(cfg).run()
   print(result.summary())
   ```

5. **Expected outputs / pass criteria**:
   - `result.metrics` non-empty for every model; `result.backtest_metrics`
     non-empty with `total_trades > 0` (this was silently `{}` before the
     2026-08 fix — if it regresses to `{}`, check the Backtester merge for
     column collisions in the factory `_run_evaluation` prices frame).
   - Log line per horizon shows the barrier source, e.g.
     `label_h5: k_up=1.5 k_down=1.0 max_bars=12 [barriers table]` — the SAME
     values must appear in the backtest config (parity guarantee).
   - Bundle + deploy manifest written under `cfg.output_dir`
     (`load_deploy_artifact(deploy_dir, horizon)` loads and predicts).
   - Re-running the same config/seed reproduces identical model metrics.

6. **Failure diagnostics**:
   - CUDA OOM: models auto-retry with halved batch; if a model still dies,
     reduce `cfg.training.batch_size` (512 default) or drop transformers.
   - `Insufficient data` ValueError: `n_samples >= n_splits*100 +
     n_splits*(purge+embargo)`; default embargo is 1440 bars.
   - Checkpoint resume after a runtime disconnect:
     `MLFactory(cfg).run(resume=True)` (also `ml run --resume` via CLI).
