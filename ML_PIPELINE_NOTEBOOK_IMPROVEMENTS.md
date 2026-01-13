# Improving the OHLCV ML Pipeline Notebook

This document outlines practical improvements to make your OHLCV training notebook easier to use (especially with Colab local kernels), more reliable for financial time series, and better aligned with real trading outcomes. The emphasis is on usability, path consistency, leakage control, and Optuna optimization that reflects financial performance rather than only ML metrics.

## 1) Runtime selection and notebook UX

Make runtime selection explicit and front-and-center. A single configuration cell should let the user choose among `auto`, `colab_hosted`, and `local`. If the user selects `local` while running in Colab, the notebook should print instructions for "Connect to local runtime" and continue using local paths without attempting to mount Drive.

Keep the setup cell lightweight. Do not install packages by default. Gate installs behind a boolean like `INSTALL_DEPS`. In local kernels this avoids unexpected dependency changes; in Colab it keeps the startup consistent.

## 2) Unify paths around a single repo root

The most common notebook errors come from inconsistent paths. Solve this by resolving a single repo root once and deriving all other paths from it. Use this precedence order: explicit override (`REPO_ROOT`), git root if present, then current working directory. Once `PROJECT_ROOT` is set, everything else is derived: `RAW_DATA_DIR`, `SPLITS_DIR`, `EXPERIMENTS_DIR`, `CHECKPOINT_DIR`. Avoid mixing absolute and relative paths across cells.

For Colab, Drive should be treated as a data storage layer, not a code root. If Drive is mounted and `DRIVE_DATA_PATH` is set, map raw data to `DRIVE_ROOT / DRIVE_DATA_PATH`, but keep code and outputs under the repo. This prevents half the notebook writing to Drive and half to the repo, which is a common source of confusion.

## 3) Data integrity and OHLCV specifics

Financial data problems are often silent. Add a short, fast data audit cell before any training. It should validate:
- Column presence (open, high, low, close, volume).
- Monotonic timestamp ordering and duplicate timestamps.
- Missing bars (gaps) and session boundaries.
- Sanity checks such as `low <= open/close <= high` and non-negative volume.

Be explicit about timezone and session handling. For indicators, verify that no look-ahead is introduced by the library or by rolling windows that accidentally include the current bar when the model is trained to predict the next bar.

## 4) Splits, leakage control, and evaluation discipline

Time-series splits should be the default. Your notebook already mentions purging and embargo; make it visible and configurable at the split stage, and always log the date ranges for each fold. This helps audit leakage and ensures consistency between training, validation, and test.

Provide two evaluation modes: a fast single holdout split for iteration and a slower walk-forward or time-series CV for reliable estimates. When using Optuna, use the time-series CV path, not a random split. The notebook should log which method was used in the final summary so it is clear how the scores were produced.

## 5) Training metrics that reflect trading reality

Classification metrics are necessary but insufficient in finance. Add an optional evaluation cell that computes trading-aware metrics from model predictions, even if it is a simplified backtest. At minimum, include cumulative return, max drawdown, Sharpe or Sortino, turnover, and win rate after costs and slippage. This makes it easy to see when a model with a strong F1 score is unprofitable in practice.

Also report label distribution and class imbalance after filtering invalid labels. Imbalance often varies by horizon. A quick per-horizon distribution table helps explain why some horizons are difficult and prevents misinterpretation of macro F1 or accuracy.

## 6) Optuna improvements for financial optimization

The objective should align with financial outcomes. For each trial, train and evaluate using a time-aware split, then compute a metric like Sharpe, Calmar, or profit factor. Optimize that metric, not just loss. If training is expensive, use fewer folds but keep the time ordering intact.

Use Optuna best practices to make studies reliable and reusable. Store studies in SQLite so they can be resumed. Use a stable `study_name` that includes symbol, horizon, model family, and feature set. Use a sampler appropriate to the space (TPE works well for mixed continuous and categorical parameters). Add pruning based on intermediate validation metrics that are meaningful for the model type (epoch-level for neural networks, boosting iteration metrics for tree models).

Record more than a single objective. Use `trial.set_user_attr` to store drawdown, turnover, and other risk metrics. This makes it possible to filter out models with unacceptable risk characteristics even if they have high Sharpe. It also makes it easier to perform multi-objective selection after the fact.

## 7) Reproducibility and artifact tracking

A good notebook run should be reproducible. Always save a run manifest that includes the full configuration, a git commit hash (if available), and a dataset fingerprint (hash or last modified time).

Log runtime details that affect results: device type, GPU model, RAM, batch size, and effective sequence length. In Colab, hardware can vary across sessions, so this context is critical for comparing results.

## 8) Maintainability and structure

The notebook should orchestrate, not implement. Move repeated logic into helper modules so the notebook cells call concise functions. This reduces duplication and makes errors less likely. A small `notebooks/helpers.py` or `src/utils/notebook_helpers.py` can house these pieces.

## 9) Sources consulted

The following sources informed these recommendations:

- Optuna documentation (pruning, samplers, persistent studies): https://optuna.readthedocs.io/
- scikit-learn documentation on time-series splitting: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- General guidance on leakage control in financial ML: Marcos Lopez de Prado, *Advances in Financial Machine Learning* (2018).

## Closing note

The highest impact changes are simple: make runtime selection explicit, unify paths through a single repo root, and align optimization with trading-aware metrics. These changes make the notebook more reliable for local kernels, reduce path errors, and produce results that are more meaningful for real trading decisions. You can also add a small \"quick start\" cell with a single-model run and an expected runtime so new users can validate their setup before committing to long training runs.
