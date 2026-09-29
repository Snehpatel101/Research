# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (0.x while the public API settles).
Phase-by-phase engineering detail lives in `COMPLETION.md`.

## [Unreleased]

### Added
- Mix and match: any subset of 16 base models × 5 stacking meta-learners
  (`voting_meta` new) × 4 training modes (standard, walk-forward, regime-aware,
  meta-labeling), each trained, stacked, backtested, bundled, deployed and
  served from raw OHLCV bars; `scripts/mix_match.py` verifies every combination
  including prediction parity between the deployed bundle and the trained model.
- `data.bar_timeframe` resampling, per-model contract sequence lengths,
  sample-uniqueness weights, AFML meta-labeling, label-span purging on bar
  positions, derived purge/embargo, exact PSR/DSR, CSCV PBO, CPCV path assembly.
- CI on uv (ruff, black, pyright, vulture, fast tests; weekly slow tests),
  `make check`, pre-commit, `slow` test marker.
- Seeded runs: `MLFactory.run` seeds Python/NumPy/torch from `random_seed`
  before any data work; the seed reaches every model, the Optuna sampler and
  its trials, feature selection, walk-forward windows and the meta-learner.
  `deterministic` config flag for deterministic torch kernels; CLI `--seed`.
- `run_manifest.json` in every run directory (config + `config_hash`, seed,
  git commit/dirty, package versions, torch/CUDA environment, data SHA-256, rows
  and time range, timing, status/error, final metrics); `ExperimentResult.manifest_path`;
  the deploy manifest references it and `validate_deploy_artifact` verifies it.
- Experiment tracking section `tracking` (`none` | `local` | `mlflow`,
  `tracking_uri`, `experiment_name`; CLI `--tracking`, `--tracking-uri`): one
  parent run per factory run, one child run per model. `mlflow` is an optional
  extra (`pip install '.[mlflow]'`); the MLflow tracker uses `MlflowClient` (no
  global active run) and a missing install fails the run before training.
- Determinism test: two runs in separate interpreters (different
  `PYTHONHASHSEED`) give bit-identical OOF, ensemble holdout and deployed
  predictions.

### Fixed
- Backtest filled at the open of the bar whose close produced the signal
  (lookahead); circuit breakers ended the simulation; label costs were in
  dollars instead of price points.
- Every horizon trained on the first horizon's labels.
- OOF fold models early-stopped on the fold they predicted; the tuner early-
  stopped on the fold it scored; label-overlap purging was a silent no-op.
- Clustered MDA feature ranking ignored the target.
- Cross-rank stacking paired predictions from different bars; inference used a
  different feature engine than training; several save/load round-trips failed.
- `random_seed` never reached the models: model seeds, the Optuna sampler, MDA
  feature ranking and the meta-learner were hard-coded to 42.
- Random-forest predictions changed in the last bits on every call (scikit-learn
  sums trees in thread-completion order when `n_jobs != 1`): MDA gave unused
  features random ±1e-17 importances, so identical runs could select different
  features. Forests now predict in a fixed tree order (row-parallel for the
  `random_forest` model); LightGBM runs with `deterministic=True` on CPU.
- Feature rankings let float noise and set iteration order (PYTHONHASHSEED)
  break ties: importances are now quantized to 12 significant digits and ties
  broken by feature name (`optimization/feature_selection/ranking.py`), and
  stable-feature counts are built in sorted order.
- Every `Trainer` silently logged a local tracking run and copied its model
  checkpoints into it (a hidden `global.yaml` default); tracking is now off
  unless `tracking.backend` is set, and local runs reference artifacts instead
  of copying them.

### Removed
- Dead serving/monitoring chain, aspirational config layer, phantom types and
  ~23k further lines of verified dead code.
