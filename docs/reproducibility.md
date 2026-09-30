# Reproducibility and tracking

## Seeded runs

`random_seed` (CLI `--seed`, default 42) seeds Python, NumPy and torch before any
data work and reaches every model (`random_state` / `random_seed`), the Optuna
sampler and each of its trials, feature selection (MDA rankings, event sampling,
governance diagnostics), the stacking meta-learner and the standalone evaluators
(`ml cv`, `ml walk-forward`, `ml cpcv-pbo`).

Same config + seed + data + code gives bit-identical out-of-fold predictions,
ensemble holdout predictions and deployed-bundle predictions on CPU.
`tests/e2e/test_determinism_e2e.py` runs the pipeline twice in separate
interpreters with different `PYTHONHASHSEED`s and compares them bit for bit;
`tests/integration/test_threaded_determinism.py` does the same for
multi-threaded random-forest / LightGBM fits and MDA rankings in the fast suite.

What makes that hold:

- Forests predict in a fixed tree order (scikit-learn otherwise sums trees in
  thread-completion order, so probabilities differ in the last bits between
  calls and near-tied feature importances swap ranks).
- Feature rankings round every score to 12 significant digits of its own
  magnitude and break ties by feature name; nothing iterates a set of names.
- LightGBM runs with `deterministic=True` on CPU.

`deterministic: true` additionally forces deterministic torch kernels (needed on
GPU; slower).

## Run manifest

Every run writes `<run_dir>/run_manifest.json`. It says `running` until the run
ends `success` or `failed` (with the error type and message) and records:

- **provenance** (fixed at start, fingerprinted by `provenance_sha256`): the full
  config and its `config_hash`, the seed, the git commit, dirty flag and a hash
  of the uncommitted diff, package versions, the Python / torch / CUDA
  environment and the input data's SHA-256 (a directory dataset is hashed file
  by file);
- **data**: rows, time range, source and bar timeframes, training rows;
- **results**: final metrics, backtest summary and artifact paths;
- **tracking**: the tracker backend and run ID.

`config_hash` covers every setting that changes results. Naming, placement and
reporting fields (`name`, `description`, `run_id`, `output_dir`, `verbose`,
`tracking`, and the governance report/registry paths) are left out, so re-running
an experiment under a new run ID keeps its hash. Credentials in URIs
(`scheme://user:pass@host`) are masked.

`result.manifest_path` points to the manifest; `deploy/manifest.json` references
it (relative path plus `provenance_sha256`), and `validate_deploy_artifact`
checks the reference.

### Resume

`ml run --resume <run_dir>` (or `MLFactory(config).resume_from_checkpoint()`)
continues from the last checkpoint when the run's `config_hash` matches the one
the checkpoints were written with; tracking, verbosity or names may change. A
mismatch is refused with an error and the checkpoints are kept;
`resume_from_checkpoint(restart_on_config_change=True)` discards them and starts
over. A resumed run keeps its original manifest provenance and appends a
`resumes` entry (time, stage, code and package versions of the resume).

## Experiment tracking

```yaml
random_seed: 42
tracking:
  backend: mlflow                # none (default) | local | mlflow
  tracking_uri: http://localhost:5000
  experiment_name: mes_research  # default: the experiment name
```

- `local` writes JSON runs under `<output_dir>/tracking/` (no dependencies;
  artifacts are recorded by reference, not copied).
- `mlflow` needs the optional extra (`uv pip install -e ".[mlflow]"`). A run that
  asks for MLflow without it fails before training instead of falling back.
  Directories such as model checkpoints are recorded by path, not uploaded.

Each `MLFactory.run` opens one parent run (flattened config, config hash,
commit, data fingerprint, final metrics, backtest summary, artifact paths) and
every trained model logs a child run under it. Tracking is best-effort inside
training: a tracking-server failure is a warning, never a failed or unsaved
model.
