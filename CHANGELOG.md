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
- CI hardening: fast tests on Python 3.11 and 3.12, `uv lock --check`,
  packaging smoke job (sdist + wheel built, wheel installed into a fresh venv,
  every module imported, all 21 models registered, CLI run; `make
  wheel-smoke`), every CI and `make install` environment pinned to `uv.lock`
  (`scripts/lock_constraints.sh`, CPU torch at the locked version), uv pinned
  via `[tool.uv] required-version`, Dependabot for Actions, uv (lockfile-only)
  and pre-commit hooks (weekly, minor/patch grouped), read-only token, job
  timeouts, superseded PR runs cancelled (never runs on main).
- Documentation site (mkdocs-material + mkdocstrings, `make docs` /
  `make docs-serve`, `docs` extra, CI `docs` job with `--strict`): getting
  started, concepts (the methodology and why each piece exists), mix and match,
  deploy and serve, API reference; configuration and CLI reference pages
  generated from the code (`scripts/gen_config_docs.py`,
  `scripts/gen_cli_docs.py`); Markdown link checker (`scripts/check_md_links.py`).
- `examples/`: quickstart, 2D+3D+4D ensemble, walk-forward + meta-labeling —
  each runs in a few minutes on a CPU on synthetic bars.
- Opt-in Lopez de Prado options (defaults leave results unchanged):
  `data.labeling.event_sampling: "cusum"` (AFML ch. 2; labels only CUSUM event
  bars, threshold `"auto"` = multiple of the training-rows volatility, frozen
  into the bundle; the backtest acts on event bars, `predict_from_raw` flags
  them in `metadata["is_event"]`), `data.features.frac_diff` (AFML ch. 5;
  fixed-window `ffd_log_*` features, `d: "auto"` fitted on training rows and
  replayed from the recorded feature spec) and
  `evaluation.position_sizing: "probability"` (AFML ch. 10 bet size from the
  predicted probability, `ProbabilityBetSizer`).
- `ExperimentConfig.validate()` (checked by `MLFactory`); `scripts/mix_match.py
  --set KEY=VALUE` config overrides; YAML numbers in scientific notation without a
  dot (`1e-5`) load as numbers; backtest summaries report `zero_size_signals`.

### Changed
- Historical audits, investigation notes and phase reports moved from the
  repository root and `docs/` to `docs/archive/` (history kept with `git mv`);
  `COMMANDS.md` moved to the root.
- Walk-forward runs calibrate the labeler's cost term (and the CUSUM threshold /
  auto frac-diff d) on the bars before the first test window instead of the
  whole training split.
- `PipelineConfig.split_embargo_bars`: the val/test embargo stays in bars when
  event sampling makes the CV embargo count samples.
- `frac_diff_ffd(..., max_window=)` and `find_min_d(..., threshold=, max_window=)`
  accept an explicit window (the default keeps the data-length-dependent cap).
- The pre-training leakage check ignores invalid-label (-99) rows, and
  `UnifiedDataPreparation.prepare` no longer warns about them (they are dropped
  by `filter_invalid_labels`).
- Packaging metadata: SPDX `license = "MIT"` + `license-files` (PEP 639,
  setuptools>=77); the sdist ships the package, README, LICENSE and CHANGELOG
  only (no tests, scripts, docs or project notes). The unused path constants
  (`src/core/paths.py`, `src/models/config/paths.py`) are deleted.

### Fixed
- A wheel install shipped without `config/global.yaml` (it lived outside the
  package), so every process-wide default fell back to its hard-coded value;
  the file moved to `src/config/global.yaml` and ships as package data.
- Backtest filled at the open of the bar whose close produced the signal
  (lookahead); circuit breakers ended the simulation; label costs were in
  dollars instead of price points.
- Every horizon trained on the first horizon's labels.
- OOF fold models early-stopped on the fold they predicted; the tuner early-
  stopped on the fold it scored; label-overlap purging was a silent no-op.
- Clustered MDA feature ranking ignored the target.
- Cross-rank stacking paired predictions from different bars; inference used a
  different feature engine than training; several save/load round-trips failed.

### Removed
- Dead serving/monitoring chain, aspirational config layer, phantom types and
  ~23k further lines of verified dead code.
- 15 stale ad-hoc scripts in `scripts/` (one-off smoke/verification runs
  superseded by `tests/e2e/`, `scripts/mix_match.py` and `ml run`, plus a
  finished migration tool); `scripts/` is now linted and formatted in CI,
  `make check` and pre-commit, and `notebooks/colab_test_runner.ipynb` runs
  `scripts/mix_match.py`.
