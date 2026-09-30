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

### Changed
- Historical audits, investigation notes and phase reports moved from the
  repository root and `docs/` to `docs/archive/` (history kept with `git mv`);
  `COMMANDS.md` moved to the root.
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
