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

### Removed
- Dead serving/monitoring chain, aspirational config layer, phantom types and
  ~23k further lines of verified dead code.
