# ML Factory - Remaining Tasks

**Status:** Phases 0-10 Complete | Production Ready
**Last Updated:** 2026-01-24

---

## Completed

See COMPLETION.md for details on Phases 0-10.

---

## Phase 11: Deferred Items (Low Priority)

| Task | Description | Location | Action |
|------|-------------|----------|--------|
| 5C | Deployment bundle | `src/factory.py` | Add tar.gz packaging |
| 4C | Diversity analysis | `src/models/ensemble/` | Wire DiversityAnalyzer to training |
| 4D | Deflated Sharpe | `src/validation/` | Add DSR gate post-Optuna |
| 4E | Bootstrap CIs | `src/models/evaluation/` | Wire BootstrapCI to reports |
| 4F | Auto calibration | `src/models/training/` | Wire CalibrationManager |
| 4G | Bet sizing | `src/inference/` | Connect BetSizer to backtest |
| - | MTF ablation | `src/config/` | Add `mtf.enabled` flag |

These integrate existing standalone systems into the main workflow. Not blocking.

---

*For completed phase details, see COMPLETION.md*
