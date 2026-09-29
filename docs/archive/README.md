# Archive: historical audits and process reports

These documents record how ML Factory got to its current state: audits,
investigation notes, verification reports and implementation plans from
earlier phases. They are kept for history and are **not maintained** — code
paths, numbers and recommendations in them may be out of date. The current
state lives in the root documents (`CLAUDE.md`, `DIRECTION.md`,
`CLEANUP_PLAN.md`, `CLEANUP_TASKS.md`, `COMPLETION.md`, `DECISIONS.md`) and the
user documentation in `docs/`. The archive is excluded from the docs site.

Moved here in Phase 117 with `git mv`, so `git log --follow <file>` shows each
file's full history.

## Audits and reviews

| Document | What it is |
|---|---|
| [AUDIT_2026-02-26.md](AUDIT_2026-02-26.md) | Comprehensive audit (17 items, addressed in Phases 80–84) |
| [AUDIT_SYMBOL_ROBUSTNESS.md](AUDIT_SYMBOL_ROBUSTNESS.md) | Multi-symbol robustness audit |
| [FEATURE_SELECTION_AUDIT.md](FEATURE_SELECTION_AUDIT.md) | Feature selection audit |
| [HARDCODEFIXES.md](HARDCODEFIXES.md) | Hardcoded values / backtest execution mismatch audit (Phase 86) |
| [REPO_AUDIT.md](REPO_AUDIT.md) | Repository audit for intraday trading |
| [PIPELINE_REVIEW_2026-02-02.md](PIPELINE_REVIEW_2026-02-02.md) | Critical review of the pipeline |
| [SNEH_SNEH_SNEH.md](SNEH_SNEH_SNEH.md) | Full-stack adversarial audit (source of THEETASKLIST) |
| [SNEH.md](SNEH.md) | Notebook EDA investigation notes |
| [CHILL.md](CHILL.md) | Verified issues registry |
| [full-review/00-scope.md](full-review/00-scope.md) | Scope of a full review pass |

## Plans and task lists

| Document | What it is |
|---|---|
| [THEETASKLIST.md](THEETASKLIST.md) | Verified implementation plan (Phases 103–113) |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Research-backed financial / ML improvement plan |
| [PERFORMANCE_FIXES.md](PERFORMANCE_FIXES.md) | Performance anti-patterns in feature engineering |
| [NOTEBOOK_PIPELINE_GAP_ANALYSIS.md](NOTEBOOK_PIPELINE_GAP_ANALYSIS.md) | Notebook vs pipeline configuration gaps |
| `audit/` | Deploy-artifact audit and plans (Phases 51–52): [phase 1 findings](audit/phase1-audit/CONSOLIDATED-FINDINGS.md), [phase 2 roadmap](audit/phase2-planning/UNIFIED-ROADMAP.md), [phase 3 master plan](audit/phase3-implementation/MASTER-IMPLEMENTATION-PLAN.md), [deploy-artifact plan](audit/deploy-plan/FINAL-DEPLOY-ARTIFACT-PLAN.md) |

## Test and verification reports

| Document | What it is |
|---|---|
| [EndtoEndtests.md](EndtoEndtests.md) | End-to-end smoke test of all models (2026-02-19) |
| [PIPELINE_VERIFICATION_REPORT.md](PIPELINE_VERIFICATION_REPORT.md) | Pipeline verification report |
| [COLLAB_TOM.md](COLLAB_TOM.md) | Colab full-test-suite command sheet |

## Model configuration notes

| Document | What it is |
|---|---|
| [JACOB.md](JACOB.md) | Model configuration and ensemble guide (VRAM-adaptive defaults) |
| [CELL2_MODEL_CONFIGURATIONS.md](CELL2_MODEL_CONFIGURATIONS.md) | Notebook cell 2 model configurations |
