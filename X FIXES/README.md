# X FIXES (Static + Architectural Deep Dive)

This folder contains a repo-wide static analysis focused on why the current codebase (especially `src/`) does **not yet** cleanly support the intended end-goal:

> Upload one dataset → automatically produce **nine** timeframe datasets → train a **single model** or **heterogeneous ensemble** with proper feature practices, meta-learners, and model management for OHLCV.

## Documents

- `X FIXES/ISSUE_LIST_DETAILED.md` — The main deliverable: prioritized issue list with evidence + impact + recommended fix direction.
- `X FIXES/GOAL_GAP_MATRIX.md` — Goal → required capabilities → current implementation → gaps.
- `X FIXES/INCONSISTENCY_CATALOG.md` — Concrete contradictions across docs/code/config/paths that create “split-brain”.
- `X FIXES/DATASET_TIMEFRAME_STRATEGY.md` — What “nine datasets” can mean, and the architectural implications (labels/splits/horizons/MTF).
- `X FIXES/PROGRAM_ALIGNMENT_BLUEPRINT.md` — Concrete “program contracts” to align fanout datasets, per-model views, training/CV, and bundling.
- `X FIXES/CONFIG_SINGLE_SOURCE_OF_TRUTH.md` — How to collapse the current multiple config systems into one authoritative layer.
- `X FIXES/REMEDIATION_ROADMAP.md` — Sequenced roadmap with acceptance criteria to reach the target workflow.

## Ground Rules Followed

- No code edits were made in this session.
- These documents reference existing files for evidence; they do not change behavior.
