Read CLEANUP_PLAN.md (roadmap) and CLEANUP_TASKS.md (details) for $ARGUMENTS.

Orchestrate 7 sequential task agents with context handoffs:

Implementation (Agents 1-6):
- Each handles one task category from CLEANUP_TASKS.md
- Run `ruff check .` after each agent
- Pass full context to next agent

Post-Implementation (Agent 7):
- Review remaining issues
- Run full validation checklist:
  - `ruff check .` and `pyright topstepx_backend/` pass
  - Backend starts without errors
  - Frontend builds (`npm run build`)

Per CLAUDE.md: delete don't adapt, small diffs, verify compiles.
