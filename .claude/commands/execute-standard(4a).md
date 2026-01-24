Read CLEANUP_TASKS.md for task details on $ARGUMENTS.

Use 4 sequential task agents with context handoffs:
1. Agent 1: First task category
2. Agent 2: Second task category
3. Agent 3: Third task category
4. Agent 4: Fourth task category

After implementation, run verification:
- `ruff check .` passes
- `pyright topstepx_backend/` passes (0 errors)
- Backend starts without errors
- Report any regressions

Per CLAUDE.md: delete don't adapt, verify compiles before committing.
