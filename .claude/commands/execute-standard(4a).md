Read CLEANUP_TASKS.md for task details on $ARGUMENTS.

Use 4 sequential task agents with context handoffs:
1. Agent 1: First task category
2. Agent 2: Second task category
3. Agent 3: Third task category
4. Agent 4: Fourth task category

After implementation, run verification:
- `ruff check src/` passes
- `python -c "from src.<module> import <Class>; print('OK')"` for new code
- Report any regressions

**REQUIRED: Update docs after execution completes:**
1. Update CLEANUP_PLAN.md - mark completed items in tables
2. Update CLEANUP_TASKS.md - mark tasks as ✅ complete
3. If all items in a category done, add summary to COMPLETION.md

Per CLAUDE.md: delete don't adapt, verify compiles before committing.
