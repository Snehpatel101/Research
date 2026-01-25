**Reads:** CLEANUP_TASKS.md (task details)

**Tier:** Medium (1 agent, standard task)

Execute changes for $ARGUMENTS, then verify:
- `ruff check src/` passes
- `python -c "from src.<module> import <Class>; print('OK')"` for new code

**Updates:** Use `doc-updater` (Sonnet) to:
- Mark completed items in CLEANUP_PLAN.md and CLEANUP_TASKS.md
- If category done, add summary to COMPLETION.md

Per CLAUDE.md: delete don't adapt, verify compiles.
