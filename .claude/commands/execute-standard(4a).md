Read CLEANUP_TASKS.md for task details on $ARGUMENTS.

Execute changes, then verify:
- `ruff check src/` passes
- `python -c "from src.<module> import <Class>; print('OK')"` for new code

Use doc-updater subagent to:
- Mark completed items in CLEANUP_PLAN.md and CLEANUP_TASKS.md
- If category done, add summary to COMPLETION.md

Per CLAUDE.md: delete don't adapt, verify compiles.
