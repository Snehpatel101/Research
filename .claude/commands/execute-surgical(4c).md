Read CLEANUP_TASKS.md for the specific fix details on $ARGUMENTS.

Execute minimal change:
- Single focused fix only (delete, don't adapt)
- Stop if unexpected dependencies appear - report rather than expand scope
- Run `ruff check` on changed files only

Use doc-updater subagent to mark task complete in CLEANUP_PLAN.md and CLEANUP_TASKS.md.
