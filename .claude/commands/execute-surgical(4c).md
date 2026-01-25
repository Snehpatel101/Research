**Reads:** CLEANUP_TASKS.md (specific fix details)

**Tier:** Light (1 agent, single fix)

Execute minimal change for $ARGUMENTS:
- Single focused fix only (delete, don't adapt)
- Stop if unexpected dependencies appear - report rather than expand scope
- Run `ruff check` on changed files only

**Updates:** Use `doc-updater` (Sonnet) to mark task complete in CLEANUP_PLAN.md and CLEANUP_TASKS.md.
