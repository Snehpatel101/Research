Read CLEANUP_TASKS.md for the specific fix details on $ARGUMENTS.

Use 2 sequential task agents:
1. Implementation: Execute the targeted change (delete, don't adapt)
2. Verification: Run `ruff check src/` and verify imports work

Keep scope narrow. If unexpected dependencies appear, stop and report rather than expanding scope.

**REQUIRED: Update docs after execution completes:**
1. Update CLEANUP_PLAN.md - mark item complete in table
2. Update CLEANUP_TASKS.md - mark task as ✅ complete

Per CLAUDE.md: small diffs, verify compiles, commit early.
