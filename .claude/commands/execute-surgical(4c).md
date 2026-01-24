Read CLEANUP_TASKS.md for the specific fix details on $ARGUMENTS.

Use 2 sequential task agents:
1. Implementation: Execute the targeted change (delete, don't adapt)
2. Verification: Run `ruff check .` and `pyright topstepx_backend/`

Keep scope narrow. If unexpected dependencies appear, stop and report rather than expanding scope.

Per CLAUDE.md: small diffs, verify compiles, commit early.
