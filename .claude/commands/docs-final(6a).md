**Reads:** DIRECTION.md, CLEANUP_PLAN.md, CLEANUP_TASKS.md, COMPLETION.md

**Tier:** Heavy (1 agent, phase close-out)

Use `doc-updater` (Sonnet) to close out $ARGUMENTS:
1. Archive to COMPLETION.md: summary, lines changed, lessons learned
2. Remove completed items from CLEANUP_PLAN.md
3. Remove completed tasks from CLEANUP_TASKS.md
4. Update DIRECTION.md metrics if architecture changed

**Updates:** All 4 root docs. Verify cross-document consistency.

**Note:** This command closes out a phase - use after all work is verified complete.
