Read CLEANUP_PLAN.md (roadmap) and CLEANUP_TASKS.md (details) for $ARGUMENTS.

Execute in phases with verification between each:

**Phase 1-3:** Implementation
- Execute changes per CLEANUP_TASKS.md
- Run `ruff check src/` after each major change
- Verify imports work

**Phase 4:** Validation
- Run full validation: ruff, imports, tests if applicable

**Phase 5:** Documentation (use doc-updater subagent)
- Update CLEANUP_PLAN.md - mark completed
- Update CLEANUP_TASKS.md - mark tasks ✅
- Add phase summary to COMPLETION.md
- Remove completed items from PLAN and TASKS

Per CLAUDE.md: delete don't adapt, small diffs, verify compiles.
