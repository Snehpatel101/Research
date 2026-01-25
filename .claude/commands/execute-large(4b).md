**Reads:** CLEANUP_PLAN.md (roadmap), CLEANUP_TASKS.md (details)

**Tier:** Heavy (multi-phase implementation)

Execute $ARGUMENTS in phases with verification between each:

**Phases 1-3: Implementation**
- Execute changes per CLEANUP_TASKS.md
- Run `ruff check src/` after each major change
- Verify imports work

**Phase 4: Validation**
- Run full validation: ruff, imports, tests if applicable

**Phase 5: Documentation**
Use `doc-updater` (Sonnet) to:
- Update CLEANUP_PLAN.md - mark completed
- Update CLEANUP_TASKS.md - mark tasks done
- Add phase summary to COMPLETION.md
- Remove completed items from PLAN and TASKS

Per CLAUDE.md: delete don't adapt, small diffs, verify compiles.
