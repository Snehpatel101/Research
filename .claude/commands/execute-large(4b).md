Read CLEANUP_PLAN.md (roadmap) and CLEANUP_TASKS.md (details) for $ARGUMENTS.

Orchestrate 7 sequential task agents with context handoffs:

Implementation (Agents 1-6):
- Each handles one task category from CLEANUP_TASKS.md
- Run `ruff check src/` after each agent
- Pass full context to next agent

Post-Implementation (Agent 7):
- Review remaining issues
- Run full validation checklist:
  - `ruff check src/` passes
  - `python -c "from src.<module> import <Class>; print('OK')"` for new code
  - test the actual pipeline.

**REQUIRED: Update docs after execution completes:**
1. Update CLEANUP_PLAN.md - mark completed items in tables
2. Update CLEANUP_TASKS.md - mark tasks as ✅ complete
3. Add phase summary to COMPLETION.md with:
   - Impact (lines added/removed)
   - Tasks completed
   - Files created/modified
   - Lessons learned

Per CLAUDE.md: delete don't adapt, small diffs, verify compiles.
