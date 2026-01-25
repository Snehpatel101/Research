**Reads:** DIRECTION.md, CLEANUP_PLAN.md, CLEANUP_TASKS.md, COMPLETION.md

**Tier:** Heavy (6 agents, full phase scope)

Spawn 6 parallel subagents to analyze $ARGUMENTS:
- `codebase-analyzer` (Opus): structure, dead code, performance
- `code-reviewer` (Sonnet): CLAUDE.md violations
- `contract-verifier` (Sonnet): type/schema issues
- `integration-checker` (Sonnet): import/dep problems
- Explore (very thorough): ML pipeline improvements
- Explore (very thorough): architecture patterns

**Output:** Synthesized, prioritized findings.

**Updates:** Use `doc-updater` to update CLEANUP_PLAN.md and CLEANUP_TASKS.md with implementation plan.
