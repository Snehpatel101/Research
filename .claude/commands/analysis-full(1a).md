Read DIRECTION.md (architecture), CLEANUP_PLAN.md (current phase), CLEANUP_TASKS.md (pending work), and COMPLETION.md (what's already done/disproven).

Then spawn 6 parallel subagents to analyze $ARGUMENTS:
- codebase-analyzer: structure, dead code, performance
- code-reviewer: CLAUDE.md violations
- contract-verifier: type/schema issues
- integration-checker: import/dep problems
- Explore (very thorough): ML pipeline improvements
- Explore (very thorough): architecture patterns

Synthesize into prioritized findings. Use doc-updater subagent to update CLEANUP_PLAN.md and CLEANUP_TASKS.md with implementation plan.
