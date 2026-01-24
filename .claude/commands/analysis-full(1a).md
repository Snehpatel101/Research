Read DIRECTION.md (architecture), CLEANUP_PLAN.md (current phase), CLEANUP_TASKS.md (pending work), and COMPLETION.md (what's already done/disproven).

Then spawn 4 parallel task agents to analyze $ARGUMENTS:
- Structure: codebase organization, module boundaries
- Dead code: unused imports, unreachable paths, orphaned files
- Performance: O(n²) patterns, blocking calls, cache misses
- Architecture: pattern violations per CLAUDE.md standards

Synthesize into prioritized findings. Check COMPLETION.md - many past claims were disproven.
