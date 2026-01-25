**Reads:** DIRECTION.md, CLEANUP_PLAN.md

**Tier:** Medium (3 agents, optimization focus)

Spawn 3 parallel subagents to analyze $ARGUMENTS:
- `codebase-analyzer` (Opus): dead code, performance bottlenecks, O(n^2) patterns
- `code-reviewer` (Sonnet): CLAUDE.md compliance, encapsulation violations
- Explore (medium): redundancy, unused abstractions

**Output:** Findings ranked by impact x effort with file:line locations.

Skip items marked "NOT Doing" in CLEANUP_PLAN.md.
