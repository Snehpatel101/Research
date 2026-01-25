Read DIRECTION.md and CLEANUP_PLAN.md to understand current compliance status.

Spawn 3 parallel subagents:
- codebase-analyzer: dead code, performance bottlenecks, O(n²) patterns
- code-reviewer: CLAUDE.md compliance, encapsulation violations
- Explore (medium): redundancy, unused abstractions

Rank findings by: impact × effort. Return file:line locations. Skip "NOT Doing" items.
