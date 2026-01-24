Read DIRECTION.md and CLEANUP_PLAN.md to understand current compliance status.

Spawn 3-5 parallel task agents to identify improvements in $ARGUMENTS:
- Code quality: encapsulation violations, raw SQL in services, scattered config
- Architecture: service hierarchy compliance, repository pattern violations
- Redundancy: duplicate logic, unused abstractions, dead patterns
- Performance: blocking I/O, missing caching, O(n²) algorithms

Rank by impact × effort. Reference file:line locations. Skip anything in "NOT Doing" section.
