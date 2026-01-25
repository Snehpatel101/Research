**Tier:** Medium (4 agents, comprehensive verification)

Run 4 parallel subagents to verify $ARGUMENTS:

- `code-reviewer` (Sonnet): Check CLAUDE.md standards, no adapters, encapsulation
- `contract-verifier` (Sonnet): API schemas, type definitions match implementations
- `integration-checker` (Sonnet): All imports resolve, no circular deps, no orphans
- Explore (medium): Runtime validation, check for errors

**Output:** Consolidated report with `PASS` | `FAIL` per category.
