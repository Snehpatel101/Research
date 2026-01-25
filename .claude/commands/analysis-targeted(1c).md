**Reads:** CLEANUP_TASKS.md, COMPLETION.md

**Tier:** Light (1 agent, focused investigation)

Use `codebase-analyzer` (Opus) to analyze $ARGUMENTS:
- Dependencies: imports, consumers, callers
- Contracts: signatures, return types, side effects
- Behavior: current vs documented vs intended

**Output:** Focused analysis with file:line evidence.

**Note:** Many past claims were disproven - be skeptical. Ask clarifying questions if needed.
