**Reads:** COMPLETION.md (first - items may already be verified/disproven)

**Tier:** Medium (3 agents, multiple items)

Spawn 3 parallel subagents to verify $ARGUMENTS:
- `codebase-analyzer` (Opus): trace usage, check dynamic access
- `integration-checker` (Sonnet): confirm no runtime paths depend on items
- Explore (quick): verify no config-driven or public API usage

**Output:** Table with Item | Status | Evidence
