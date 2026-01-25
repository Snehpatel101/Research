**First check COMPLETION.md** for previously disproven claims.

Spawn 3 parallel subagents to verify $ARGUMENTS:
- codebase-analyzer: trace usage, check dynamic access
- integration-checker: confirm no runtime paths depend on items
- Explore (quick): verify no config-driven or public API usage

Return table: Item | Status | Evidence
