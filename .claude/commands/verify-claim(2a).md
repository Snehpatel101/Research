**Reads:** COMPLETION.md (first - claim may already be verified/disproven)

**Tier:** Light (1 agent, single claim)

Use `codebase-analyzer` (Opus) to verify $ARGUMENTS:
- Trace all usages with grep
- Check for dynamic access (getattr, **kwargs, config-driven)
- Confirm no runtime paths depend on item

**Output:** `VERIFIED` | `DISPROVEN` | `INCONCLUSIVE` with file:line evidence.
