**First check COMPLETION.md** for previously disproven claims.

Spawn 3-4 parallel task agents to verify proposed changes for $ARGUMENTS:
- Trace each item's usage across the codebase
- Confirm dead code status with grep + import analysis
- Check for dynamic access (getattr, **kwargs, config-driven)
- Validate no runtime paths depend on flagged items

Return verified/disproven status per item with file:line evidence.
