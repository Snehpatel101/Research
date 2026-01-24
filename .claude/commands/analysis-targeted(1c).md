Read CLEANUP_TASKS.md and COMPLETION.md for context on $ARGUMENTS.

Spawn 2-3 task agents to analyze:
- Dependencies: all imports, consumers, callers of this code
- Contracts: function signatures, return types, side effects
- Behavior: current vs documented vs intended behavior

After analysis, ask clarifying questions. Many past investigations revealed claims were wrong.
