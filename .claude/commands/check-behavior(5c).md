**Tier:** Heavy (2 agents, behavioral verification)

Run 2 parallel task agents to verify $ARGUMENTS still works:

**Agent 1 - Execution Tracing:**
- Trace code path entry to exit
- Confirm inputs to outputs
- Check for side effects

**Agent 2 - Edge Cases:**
- Null/empty/boundary inputs
- Error conditions

**Output:** `PASS` | `FAIL` with file:line evidence.
