Use 2 parallel task agents to verify $ARGUMENTS still works:

**Agent 1 - Execution Tracing:**
- Trace code path entry → exit
- Confirm inputs → outputs
- Check for side effects

**Agent 2 - Edge Cases:**
- Null/empty/boundary inputs
- Error conditions

Return: ✅ PASS | ❌ FAIL with file:line evidence.
