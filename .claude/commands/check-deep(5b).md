Run 4 parallel task agents for comprehensive verification of $ARGUMENTS:

1. Code Review Agent:
   - Check all changes against CLAUDE.md standards
   - Verify no adapters/compatibility layers added
   - Confirm internal state is encapsulated

2. Contract Verification Agent:
   - API schemas consistent
   - Type definitions match implementations
   - No raw SQL in services

3. Integration Agent:
   - All imports resolve
   - No circular dependencies
   - No orphaned code

4. Runtime Agent:
   - Backend starts, check logs for errors
   - Frontend builds and typechecks
   - Run validation checklist

Return consolidated report: status, violations found, next actions.
