Run 4 parallel task agents for comprehensive verification of $ARGUMENTS:

- code-reviewer: Check CLAUDE.md standards, no adapters, encapsulation
- contract-verifier: API schemas, type definitions match implementations
- integration-checker: All imports resolve, no circular deps, no orphans
- Explore (medium): Runtime validation, check for errors

Return consolidated report: ✅ PASS | ❌ FAIL per category.
