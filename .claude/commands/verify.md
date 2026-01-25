Run 4 parallel subagents to verify $ARGUMENTS:
- code-reviewer: CLAUDE.md standards
- contract-verifier: types/schemas
- integration-checker: imports/deps
- codebase-analyzer: dead code check

Return: ✅ PASS | ❌ FAIL with evidence.

**For specialized verification, use:**
- `/verify-claim(2a)` - single claim check
- `/verify-batch(2b)` - multiple items
- `/verify-contracts(2c)` - deep contract analysis
