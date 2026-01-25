Run 4 parallel subagents to verify $ARGUMENTS:
- `code-reviewer` (Sonnet): CLAUDE.md standards
- `contract-verifier` (Sonnet): types/schemas
- `integration-checker` (Sonnet): imports/deps
- `codebase-analyzer` (Opus): dead code check

**Output:** `PASS` | `FAIL` with evidence.

**Tiered alternatives:**
| Tier | Command | Use When |
|------|---------|----------|
| Light | `/verify-claim(2a)` | Single claim check |
| Medium | `/verify-batch(2b)` | Multiple items |
| Heavy | `/verify-contracts(2c)` | Deep contract analysis |
