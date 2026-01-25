**Reads:** DIRECTION.md (architectural contracts)

**Tier:** Heavy (3 agents sequential, deep contract analysis)

Run 3 subagents sequentially:
1. `contract-verifier` (Sonnet): static analysis (signatures, types, interfaces)
2. `code-reviewer` (Sonnet): behavioral docs (expected vs actual, side effects)
3. `integration-checker` (Sonnet): runtime validation (execution paths, unverifiable assumptions)

**Output:** Consolidated contract compliance report for $ARGUMENTS.
