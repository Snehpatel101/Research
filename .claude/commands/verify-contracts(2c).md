Read DIRECTION.md for architectural contracts.

Run 3 subagents sequentially:
- contract-verifier: static analysis (signatures, types, interfaces)
- code-reviewer: behavioral docs (expected vs actual, side effects)
- integration-checker: runtime validation (execution paths, unverifiable assumptions)

Return consolidated contract compliance report.
