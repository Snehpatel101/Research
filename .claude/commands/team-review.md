**Tier:** Heavy (2-4 agents, parallel dimensional review → consolidate)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Review code along multiple quality dimensions in parallel.

**Arguments:** $ARGUMENTS — files or scope to review

**Team:** Code Review
- 2-4 `team-reviewer` (Opus) — each reviews one dimension
- Team lead consolidates findings

**Available Dimensions:**
- **security** — data leakage, unsafe pickle, injection
- **performance** — O(n^2), copies, caching, blocking I/O
- **architecture** — canonical imports, duplicates, cycles
- **testing** — coverage gaps, flaky tests
- **ml-integrity** — lookahead, leakage, reproducibility

**Process:**

1. **Create team** via TeamCreate with name `review-<short-scope>`

2. **Select dimensions** based on the scope:
   - For model code: ml-integrity + performance + architecture
   - For pipeline code: security + architecture + performance
   - For new features: architecture + testing + security
   - For full review: all applicable dimensions (max 4)

3. **Create tasks** — one per dimension, all independent (no blockedBy)
   - Each task specifies: dimension, files to review, what to look for

4. **Spawn 2-4 team-reviewers in parallel:**
   - Each reviewer focuses on ONE dimension only
   - Each produces structured findings with severity levels
   - Each reports to team lead via message

5. **Consolidate** — After all reviewers complete:
   - Merge findings, deduplicate
   - Sort by severity (CRITICAL → HIGH → MEDIUM → LOW)
   - Present unified review report

6. **Shutdown** team when complete

**Output Format:**
```
## Code Review: [scope]

### Critical (must fix)
- [finding] — `file:line`

### High (should fix)
- [finding] — `file:line`

### Medium (consider fixing)
- [finding] — `file:line`

### Summary
- Dimensions reviewed: N
- Total findings: N (C critical, H high, M medium, L low)
- Overall: PASS | NEEDS WORK | FAIL
```
