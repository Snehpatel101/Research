**Tier:** Heavy (1-3 agents sequential: analyze → write → run → report)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Plan tests, write them, run them, and report results.

**Arguments:** $ARGUMENTS — what to test (module, feature, or scope)

**Team:** Testing
- `team-tester` in plan mode (Sonnet) — analyzes code, produces test plan
- `team-tester` in write mode (Sonnet) — writes test files
- `team-tester` in run mode (Sonnet) — executes tests and reports

**Process:**

1. **Create team** via TeamCreate with name `test-<short-scope>`

2. **Phase 1 — Analyze:**
   - Create task: "Analyze $ARGUMENTS and produce test plan"
   - Spawn team-tester in plan mode
   - Tester reads source code, identifies coverage gaps, edge cases
   - Produces structured test plan with proposed test names and locations

3. **Phase 2 — Write:**
   - Create task: "Write tests per plan" — blockedBy Phase 1
   - Spawn team-tester in write mode
   - Writes test files following pytest conventions
   - Runs `ruff check` and `black` on new test files
   - Reports which test files were created

4. **Phase 3 — Run:**
   - Create task: "Run tests and report results" — blockedBy Phase 2
   - Spawn team-tester in run mode
   - Executes: `python -m pytest tests/path/ -v`
   - Reports pass/fail with failure details

5. **Report** to user:
   - Test plan summary
   - Files created
   - Pass/fail results
   - Recommended next steps for failures

6. **Shutdown** team when complete

**ML Factory Test Priorities:**
1. Data leakage (purge/embargo in CV)
2. Lookahead bias (shift(1) in MTF)
3. Reproducibility (same config = same output)
4. Contract compliance (models respect contracts)
5. Pipeline integrity (12 stages, correct shapes)
