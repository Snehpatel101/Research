**Tier:** Utility (no agents spawned)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Gracefully shut down the current team.

**Arguments:** $ARGUMENTS
- `--force` — Skip confirmation and shut down immediately
- (no args) — Confirm with user before shutting down

**Process:**

1. Run TaskList to check for in-progress tasks
2. If there are in-progress tasks and `--force` was NOT given:
   - Show the in-progress tasks
   - Ask user: "There are N tasks still in progress. Shut down anyway?"
   - If user says no, abort
3. For each active teammate:
   - Send a `shutdown_request` via SendMessage
   - Wait for acknowledgment
4. After all teammates have shut down, call TeamDelete to clean up
5. Report: "Team shut down. N tasks completed, M tasks abandoned."

**If `--force` is given:** Skip step 2 confirmation and proceed directly to shutdown.
