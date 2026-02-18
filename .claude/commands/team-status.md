**Tier:** Utility (no agents spawned)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Display current team status.

1. Read the team config to discover members:
   - `~/.claude/teams/*/config.json`
2. Run TaskList to get all tasks and their status
3. Display a summary:

```
## Team Status

### Members
| Name | Type | Status |
|------|------|--------|
| [name] | [agentType] | active/idle |

### Tasks
| ID | Subject | Owner | Status | Blocked By |
|----|---------|-------|--------|------------|
| [id] | [subject] | [owner] | [status] | [blockedBy] |

### Progress
- Total: N | Completed: N | In Progress: N | Pending: N | Blocked: N
```

If no team is active, report: "No active team found. Use `/team-spawn` to create one."
