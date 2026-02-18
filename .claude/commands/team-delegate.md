**Tier:** Utility (no agents spawned)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Task assignment dashboard — view, assign, and rebalance tasks.

**Arguments:** $ARGUMENTS
- `assign <task-id> <agent-name>` — Assign a specific task to an agent
- `rebalance` — Redistribute pending tasks evenly among idle agents
- (no args) — Show assignment dashboard

**Dashboard Mode (no args):**

1. Run TaskList to get all tasks
2. Read team config to get members
3. Display:

```
## Task Dashboard

### Unassigned Tasks
| ID | Subject | Blocked By |
|----|---------|------------|
| [id] | [subject] | [blockedBy] |

### Agent Workload
| Agent | In Progress | Completed | Available |
|-------|-------------|-----------|-----------|
| [name] | N | N | yes/no |

### Suggested Assignments
- Task [id] → [agent-name] (reason: [matches agent specialty])
```

**Assign Mode:** Use TaskUpdate to set the owner, then message the agent about their new task.

**Rebalance Mode:**
1. Find all pending, unblocked, unassigned tasks
2. Find all agents with no in-progress tasks
3. Assign tasks round-robin, matching agent specialties where possible:
   - Code changes → team-implementer
   - Documentation → team-documenter
   - Investigation → team-researcher or team-debugger
   - Review → team-reviewer
   - Tests → team-tester
4. Message each agent about their new assignments
