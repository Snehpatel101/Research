**Tier:** Heavy (2 agents sequential)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Plan an implementation approach, then update documentation.

**Arguments:** $ARGUMENTS — the topic or feature to plan

**Team:** Planning & Documentation
- `team-planner` (Opus) — explores codebase, designs implementation plan
- `team-documenter` (Sonnet) — updates root docs with the plan

**Process:**

1. **Create team** via TeamCreate with name `plan-<short-topic>`
2. **Create tasks:**
   - Task 1: "Explore codebase and design plan for: $ARGUMENTS" → assign to team-planner
   - Task 2: "Update CLEANUP_PLAN and CLEANUP_TASKS with new plan" → assign to team-documenter, blockedBy Task 1
3. **Spawn team-planner** (Task tool, subagent_type=general-purpose, agent config=team-planner)
   - Reads DIRECTION.md, CLEANUP_PLAN.md, COMPLETION.md for context
   - Explores relevant code paths
   - Produces structured plan with file:line references
   - Messages team lead with the plan
4. **When planner completes, spawn team-documenter**
   - Reads the planner's output from task description
   - Updates CLEANUP_PLAN.md and CLEANUP_TASKS.md with the new phase/tasks
   - Messages team lead when done
5. **Synthesize** — Present the plan to the user for approval
6. **Shutdown** team when complete

**Output:** Implementation plan with file:line references, updated in root docs, ready for user approval.
