**Tier:** Heavy (team orchestration)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Route to the appropriate team command based on the task type.

**Arguments:** $ARGUMENTS

**Routing Logic:**

Analyze the user's request and route to the best team command:

| If the request involves... | Route to |
|---------------------------|----------|
| Planning, designing, exploring approaches | `/team-plan` |
| Writing code, implementing features, fixing bugs | `/team-implement` |
| Investigating topics, comparing approaches | `/team-research` |
| Reviewing code quality, security, performance | `/team-review` |
| Writing or running tests | `/team-test` |

**Process:**

1. Analyze `$ARGUMENTS` to determine the task type
2. Tell the user which team command you're routing to and why
3. Execute the appropriate team command with the original arguments

**Examples:**
- `"add retry logic to the API client"` → `/team-implement`
- `"investigate walk-forward validation approaches"` → `/team-research`
- `"review the new feature selection code"` → `/team-review`
- `"plan the next optimization phase"` → `/team-plan`
- `"write tests for the adapter layer"` → `/team-test`

If the request doesn't clearly fit one category, ask the user which team workflow they prefer.
