**Tier:** Heavy (3-4 agents, parallel research → synthesize)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Research a topic from multiple angles in parallel, then synthesize findings.

**Arguments:** $ARGUMENTS — the research topic

**Team:** Research
- 2-3 `team-researcher` (Sonnet) — investigate from different angles
- `team-synthesizer` (Opus) — merge findings into unified summary

**Process:**

1. **Create team** via TeamCreate with name `research-<short-topic>`

2. **Decompose the topic** into 2-3 research angles. Examples:
   - For "walk-forward validation": (1) academic literature, (2) existing implementations in codebase, (3) Python library support
   - For "transformer optimization": (1) architecture papers, (2) training tricks, (3) our current implementation gaps
   - For "feature engineering": (1) financial domain features, (2) automated feature selection, (3) what competitors do

3. **Create tasks** — one per research angle, all independent (no blockedBy)

4. **Spawn 2-3 team-researchers in parallel:**
   - Each gets a specific angle and clear scope
   - Each searches both codebase AND web
   - Each reports structured findings with sources

5. **Create synthesis task** — blockedBy all research tasks

6. **When all researchers complete, spawn team-synthesizer:**
   - Reads all researcher findings from messages/tasks
   - Resolves conflicts between sources
   - Ranks recommendations by impact and feasibility
   - Produces unified summary with ML Factory applicability

7. **Present** synthesized findings to user
8. **Shutdown** team when complete

**Output:** Synthesized research report with ranked, actionable recommendations.
