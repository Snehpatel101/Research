**Tier:** Heavy (3-5 agents, plan → parallel implement → verify)

**Pre-flight:** Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set:
```bash
echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
```
If not set, tell the user: "Team features require `export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

**Action:** Plan, implement in parallel, then verify the implementation.

**Arguments:** $ARGUMENTS — what to implement

**Team:** Implementation
- `team-lead` (Opus) — orchestrates, owns shared files
- `team-planner` (Opus) — designs the plan (phase 1)
- 1-3 `team-implementer` (Sonnet) — write code in parallel (phase 2)
- `team-reviewer` (Opus) — verifies quality (phase 3)

**Process:**

1. **Create team** via TeamCreate with name `impl-<short-topic>`

2. **Phase 1 — Plan:**
   - Create task: "Design implementation plan for: $ARGUMENTS"
   - Spawn team-planner to explore codebase and produce plan
   - Wait for plan, present to user for approval before proceeding

3. **Phase 2 — Implement (parallel):**
   - Team lead decomposes plan into file-disjoint subtasks
   - Assign **exclusive file ownership** — no two implementers touch the same file
   - Create tasks with clear file boundaries and acceptance criteria
   - Spawn 1-3 team-implementers in parallel
   - Each implementer: edit files → ruff check → black → verify imports
   - Team lead owns shared files (configs, __init__.py) if needed

4. **Phase 3 — Verify:**
   - Create task: "Review implementation for: $ARGUMENTS"
   - Spawn team-reviewer with dimension=architecture
   - Reviewer checks: imports correct, no duplicates, contracts respected
   - Run `ruff check src/` and `black --check src/` on all changed files

5. **Report** results to user with summary of changes
6. **Shutdown** team when complete

**File Ownership Example:**
```
Implementer A: src/models/lstm.py, src/models/gru.py
Implementer B: src/data/pipeline.py, src/data/features.py
Team Lead: src/core/types.py (shared), src/__init__.py
```
