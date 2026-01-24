# Agent Workflow Prompts - User Guide

**What This Is:** Copy/paste prompts for orchestrating Claude's multi-agent analysis, cleanup, and verification workflows.  

**Root Documentation ():**
- `CLAUDE.md` — Your project's main context file
- `DIRECTION_TEMPLATE.md` — Architecture overview and refactoring trajectory
- `COMPLETION_TEMPLATE.md` — Archive of completed phases and verification results
- `CLEANUP_TASKS_TEMPLATE.md` — Active work and task tracking
- `CLEANUP_PLAN_TEMPLATE.md` — Phase planning and execution roadmaps

## Quick Start

**Before using any prompt:**
1. Make sure your root documentation templates are in place
2. Clear context if starting fresh (`/clear` in Claude Code, or new chat)
3. Replace bracketed placeholders with your actual module, directory, or codebase names


## Phase 1: Initial Analysis

> **When to use:** After `/clear` or starting a new session. This is your starting point.

### 1A: Full Codebase Analysis
*Use when you want a broad sweep of the entire codebase.*

```
Read all root documentation , then spawn 4 parallel specialized task agents to analyze the codebase for:
- Codebase organization and structure
- Dead code and cleanup opportunities  
- Performance optimization targets
- Architecture improvements

Synthesize findings into a unified report.
```

### 1B: Optimization-Focused Analysis
*Use when you specifically want improvement recommendations.*

```
Read all root documentation , then use 3-5 parallel specialized task agents to identify:
- Code quality improvements
- Architectural optimizations
- Redundancy elimination opportunities
- Performance bottlenecks

Present findings ranked by impact and effort.
```

### 1C: Targeted Module Deep-Dive
*Use when you want to focus on a specific module and have Claude ask you clarifying questions.*

```
Read all root documentation , then spawn 2-3 specialized task agents to analyze the [module/subsystem] module:
- Map all dependencies and consumers
- Identify behavioral contracts
- Document current vs intended behavior

After analysis, ask clarifying questions about expected behaviors and goals for this module.
```

---

## Phase 2: Verification & Validation

> **When to use:** When Claude's recommendations look questionable, too aggressive, or you're not sure if something is actually dead code.

### 2A: Single Claim Verification
*Use when one specific thing seems off.*

```
Verify that claim. [Add your context here, e.g., "This function handles backfill operations and should run on startup to fill gaps in TimescaleDB."]
```

### 2B: Batch Verification
*Use when Claude flagged a lot of stuff and you want it double-checked before proceeding.*

```
Use 3-4 parallel specialized task agents to verify all proposed deletions and modifications:
- Trace each item's usage across the codebase
- Confirm dead code status with call graph analysis
- Validate that no runtime paths depend on flagged items
- Use all available tools (grep, AST analysis, test coverage)

Return a verified/disproven status for each item with evidence.
```

### 2C: Behavioral Contract Verification
*Use when you need Claude to confirm assumptions before it starts changing things.*

```
Before proceeding, verify these behavioral assumptions:
1. [Your assumption 1]
2. [Your assumption 2]
3. [Your assumption 3]

Use static analysis and runtime tracing where possible. Flag any assumptions that cannot be verified.
```

---

## Phase 3: Documentation Update

> **When to use:** After analysis and verification are done, before you start executing changes.

### 3A: Update Task Documentation
*Use after you've discussed findings and want the task list updated.*

```
Update the task list with verified findings:
- Remove disproven items
- Add newly discovered issues
- Adjust priorities based on investigation results
- Document any deferred items with rationale
```

### 3B: Full Documentation Refresh
*Use when you want all root docs brought up to date.*

```
Update all root documentation  to reflect:
- Current codebase state
- Completed and pending phases
- Verified cleanup targets
- Architecture decisions and rationale

Ensure consistency across all documentation files.
```

---

## Phase 4: Execution

> **When to use:** Documentation is current, tasks are verified, you're ready to make changes.

### 4A: Standard Execution (Moderate Scope)
*Use for typical cleanup work — a handful of related tasks.*

```
Use 4 sequential specialized task agents with mandatory context handoffs:
1. Specialized Task Agent 1: [First task category]
2. Specialized Task Agent 2: [Second task category]  
3. Specialized Task Agent 3: [Third task category]
4. Specialized Task Agent 4: [Fourth task category]

After all implementation, spawn a specialized verification subagent to:
- Start frontend and backend
- Review startup logs for errors
- Perform basic functional tests
- Report any regressions
```

### 4B: Large-Scale Execution (Extensive Changes)
*Use for major refactors or when changes touch many parts of the system.*

```
Orchestrate 7 sequential specialized task agents with mandatory context handoffs for PHAS

Implementation (Specialized Task Agents 1-6):
- Each specialized task agent handles one task category
- Each passes full context to the next
- After each task agent, spawn a specialized verification subagent to validate that step's changes

Post-Implementation:
- Specialized Task Agent 7: Review remaining issues and risks
- Final specialized verification subagent: Boot frontend + backend, monitor logs, perform live tests
- Parallel specialized review subagents: Cross-check the final result

Report any inconsistencies or failures immediately.
```

### 4C: Surgical Execution (Targeted Fixes)
*Use for small, focused changes.*

```
Use 2 sequential specialized task agents for targeted fixes:
1. Specialized implementation task agent: Execute the specific changes in [file/area]
2. Specialized verification task agent: Validate changes don't break existing behavior

Keep scope narrow. Escalate if unexpected dependencies are discovered.
```

---

## Phase 5: Post-Execution Verification

> **When to use:** After execution completes, or if something seems broken.

### 5A: Standard Post-Execution Check
*Use after any execution phase to confirm nothing broke.*

```
Use parallel specialized task agents to verify:
- All modified files function correctly
- No regressions in existing behavior
- Frontend and backend start without errors
- Logs show no new warnings or errors

Report: "Are we missing anything?" with risks and recommended next actions.
```

### 5B: Deep Verification (After Large Changes)
*Use after major refactors or when you want comprehensive validation.*

```
Run parallel specialized subagents for comprehensive verification:

1. Specialized Code Review Subagent:
   - Review all changes from the last execution
   - Check for incomplete migrations
   - Identify any orphaned code

2. Specialized Contract Verification Subagent:
   - Validate API schemas haven't drifted
   - Confirm type definitions are consistent
   - Check environment variables and config

3. Specialized Integration Subagent:
   - Verify no dead paths or broken wiring
   - Confirm all imports resolve
   - Check dependency graph integrity

4. Specialized Runtime Subagent:
   - Start frontend and backend
   - Monitor logs during startup
   - Perform smoke tests on critical paths

Return a consolidated report: status, risks, and recommended next actions.
```

### 5C: Targeted Behavior Verification
*Use when you need to confirm a specific feature still works.*

```
Verify that [feature/behavior] still works correctly:
- Trace the execution path
- Confirm expected inputs produce expected outputs
- Check edge cases: [edge case 1], [edge case 2]
- Report pass/fail with evidence
```

---

## Workflow Cheatsheet

### Minimal Workflow (Small Task)
```
1. Phase 1C → Targeted analysis
2. Phase 3A → Update tasks
3. Phase 4C → Surgical execution
4. Phase 5A → Standard verification
```

### Standard Workflow (Typical Cleanup)
```
1. Phase 1A → Full analysis
2. Phase 2B → Batch verification (if Claude's suggestions look fishy)
3. Phase 3B → Full documentation update
4. Phase 4A → Standard execution
5. Phase 5A → Standard verification
```

### Comprehensive Workflow (Major Refactor)
```
1. Phase 1A → Full analysis
2. Phase 2B → Batch verification
3. Phase 3B → Full documentation update
4. Phase 4B → Large-scale execution
5. Phase 5B → Deep verification
6. Phase 3B → Final documentation update
```

---

## Tips & Best Practices

### Scaling Agents

| Scope | Task Agents | Verification Pattern |
|-------|-------------|---------------------|
| Small (single module) | 2-3 | Single verification subagent |
| Medium (multiple modules) | 3-4 | Step-by-step verification subagents |
| Large (system-wide) | 5-7 | Per-step + final + parallel review subagents |

### When to Pause and Verify (Phase 2)

Use Phase 2 prompts when:
- Claude proposes deleting code you don't recognize
- The cleanup list seems suspiciously long
- Claude sounds overconfident ("this is definitely unused")
- Changes affect core infrastructure (DI, config, startup, bootstrap)

### Red Flags — Always Double-Check

- Deleting files without clear dead-code evidence
- Modifying shared utilities or base classes
- Changes to initialization or startup sequences
- Touching code with no test coverage

### General Tips

- **Always start with root documentation ** — Claude needs context from your CLAUDE.md + 
- **Trust but verify** — Claude's analysis is a starting point, not gospel
- **Check behavior yourself** — After Phase 4, actually test the frontend/backend manually
- **When in doubt, verify before deleting** — It's easier to keep code than resurrect it
- **Document everything** — Future you will thank present you