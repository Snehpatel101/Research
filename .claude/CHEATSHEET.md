# ML Factory Commands

## Quick Matrix

| Task | Light | Medium | Heavy |
|------|-------|--------|-------|
| **Analyze** | `/analysis-targeted(1c)` | `/analysis-optimization(1b)` | `/analysis-full(1a)` |
| **Verify** | `/verify-claim(2a)` | `/verify-batch(2b)` | `/verify-contracts(2c)` |
| **Execute** | `/execute-surgical(4c)` | `/execute-standard(4a)` | `/execute-large(4b)` |
| **Docs** | `/docs-tasks(3a)` | `/docs-full(3b)` | `/docs-final(6a)` |
| **Check** | `/check-standard(5a)` | `/check-deep(5b)` | `/check-behavior(5c)` |

## When to Use Each Tier

| Tier | Scope | Agents | Example |
|------|-------|--------|---------|
| **Light** | 1 file, 1 fix | 1 | Fix typo, verify single claim |
| **Medium** | 1 task, 2-5 files | 2-4 | Standard task, batch verify |
| **Heavy** | 1 phase, 10+ files | 5-6 | Full audit, phase execution |

## Subagents

| Agent | Model | Purpose |
|-------|-------|---------|
| `codebase-analyzer` | Opus | Dead code, performance, architecture |
| `code-reviewer` | Sonnet | CLAUDE.md standards compliance |
| `contract-verifier` | Sonnet | Type/schema validation |
| `integration-checker` | Sonnet | Import/dependency analysis |
| `doc-updater` | Sonnet | Root doc updates |

## Workflow Patterns

### Standard Flow
```
Check COMPLETION.md → Analyze → Verify → Execute → Update docs
```

### Quick Fix
```
/execute-surgical(4c) [description]
```

### Phase Work
```
/analysis-full(1a) [phase]
/execute-large(4b) [phase]
/docs-final(6a) [phase]
```

## Project Rules

| Rule | Why |
|------|-----|
| Check COMPLETION.md first | Many claims already disproven |
| PLAN + TASKS sync together | They're mirrors |
| Delete, don't adapt | No compatibility layers |
| Verify before delete | Use `/verify-claim` first |

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Heavy command for 1 fix | `/execute-surgical(4c)` |
| Update PLAN without TASKS | Always sync both |
| Skip COMPLETION.md | Check first, avoid rework |
