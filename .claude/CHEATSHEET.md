# Claude Commands Cheatsheet

## Quick Commands (Entry Points)

| Command | Purpose | Depth |
|---------|---------|-------|
| `/analyze [topic]` | Quick codebase analysis | Light |
| `/verify [changes]` | 4-way parallel verification | Medium |
| `/execute [task]` | Execute + update docs | Medium |
| `/docs [changes]` | Update root docs | Light |
| `/check [area]` | Validation checklist | Light |

## Tiered Commands (Full Control)

### Analysis Tiers
| Command | Agents | Use When |
|---------|--------|----------|
| `/analysis-targeted(1c)` | 1 | Investigating specific item |
| `/analysis-optimization(1b)` | 3 | Finding improvements, ranked |
| `/analysis-full(1a)` | 6 | Comprehensive codebase audit |

### Verification Tiers
| Command | Agents | Use When |
|---------|--------|----------|
| `/verify-claim(2a)` | 1 | Single dead code claim |
| `/verify-batch(2b)` | 3 | Multiple items at once |
| `/verify-contracts(2c)` | 3 seq | Deep contract analysis |

### Execution Tiers
| Command | Scope | Use When |
|---------|-------|----------|
| `/execute-surgical(4c)` | Narrow | Single focused fix |
| `/execute-standard(4a)` | Normal | Standard task |
| `/execute-large(4b)` | Phase | Multi-step phase work |

### Documentation Tiers
| Command | Scope | Use When |
|---------|-------|----------|
| `/docs-tasks(3a)` | Tasks only | Updating task status |
| `/docs-full(3b)` | All 4 docs | Major changes |
| `/docs-final(6a)` | Archive | Closing out a phase |

### Check Tiers
| Command | Depth | Use When |
|---------|-------|----------|
| `/check-standard(5a)` | Basic | Pre-commit validation |
| `/check-deep(5b)` | 4-way | Comprehensive review |
| `/check-behavior(5c)` | Runtime | Execution path testing |

---

## Subagents (Context-Isolated)

| Agent | Model | Tools | Purpose |
|-------|-------|-------|---------|
| `codebase-analyzer` | Haiku | Read-only + Bash | Dead code, perf, arch |
| `code-reviewer` | Haiku | Read-only | CLAUDE.md standards |
| `contract-verifier` | Haiku | Read-only | Type/schema validation |
| `integration-checker` | Haiku | Read-only + Bash | Import/dep analysis |
| `doc-updater` | Sonnet | Read + Edit | Root doc updates |

**Built-in:**
| Agent | When to Use |
|-------|-------------|
| `Explore (quick/medium/very thorough)` | Codebase search |
| `Plan` | Research before planning |

---

## Context Efficiency

**Why subagents save context:**
```
Old: Commands embed instructions → all in main context
New: Commands invoke subagents → work in isolated context, return summary only
```

Per [Anthropic docs](https://code.claude.com/docs/en/sub-agents):
> "Subagents use their own isolated context windows, and only send relevant information back"

**Model selection:**
- Haiku = fast, cheap, good for read-only analysis
- Sonnet = capable, use for writes/complex reasoning
- Inherit = same as parent conversation

---

## Workflow Examples

### Quick fix
```
/execute-surgical(4c) F822 exports in numba_functions.py
```

### Comprehensive analysis
```
/analysis-full(1a) Phase 19 optimization
```

### Verify before delete
```
/verify-claim(2a) orchestrator.py is dead code
```

### Close out phase
```
/docs-final(6a) Phase 18 complete
```

---

## Sources
- [Claude Code Subagents Docs](https://code.claude.com/docs/en/sub-agents)
- [Anthropic Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
