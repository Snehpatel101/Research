# Cleanup Plan: [Project Name]

**Status:** Phases X-Y Complete | Phase Z Analysis Ready
**Generated:** [DATE]
**Total Lines Removed:** [AGGREGATE]

---

## Table of Contents
- [Completed Phases Summary](#completed-phases-summary)
- [Phase N: Current Phase](#phase-n-title)
- [Execution Roadmap](#execution-roadmap)
- [NOT Doing (Deferrals)](#not-doing-deferrals)

---

## Completed Phases Summary

| Phase | Description | Lines Removed | Impact |
|-------|-------------|---------------|--------|
| 1 | [Brief description] | ~XXX | [Metric improvement] |
| 2 | [Brief description] | ~XXX | [Metric improvement] |

**Total Lines Removed:** ~X,XXX

---

## Phase [N]: [Title]

**Priority:** [CRITICAL | HIGH | MEDIUM | LOW]
**Status:** Analysis Complete | Execution Pending | COMPLETE
**Source:** [N-agent investigation / Manual analysis]

### Problem Statement

[What's wrong, quantified. Include metrics where possible.]

### Architecture Analysis

**Current State:**
```
[ASCII diagram showing current flow]
```

**Target State:**
```
[ASCII diagram showing improved flow]
```

### Execution Roadmap

| Sub-Phase | Task | Priority | Est. Effort | Dependencies | Status |
|-----------|------|----------|-------------|--------------|--------|
| [N]A | [Task description] | CRITICAL | [Effort] | None | Pending |
| [N]B | [Task description] | HIGH | [Effort] | [N]A | Pending |
| [N]C | [Task description] | MEDIUM | [Effort] | None | Pending |

### Files to Modify

| File | Line Range | Changes Required |
|------|------------|------------------|
| `path/to/file.py` | 100-150 | [Description of change] |
| `path/to/other.py` | 200-250 | [Description of change] |

### Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| [Startup time] | [10s] | [4s] | [60%] |
| [Lines of code] | [500] | [300] | [-200] |

---

## Phase [N+1]: [Title] (Pending)

**Priority:** [LEVEL]
**Status:** Identified | Analysis Pending
**Blocked By:** Phase [N]

[Brief description of what this phase will address]

---

## NOT Doing (Deferrals)

### 1. [Deferred Task/Approach]

**Reason:** [Why this was explicitly rejected]
- [Supporting detail 1]
- [Supporting detail 2]

### 2. [Another Deferred Item]

**Reason:** [Rationale]

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| COMPLETE | Work finished and verified |
| Pending | Ready to execute |
| Analysis | Investigation in progress |
| Deferred | Explicitly not doing |
| Blocked | Waiting on dependency |

---

## Validation Checklist

- [ ] Type checking passes (`mypy`)
- [ ] Linting passes (`ruff check`)
- [ ] Backend starts without errors
- [ ] Frontend builds without errors

---

## Notes

[Any additional context, decisions made, or future considerations]
