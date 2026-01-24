# [Project Name]: Direction & Analysis

**Generated:** [DATE]
**Status:** Phases [X-Y] Complete | Phase [Z] Identified

[One-line summary of what this document synthesizes]

---

## Table of Contents
- [Current State](#current-state)
- [Architecture Overview](#architecture-overview)
- [Data Flows](#data-flows)
- [Refactoring Trajectory](#refactoring-trajectory)
- [Cleanup Phases](#cleanup-phases)
- [What NOT to Do](#what-not-to-do)
- [Summary](#summary)

---

## Current State

### Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Files | ~XXX | [Up/Down from previous] |
| Total LOC | ~XXX,XXX | [Up/Down from previous] |
| [Domain Files] | ~XXX | [Assessment] |
| [Test Coverage] | XX% | [Assessment] |

### Compliance Status

| Standard | Status | Notes |
|----------|--------|-------|
| [Pattern 1 - e.g., DI compliance] | XX% | [Details] |
| [Pattern 2 - e.g., No raw SQL in services] | 100% | Enforced |
| [Pattern 3 - e.g., State encapsulation] | XX% | [Remaining violations] |

---

## Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     [External Layer]                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ [Input 1]   │    │ [Input 2]   │    │ [Input 3]   │      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    [Core Layer]                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              [Central Component]                     │    │
│  │   ┌────────┐  ┌────────┐  ┌────────┐               │    │
│  │   │Service1│  │Service2│  │Service3│               │    │
│  │   └────────┘  └────────┘  └────────┘               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  [Persistence Layer]                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ [Store 1]   │    │ [Store 2]   │    │ [Cache]     │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Role | When Used |
|-----------|------|-----------|
| [Component 1] | [Primary responsibility] | [Trigger/condition] |
| [Component 2] | [Primary responsibility] | [Trigger/condition] |
| [Component 3] | [Primary responsibility] | [Trigger/condition] |

---

## Data Flows

### [Primary Flow Name]

```
[Source]
    │
    ▼
[Processor 1]
    │
    ▼ [Event/Message Type]
[Processor 2]
    │
    ├──────────────────┐
    ▼                  ▼
[Output 1]        [Output 2]
```

**Flow Description:**
1. [Step 1 explanation]
2. [Step 2 explanation]
3. [Step 3 explanation]

### [Secondary Flow Name]

[Similar pattern]

---

## Decision Trees

### When to Use [Approach A] vs [Approach B]

```
Need to [action]?
├── [Condition 1] → Use [Approach A]
├── [Condition 2] → Use [Approach B]
├── [Condition 3] → Use [Approach C]
└── [Default case] → Use [Approach D]
```

### [Another Decision]

[Similar format]

---

## Refactoring Trajectory

### Completed Phases

| Phase | Description | Lines Removed | Date |
|-------|-------------|---------------|------|
| 1-3 | [Summary of early phases] | ~X,XXX | [Date] |
| 4-6 | [Summary of middle phases] | ~X,XXX | [Date] |
| 7-N | [Summary of recent phases] | ~X,XXX | [Date] |

**Total Impact:** ~X,XXX lines removed, XX% compliance improvement

### Key Wins

- [Major accomplishment 1]
- [Major accomplishment 2]
- [Major accomplishment 3]

---

## Cleanup Phases

### Completed

| Phase | Task | Status |
|-------|------|--------|
| [N] | [Description] | COMPLETE |

### Current / Pending

| Phase | Task | Priority | Effort |
|-------|------|----------|--------|
| [N+1] | [Description] | HIGH | [Estimate] |
| [N+2] | [Description] | MEDIUM | [Estimate] |

---

## What NOT to Do

### 1. Don't [Anti-Pattern 1]

**Reason:** [Architectural rationale]
- [Supporting detail 1]
- [Supporting detail 2]
- [What to do instead]

### 2. Don't [Anti-Pattern 2]

**Reason:** [Rationale]
- [Details]

### 3. Don't [Anti-Pattern 3]

**Reason:** [Rationale]
- [Details]

---

## Summary

### Wins So Far

- [X,XXX] lines removed across [N] phases
- [XX%] compliance with [standard]
- [Performance/quality improvement]

### Remaining Work

- Phase [N+1]: [Brief description]
- Phase [N+2]: [Brief description]

### Strategic Direction

[1-2 paragraphs on where the project is heading and why]

---

## Appendix: Subsystem Details

### [Subsystem 1]

**Current Architecture:**
```
[Diagram]
```

**Components:**
| Component | Status | Notes |
|-----------|--------|-------|
| [Comp A] | Active | [Details] |
| [Comp B] | DELETE in Phase X | [Reason] |

**Known Issues:**
- [Issue 1]: [Fix required/planned]
- [Issue 2]: [Fix required/planned]

### [Subsystem 2]

[Similar format]
