# Commands Cheat Sheet

## Analysis (1x)
| Command | Use When |
|---------|----------|
| `/analysis-full` | Full codebase analysis (4 parallel agents) |
| `/analysis-optimization` | Find improvements ranked by impact × effort |
| `/analysis-targeted` | Deep dive on specific module/file |

## Verification (2x)
| Command | Use When |
|---------|----------|
| `/verify-claim` | Check if single claim is true (checks COMPLETION.md first) |
| `/verify-batch` | Verify multiple proposed deletions/changes |
| `/verify-contracts` | Validate behavioral assumptions |

## Documentation (3x)
| Command | Use When |
|---------|----------|
| `/docs-tasks` | Update CLEANUP_TASKS.md with findings |
| `/docs-full` | Update all root docs together |

## Execution (4x)
| Command | Use When |
|---------|----------|
| `/execute-surgical` | Single targeted fix (2 agents) |
| `/execute-standard` | Medium task (4 agents) |
| `/execute-large` | Major phase (7 agents) |

## Checking (5x)
| Command | Use When |
|---------|----------|
| `/check-standard` | Run validation checklist |
| `/check-deep` | Comprehensive 4-agent verification |
| `/check-behavior` | Verify specific behavior still works |

## Final (6x)
| Command | Use When |
|---------|----------|
| `/docs-final` | Archive completed phase to COMPLETION.md |
