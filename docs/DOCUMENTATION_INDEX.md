# memOS Documentation Index

> **Updated**: 2025-12-30 | **Parent**: [CLAUDE.md](../CLAUDE.md)

This index organizes all memOS documentation by category and status.

---

## Primary Documentation (Active)

### Core Documentation
| File | Purpose | Status |
|------|---------|--------|
| [CLAUDE.md](../CLAUDE.md) | AI assistant guidance, API reference | CURRENT |
| [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) | Project overview, architecture | CURRENT |
| [SSOT_CROSS_PROJECT_ARCHITECTURE.md](../SSOT_CROSS_PROJECT_ARCHITECTURE.md) | Cross-project data ownership | CURRENT |

### Agentic Search Documentation
| File | Purpose | Status |
|------|---------|--------|
| [AGENTIC_OVERVIEW.md](../server/agentic/AGENTIC_OVERVIEW.md) | Module overview, preset system | CURRENT |
| [FEATURE_AUDIT_REPORT.md](../server/agentic/FEATURE_AUDIT_REPORT.md) | 40+ feature flag reference | CURRENT |
| [ENGINEERING_BEST_PRACTICES.md](../server/agentic/ENGINEERING_BEST_PRACTICES.md) | Domain corpus RAG patterns | CURRENT |
| [FANUC_TEST_QUERIES.md](../server/agentic/FANUC_TEST_QUERIES.md) | Test query set | CURRENT |

---

## Implementation Reports (Complete)

### Phase 24-26: Engineering Remediation
| File | Phase | Status |
|------|-------|--------|
| [ORCHESTRATOR_CONSOLIDATION_REPORT.md](../ORCHESTRATOR_CONSOLIDATION_REPORT.md) | Phase 24 | COMPLETE |
| [FEATURE_COMBINATION_BEST_PRACTICES_REPORT.md](../FEATURE_COMBINATION_BEST_PRACTICES_REPORT.md) | Phase 25 | COMPLETE |
| [FEATURE_SYNERGY_AUDIT_REPORT.md](../FEATURE_SYNERGY_AUDIT_REPORT.md) | Phase 26 | COMPLETE |
| [COMPREHENSIVE_ENGINEERING_AUDIT_REPORT.md](../COMPREHENSIVE_ENGINEERING_AUDIT_REPORT.md) | All | COMPLETE |

### Optimization Implementation
| File | Focus | Status |
|------|-------|--------|
| [KV_CACHE_IMPLEMENTATION_PLAN.md](../server/agentic/KV_CACHE_IMPLEMENTATION_PLAN.md) | KV cache optimization | Phase 2 Complete |
| [OPTIMIZATION_ANALYSIS.md](../server/agentic/OPTIMIZATION_ANALYSIS.md) | Performance benchmarks | COMPLETE |
| [SCRATCHPAD_INTEGRATION.md](../server/agentic/SCRATCHPAD_INTEGRATION.md) | Working memory | COMPLETE |

### Audit Reports
| File | Scope | Status |
|------|-------|--------|
| [COMPREHENSIVE_AUDIT_REPORT.md](../server/agentic/COMPREHENSIVE_AUDIT_REPORT.md) | Phases 17-21 | COMPLETE |
| [PHASE_21_AUDIT_REPORT.md](../server/agentic/PHASE_21_AUDIT_REPORT.md) | Meta-Buffer + Reasoning | COMPLETE |
| [PDF_INTEGRATION_AUDIT_REPORT.md](../server/agentic/PDF_INTEGRATION_AUDIT_REPORT.md) | FANUC PDF corpus | COMPLETE |
| [MEM0_AUDIT_REPORT.md](../server/MEM0_AUDIT_REPORT.md) | Mem0 workaround | COMPLETE |

---

## Planning Documents (Reference)

### Completed Plans
| File | Purpose | Status |
|------|---------|--------|
| [ENHANCEMENT_IMPLEMENTATION_PLAN.md](../server/agentic/ENHANCEMENT_IMPLEMENTATION_PLAN.md) | Phases 1-26 roadmap | COMPLETE |
| [CONTEXT_CURATION_PLAN.md](../server/agentic/CONTEXT_CURATION_PLAN.md) | DIG scoring | Phase 17 Complete |
| [ACADEMIC_SEARCH_UPGRADE_PLAN.md](../server/agentic/ACADEMIC_SEARCH_UPGRADE_PLAN.md) | Academic domain | COMPLETE |
| [ADAPTIVE_REFINEMENT_REPORT.md](../server/agentic/ADAPTIVE_REFINEMENT_REPORT.md) | Low confidence | COMPLETE |

### In Progress Plans
| File | Purpose | Status |
|------|---------|--------|
| [HSEA_CORPUS_INGESTION_PLAN.md](../server/agentic/HSEA_CORPUS_INGESTION_PLAN.md) | FANUC corpus build | PLANNING |

---

## Integration Documentation

| File | Purpose | Status |
|------|---------|--------|
| [mcp_node_editor_integration.md](../mcp_node_editor_integration.md) | MCP Node Editor API | REFERENCE |
| [MEMOS_PDF_INTEGRATION_ENGINEERING_SUMMARY.md](../server/agentic/MEMOS_PDF_INTEGRATION_ENGINEERING_SUMMARY.md) | PDF API bridge | COMPLETE |

---

## Historical Documentation

| File | Purpose | Status |
|------|---------|--------|
| [TELEPHONE.md](../TELEPHONE.md) | Team communication (July 2025) | HISTORICAL |
| [FEATURE_INTEGRATION_ANALYSIS.md](../server/agentic/FEATURE_INTEGRATION_ANALYSIS.md) | Pre-consolidation analysis | SUPERSEDED |

---

## Documentation Hierarchy

```
memOS/
├── CLAUDE.md                      # Primary AI guidance (2700+ lines)
├── PROJECT_SUMMARY.md             # Project overview
├── SSOT_CROSS_PROJECT_ARCHITECTURE.md  # Cross-project architecture
│
├── docs/
│   └── DOCUMENTATION_INDEX.md     # This file
│
├── archive/                       # Historical/obsolete docs
│
└── server/
    ├── MEM0_AUDIT_REPORT.md       # Mem0 integration
    │
    └── agentic/
        ├── AGENTIC_OVERVIEW.md    # Module reference
        ├── FEATURE_AUDIT_REPORT.md # Feature flags
        └── ...                    # Implementation docs
```

---

## Quick Reference

### Get Started
1. Read [CLAUDE.md](../CLAUDE.md) for AI assistant guidance
2. Review [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) for architecture
3. See [AGENTIC_OVERVIEW.md](../server/agentic/AGENTIC_OVERVIEW.md) for search pipeline

### Feature Flags
- See [FEATURE_AUDIT_REPORT.md](../server/agentic/FEATURE_AUDIT_REPORT.md) for all 40+ flags
- Presets: minimal (8), balanced (18), enhanced (28), research (39), full (42+)

### Troubleshooting
- [FANUC_TEST_QUERIES.md](../server/agentic/FANUC_TEST_QUERIES.md) - Test queries
- [PHASE_21_AUDIT_REPORT.md](../server/agentic/PHASE_21_AUDIT_REPORT.md) - Bug fixes

### Research References
- [FEATURE_COMBINATION_BEST_PRACTICES_REPORT.md](../FEATURE_COMBINATION_BEST_PRACTICES_REPORT.md) - 2025 research validation
- [KV_CACHE_IMPLEMENTATION_PLAN.md](../server/agentic/KV_CACHE_IMPLEMENTATION_PLAN.md) - Performance optimization

---

*Last updated: 2025-12-30*
