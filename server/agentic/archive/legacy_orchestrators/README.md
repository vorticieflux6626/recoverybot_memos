# Legacy Orchestrators Archive

These files are archived for historical reference. They have been fully
superseded by `UniversalOrchestrator` in `orchestrator_universal.py`.

## Migration Table

| Archived File | Replacement |
|---------------|-------------|
| `orchestrator.py` | `UniversalOrchestrator(preset=BALANCED)` |
| `orchestrator_dynamic.py` | `UniversalOrchestrator(preset=RESEARCH)` |
| `orchestrator_enhanced.py` | `UniversalOrchestrator(preset=ENHANCED)` |
| `orchestrator_unified.py` | `UniversalOrchestrator(preset=ENHANCED)` |
| `orchestrator_graph_enhanced.py` | `UniversalOrchestrator(preset=RESEARCH)` |

## Preset Feature Mapping

| Legacy Class | Preset | Key Features |
|--------------|--------|--------------|
| AgenticOrchestrator | BALANCED | Query analysis, verification, CRAG, Self-RAG |
| EnhancedAgenticOrchestrator | ENHANCED | + HyDE, hybrid reranking, entity tracking |
| DynamicOrchestrator | RESEARCH | + Dynamic planning, graph cache |
| UnifiedOrchestrator | ENHANCED | Alias for ENHANCED preset |
| GraphEnhancedOrchestrator | RESEARCH | + KVFlow, prefetching |

## Usage After Migration

```python
# Old way (deprecated)
from agentic.orchestrator import AgenticOrchestrator
orchestrator = AgenticOrchestrator()

# New way
from agentic import UniversalOrchestrator, OrchestratorPreset
orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)
```

## Backward Compatibility

The `agentic/__init__.py` exports shim functions that redirect legacy class names
to UniversalOrchestrator with appropriate presets. These produce deprecation warnings.

```python
# Still works (with deprecation warning)
from agentic import AgenticOrchestrator
orchestrator = AgenticOrchestrator()  # Returns UniversalOrchestrator(BALANCED)
```

## Archive Details

- **Archived:** 2025-12-29
- **Reason:** Consolidation into single SSOT (UniversalOrchestrator)
- **Total Lines Archived:** ~5,425 lines
- **Active Orchestrator:** `orchestrator_universal.py` (~4,700 lines)
