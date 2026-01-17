# Diagram Integration Implementation Plan

**Date:** 2026-01-17
**Status:** COMPLETE ✅
**Based on:** DIAGRAM_INTEGRATION_GUIDE.md audits

## Implementation Status

| Step | Component | Status |
|------|-----------|--------|
| 1 | EventType enum (DIAGRAM_GENERATING, DIAGRAM_GENERATED) | ✅ DONE |
| 2 | Event helper functions (diagram_generating, diagram_generated) | ✅ DONE |
| 3 | TroubleshootingDiagram model | ✅ DONE |
| 4 | SearchResultData.diagram field | ✅ DONE |
| 5 | DocumentGraphService.check_diagram_available() | ✅ DONE |
| 6 | DocumentGraphService.get_troubleshooting_diagram() | ✅ DONE |
| 7 | DocumentGraphService.extract_error_codes() | ✅ DONE |
| 8 | prompts.yaml diagram_capability instruction | ✅ DONE |
| 9 | Orchestrator Phase 12.12 diagram integration | ✅ DONE |
| 10 | base_pipeline.py build_response() diagram param | ✅ DONE |
| 11 | SSE event includes full diagram content | ✅ DONE |
| 12 | Unit tests | ✅ PASSED |
| 13 | Android client integration verified | ✅ WORKING |

---

## Overview

This plan integrates PDF_Extraction_Tools visual troubleshooting diagrams into the memOS agentic pipeline, enabling interactive flowcharts for FANUC error codes in Android client responses.

## Audit Summary

| Component | Status | Key Findings |
|-----------|--------|--------------|
| DocumentGraphService | Ready | Insert at lines 206, 628; existing httpx patterns |
| Synthesizer | Ready | Context injection at line ~619; SSE patterns exist |
| prompts.yaml | Ready | Add instruction after line 146; update synthesizer.main |
| Response Models | Ready | Add TroubleshootingDiagram to models.py |
| SSE Events | Ready | Add DIAGRAM_GENERATED to EventType enum |

---

## Implementation Steps

### Step 1: Add EventType and Helper Functions

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/events.py`

1. Add to EventType enum (after line 64):
   ```python
   DIAGRAM_GENERATING = "diagram_generating"
   DIAGRAM_GENERATED = "diagram_generated"
   ```

2. Add helper functions (after line 809):
   ```python
   def diagram_generating(request_id: str, error_code: str = "") -> SearchEvent
   def diagram_generated(request_id: str, diagram: Dict, error_code: str = "") -> SearchEvent
   ```

### Step 2: Add TroubleshootingDiagram Model

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/models.py`

1. Add after SearchResponse (line ~396):
   ```python
   class TroubleshootingDiagram(BaseModel):
       type: str  # flowchart, pinout, circuit
       format: str  # html, mermaid, svg
       content: str
       error_code: Optional[str] = None
       title: Optional[str] = None
       parts_needed: List[str] = []
       tools_needed: List[str] = []
       components_affected: List[str] = []
       mastering_required: bool = False
   ```

2. Add to SearchResultData:
   ```python
   diagram: Optional[TroubleshootingDiagram] = None
   ```

### Step 3: Add DocumentGraphService Methods

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/core/document_graph_service.py`

1. Add `check_diagram_available()` after line 206
2. Add `get_troubleshooting_diagram()` after line 628
3. These methods call PDF Tools API at port 8002

### Step 4: Update prompts.yaml

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/config/prompts.yaml`

1. Add `diagram_capability` instruction after line 146
2. Add `{diagram_capability}` to synthesizer.main template

### Step 5: Update Synthesizer Context Injection

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/synthesizer.py`

1. Accept `diagram_metadata` in context parameter
2. Build `diagram_context_text` section in prompt
3. Emit DIAGRAM_GENERATED SSE event after synthesis

### Step 6: Update Orchestrator

**File:** `/home/sparkone/sdd/Recovery_Bot/memOS/server/agentic/orchestrator_universal.py`

1. Extract error codes from query using `_extract_error_codes()`
2. Check diagram availability via DocumentGraphService
3. Fetch diagram if available
4. Include in synthesis context
5. Emit SSE events
6. Include diagram in SearchResponse

---

## Testing Plan

1. **Unit Tests:**
   - Test diagram availability check
   - Test diagram fetch with mock API
   - Test error code extraction

2. **Integration Tests:**
   ```bash
   # Test PDF Tools diagram endpoint
   curl http://localhost:8002/api/v1/diagrams/html/SRVO-062

   # Test memOS search with diagram
   curl -X POST http://localhost:8001/api/v1/search/universal \
     -H "Content-Type: application/json" \
     -d '{"query": "How to troubleshoot SRVO-062?", "preset": "research"}'
   ```

3. **Android Client Test:**
   - Verify SSE event reception
   - Verify diagram rendering in WebView

---

## Files to Modify

| File | Changes |
|------|---------|
| `events.py` | Add 2 EventType values, 2 helper functions |
| `models.py` | Add TroubleshootingDiagram model, extend SearchResultData |
| `document_graph_service.py` | Add 2-3 new methods |
| `prompts.yaml` | Add diagram_capability instruction |
| `synthesizer.py` | Add diagram context injection |
| `orchestrator_universal.py` | Add diagram fetch and SSE emission |

---

## Rollback Plan

If issues occur:
1. All changes are additive (no breaking changes)
2. Diagram field is Optional (backward compatible)
3. Can disable via feature flag in config

---

## Success Criteria

- [x] PDF Tools diagram API responds correctly
- [x] memOS extracts error codes from queries
- [x] Diagrams included in SearchResponse
- [x] SSE events emitted with full diagram content
- [x] Android client receives and displays diagrams

---

## Commits

| Commit | Repository | Description |
|--------|------------|-------------|
| `dcad9532` | recoverybot_memos | Initial diagram integration (models, events, services) |
| `cd6f8734` | recoverybot_memos | Add Phase 12.12 to orchestrator |
| `330e8b73` | recoverybot_memos | Fix SSE event to include full diagram content |

---

## Verified Flow (2026-01-17)

```
Query: "What is SRVO-062 and how do I fix it?"
  ↓
Error code extraction: SRVO-062
  ↓
Diagram availability check: 200 OK
  ↓
Diagram fetch: 2889 chars HTML
  ↓
SSE event: diagram_generated (with full content)
  ↓
SearchResponse.data.diagram populated
  ↓
Android WebView renders flowchart ✅
```

