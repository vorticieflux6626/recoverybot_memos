# Diagram Integration Guide for memOS Agentic Pipeline

**Date:** 2026-01-17
**From:** PDF Extraction Tools Team
**To:** memOS Server Team
**Subject:** Integration of Visual Troubleshooting Diagrams into Agentic Response Pipeline

---

## Executive Summary

The PDF Extraction Tools API now provides **visual troubleshooting diagrams** for FANUC error codes. This document outlines how to integrate these diagrams into the memOS agentic pipeline to enhance troubleshooting responses with interactive flowcharts.

### What's Available

| Feature | Endpoint | Status |
|---------|----------|--------|
| Flowchart Generation | `GET /api/v1/diagrams/html/{error_code}` | ✅ Ready |
| Supported Codes List | `GET /api/v1/diagrams/supported-errors` | ✅ Ready |
| Error Info (no diagram) | `GET /api/v1/diagrams/error-info/{error_code}` | ✅ Ready |
| MCP Tool Definitions | `GET /api/v1/tools/mcp-manifest` | ✅ Ready |

### Supported Error Codes (12 total)

```
SRVO-001, SRVO-002, SRVO-006, SRVO-023, SRVO-030
SRVO-062, SRVO-063, SRVO-065
MOTN-017, MOTN-023
SYST-001, HOST-005
```

---

## 1. Integration Architecture

### Current Flow (Text Only)
```
User Query → memOS → PDF Tools Search → Synthesis → Text Response
```

### Enhanced Flow (With Diagrams)
```
User Query → memOS → PDF Tools Search
                  ↓
            [Error Code Detected?]
                  ↓ Yes
            PDF Tools Diagram API
                  ↓
            Synthesis + Diagram
                  ↓
            Response with diagram payload
```

---

## 2. Implementation Steps

### 2.1 Extend DocumentGraphService

**File:** `memOS/server/core/document_graph_service.py`

Add these methods to the `DocumentGraphService` class:

```python
async def get_troubleshooting_diagram(
    self,
    error_code: str,
    style: str = "dark"
) -> Optional[Dict[str, Any]]:
    """
    Fetch troubleshooting diagram from PDF Tools API.

    Args:
        error_code: FANUC error code (e.g., 'SRVO-062')
        style: Visual theme ('dark', 'light', 'print')

    Returns:
        Diagram dict with html content, or None if not available
    """
    try:
        url = f"{self.pdf_tools_base_url}/api/v1/diagrams/troubleshooting-flowchart/{error_code}"
        params = {"style": style, "format": "html"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data"):
                    return {
                        "type": "flowchart",
                        "format": "html",
                        "error_code": error_code,
                        "content": data["data"]["content"],
                        "title": data["data"].get("title"),
                        "parts_needed": data["data"].get("parts_needed", []),
                        "tools_needed": data["data"].get("tools_needed", []),
                        "components_affected": data["data"].get("components_affected", []),
                        "mastering_required": data["data"].get("mastering_required", False)
                    }
        return None
    except Exception as e:
        logger.warning(f"Diagram fetch failed for {error_code}: {e}")
        return None


async def check_diagram_available(self, error_code: str) -> bool:
    """Check if a diagram exists for this error code."""
    try:
        url = f"{self.pdf_tools_base_url}/api/v1/diagrams/error-info/{error_code}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            return response.status_code == 200
    except:
        return False


def extract_error_codes(self, text: str) -> List[str]:
    """Extract FANUC error codes from query text."""
    import re
    pattern = r'(SRVO|MOTN|SYST|HOST|INTP|PROG|FILE|CVIS)-\d{3}'
    matches = re.findall(pattern, text.upper())
    return list(set(matches))
```

### 2.2 Update Response Models

**File:** `memOS/server/agentic/models.py`

Add the diagram model:

```python
from typing import Optional, List
from pydantic import BaseModel, Field

class TroubleshootingDiagram(BaseModel):
    """Visual troubleshooting diagram for Android client."""
    type: str = Field(description="Diagram type: flowchart, pinout, circuit")
    format: str = Field(description="Content format: html, mermaid, svg")
    content: str = Field(description="Diagram content (HTML for WebView)")
    error_code: Optional[str] = Field(default=None, description="Associated error code")
    title: Optional[str] = Field(default=None, description="Diagram title")
    parts_needed: List[str] = Field(default_factory=list, description="Required parts")
    tools_needed: List[str] = Field(default_factory=list, description="Required tools")
    components_affected: List[str] = Field(default_factory=list, description="Affected components")
    mastering_required: bool = Field(default=False, description="Robot mastering needed after repair")


# Update SearchResponseData to include diagrams
class SearchResponseData(BaseModel):
    synthesized_context: str
    sources: List[Source]
    search_queries: List[str]
    confidence_score: float
    confidence_level: str
    verification_status: str
    search_trace: List[dict] = []

    # NEW: Diagram support
    diagram: Optional[TroubleshootingDiagram] = None
    diagrams: List[TroubleshootingDiagram] = Field(default_factory=list)
```

### 2.3 Integrate in Synthesizer

**File:** `memOS/server/agentic/synthesizer.py`

In the synthesis method, add diagram fetching:

```python
async def _synthesize_with_diagrams(
    self,
    query: str,
    search_results: List[SearchResult],
    document_graph_service: DocumentGraphService
) -> SynthesisResult:
    """Enhanced synthesis with automatic diagram inclusion."""

    # Extract error codes from query
    error_codes = document_graph_service.extract_error_codes(query)

    # Fetch diagrams for supported codes
    diagrams = []
    for code in error_codes:
        if await document_graph_service.check_diagram_available(code):
            diagram = await document_graph_service.get_troubleshooting_diagram(code)
            if diagram:
                diagrams.append(diagram)
                # Emit SSE event for real-time UI update
                await self._emit_event("diagram_generated", {
                    "error_code": code,
                    "type": diagram["type"]
                })

    # Include diagram context in synthesis prompt
    diagram_context = ""
    if diagrams:
        diagram_context = f"""

VISUAL AID AVAILABLE:
A troubleshooting flowchart has been generated for {', '.join([d['error_code'] for d in diagrams])}.
Reference the visual diagram in your response and mention:
- Parts needed: {diagrams[0].get('parts_needed', [])}
- Tools needed: {diagrams[0].get('tools_needed', [])}
- Whether mastering is required: {diagrams[0].get('mastering_required', False)}
"""

    # Standard synthesis with diagram context
    synthesis = await self._standard_synthesis(
        query=query,
        results=search_results,
        additional_context=diagram_context
    )

    return SynthesisResult(
        content=synthesis.content,
        sources=synthesis.sources,
        confidence=synthesis.confidence,
        diagrams=diagrams  # Include in response
    )
```

### 2.4 Add SSE Event Type

**File:** `memOS/server/agentic/events.py` (or wherever SSE events are defined)

```python
class AgentEventType(str, Enum):
    SEARCH_STARTED = "search_started"
    AGENT_PROGRESS = "agent_progress"
    URL_EVALUATION = "url_evaluation"
    SCRAPING_PROGRESS = "scraping_progress"
    SYNTHESIS_STARTED = "synthesis_started"
    SEARCH_COMPLETE = "search_complete"

    # NEW
    DIAGRAM_GENERATED = "diagram_generated"
```

### 2.5 Update prompts.yaml

**File:** `memOS/server/config/prompts.yaml`

Add diagram awareness to instructions:

```yaml
instructions:
  # ... existing instructions ...

  diagram_capability: |
    VISUAL TROUBLESHOOTING DIAGRAMS:
    For these FANUC error codes, interactive flowcharts are available:
    - SRVO-001, SRVO-002, SRVO-006 (Safety/E-Stop)
    - SRVO-023, SRVO-030, SRVO-065 (Servo/Communication)
    - SRVO-062, SRVO-063 (Encoder/Battery)
    - MOTN-017, MOTN-023 (Motion)
    - SYST-001, HOST-005 (System/Host)

    When diagrams are included:
    1. Reference them: "See the troubleshooting flowchart below"
    2. Mention required parts from the diagram metadata
    3. Mention required tools from the diagram metadata
    4. Note if mastering is required after repair

    The Android client will render diagrams automatically.
```

Update synthesizer prompt to use it:

```yaml
agent_prompts:
  synthesizer:
    main: |
      # ... existing prompt ...

      {diagram_capability}

      Your synthesized answer (with diagram reference if available):
```

---

## 3. API Reference

### 3.1 Get Flowchart HTML

```bash
GET /api/v1/diagrams/troubleshooting-flowchart/{error_code}?style=dark&format=html

# Response
{
  "success": true,
  "data": {
    "error_code": "SRVO-062",
    "title": "Pulse Coder Battery Depleted (BZAL)",
    "category": "encoder",
    "severity": "medium",
    "diagram_type": "flowchart",
    "format": "html",
    "content": "<!DOCTYPE html>...",
    "parts_needed": ["A06B-6114-K504"],
    "tools_needed": ["Multimeter"],
    "components_affected": ["CX5X Battery", "Encoder", "JF1", "JF2"],
    "mastering_required": false
  }
}
```

### 3.2 Check Diagram Availability

```bash
GET /api/v1/diagrams/error-info/{error_code}

# Response (200 = available, 404 = not available)
{
  "success": true,
  "data": {
    "error_code": "SRVO-062",
    "title": "Pulse Coder Battery Depleted (BZAL)",
    "category": "encoder",
    "severity": "medium",
    "components": ["CX5X Battery", "Encoder"],
    "step_count": 6,
    "parts_needed": ["A06B-6114-K504"],
    "tools_needed": ["Multimeter"]
  }
}
```

### 3.3 List All Supported Codes

```bash
GET /api/v1/diagrams/supported-errors

# Response
{
  "success": true,
  "data": {
    "count": 12,
    "error_codes": [
      {"error_code": "SRVO-001", "title": "...", "category": "safety", ...},
      ...
    ],
    "categories": {"safety": 3, "servo": 1, "encoder": 3, ...}
  }
}
```

---

## 4. Response Format for Android

The final response to Android should include:

```json
{
  "success": true,
  "data": {
    "synthesized_context": "SRVO-062 indicates encoder battery depletion...",
    "sources": [...],
    "confidence_score": 0.92,
    "diagram": {
      "type": "flowchart",
      "format": "html",
      "content": "<!DOCTYPE html><html>...",
      "error_code": "SRVO-062",
      "title": "Pulse Coder Battery Depleted (BZAL)",
      "parts_needed": ["A06B-6114-K504"],
      "tools_needed": ["Multimeter"],
      "components_affected": ["CX5X Battery", "Encoder", "JF1", "JF2"],
      "mastering_required": false
    }
  }
}
```

---

## 5. Testing

### Unit Test Example

```python
import pytest
from core.document_graph_service import DocumentGraphService

@pytest.mark.asyncio
async def test_diagram_fetch():
    service = DocumentGraphService()

    # Test supported code
    diagram = await service.get_troubleshooting_diagram("SRVO-062")
    assert diagram is not None
    assert diagram["type"] == "flowchart"
    assert diagram["format"] == "html"
    assert "content" in diagram
    assert len(diagram["content"]) > 100  # Has substantial HTML

    # Test unsupported code
    diagram = await service.get_troubleshooting_diagram("SRVO-999")
    assert diagram is None

@pytest.mark.asyncio
async def test_error_code_extraction():
    service = DocumentGraphService()

    codes = service.extract_error_codes("How do I fix SRVO-062 and MOTN-017?")
    assert "SRVO-062" in codes
    assert "MOTN-017" in codes
    assert len(codes) == 2
```

### Integration Test

```bash
# Start PDF Tools API
curl http://localhost:8002/api/v1/diagrams/supported-errors

# Test full flow through memOS
curl -X POST http://localhost:8001/api/v1/search/universal \
  -H "Content-Type: application/json" \
  -d '{"query": "How to troubleshoot SRVO-062?", "preset": "research"}'

# Verify response contains diagram field
```

---

## 6. Performance Considerations

| Operation | Expected Latency |
|-----------|------------------|
| Diagram availability check | <50ms |
| Full diagram fetch | <200ms |
| HTML content size | ~5-10KB per diagram |

Recommendations:
- Cache diagram availability results (5 min TTL)
- Fetch diagrams in parallel with synthesis
- Don't block synthesis on diagram fetch (include if ready)

---

## 7. Rollout Plan

### Week 1: Integration
- Implement DocumentGraphService methods
- Update response models
- Add to `research` and `full` presets only

### Week 2: Testing
- Integration tests with Android team
- Performance benchmarking
- Error handling verification

### Week 3: GA
- Enable for all presets
- Add feature flag for diagram inclusion
- Monitor analytics

---

## 8. Contact

**PDF Extraction Tools API:** Port 8002
**Documentation:** `/home/sparkone/sdd/PDF_Extraction_Tools/docs/`
**MCP Tools:** `GET /api/v1/tools/mcp-manifest`

For questions or issues, check the audit report:
`/home/sparkone/sdd/PDF_Extraction_Tools/docs/MEMOS_DIAGRAM_INTEGRATION_AUDIT.md`

---

*Generated: 2026-01-17*
