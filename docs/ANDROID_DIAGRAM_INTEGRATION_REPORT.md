# Android Diagram Integration Report

> **Date**: 2026-01-17 | **Author**: Claude Code | **Status**: ✅ RESOLVED

## Resolution Summary

**Issue**: SSE event `diagram_generated` was sending only metadata, not the actual HTML content.

**Fix**: Commit `330e8b73` updated the orchestrator to include the full `diagram` object with HTML content in the SSE event payload.

**Verified**: Android client now receives and renders Mermaid flowcharts in WebView.

---

## Original Issue (Archived)

The Android client is fully prepared to render troubleshooting diagrams from agentic search responses. ~~However, the memOS server's `diagram_generated` SSE event currently only sends **metadata** without the actual HTML content, preventing diagrams from displaying.~~

---

## 1. What's Working (Android Client)

### SSE Event Handling
The Android client now properly handles diagram-related SSE events:

```
diagram_generating  →  Captured, shows "Generating diagram for SRVO-062"
diagram_generated   →  Captured, parses diagram metadata
```

### Data Flow
```
memOS SSE Event
    ↓
AgenticSearchService.parseGatewayEventToState()
    ↓
lastGeneratedDiagram (stored)
    ↓
Merged into AgenticSearchResponse.data.diagram
    ↓
GatewayResult.diagram
    ↓
ApiServiceManager captures diagram
    ↓
Message.diagram
    ↓
MessageItem renders TroubleshootingDiagramCard
```

### Files Modified
| File | Change |
|------|--------|
| `AgenticSearchService.kt` | SSE handlers for `diagram_generating`, `diagram_generated` |
| `ChatToolIntegration.kt` | Added `diagram` field to `GatewayResult` |
| `ApiServiceManager.kt` | Diagram capture from gateway and passthrough to Message |
| `MessageItem.kt` | Safe rendering with content null-check |
| `ChatModels.kt` | `hasDiagrams()` checks for content existence |

### CLI Test Command
A test command is available to verify diagram rendering without server changes:
```bash
./automation/recoverybot-cli.sh test_diagram SRVO-062
```

---

## 2. Current Issue

### What memOS Server Sends
```json
{
  "event": "diagram_generated",
  "data": {
    "error_code": "SRVO-062",
    "diagram_type": "flowchart",
    "diagram_format": "html",
    "has_content": true
  }
}
```

### What Android Client Needs
```json
{
  "event": "diagram_generated",
  "data": {
    "diagram": {
      "type": "flowchart",
      "format": "html",
      "content": "<!DOCTYPE html><html>...(full Mermaid HTML)...</html>",
      "error_code": "SRVO-062",
      "title": "Encoder Battery Depleted (BZAL)",
      "parts_needed": ["A06B-6114-K504 (Battery)"],
      "tools_needed": ["Multimeter", "ESD Strap"],
      "components_affected": ["CX5X Battery", "Encoder"],
      "mastering_required": true
    }
  }
}
```

---

## 3. Expected TroubleshootingDiagram Schema

The Android client expects this Kotlin data class structure (serialized as JSON):

```kotlin
data class TroubleshootingDiagram(
    val type: String,                    // "flowchart", "sequence", "state"
    val format: String,                  // "html", "svg", "mermaid"
    val content: String,                 // THE ACTUAL HTML/SVG CONTENT (REQUIRED)

    @SerializedName("error_code")
    val errorCode: String? = null,       // "SRVO-062"

    val title: String? = null,           // "Encoder Battery Depleted"

    @SerializedName("parts_needed")
    val partsNeeded: List<String>? = null,

    @SerializedName("tools_needed")
    val toolsNeeded: List<String>? = null,

    @SerializedName("components_affected")
    val componentsAffected: List<String>? = null,

    @SerializedName("mastering_required")
    val masteringRequired: Boolean = false
)
```

**Critical**: The `content` field MUST contain the full HTML with embedded Mermaid.js for the diagram to render.

---

## 4. Options for memOS Server

### Option A: Include diagram in `diagram_generated` SSE event
Modify the SSE event to include the full diagram object:

```python
# In orchestrator or diagram generator
yield {
    "event": "diagram_generated",
    "data": {
        "diagram": {
            "type": "flowchart",
            "format": "html",
            "content": generate_mermaid_html(error_code),
            "error_code": error_code,
            "title": get_error_title(error_code),
            "parts_needed": get_parts_for_error(error_code),
            "tools_needed": get_tools_for_error(error_code),
            "components_affected": get_components_for_error(error_code),
            "mastering_required": requires_mastering(error_code)
        }
    }
}
```

### Option B: Include diagram in `search_completed` response
Add the diagram to the final response's data object:

```python
# In search_completed response builder
response_data = {
    "synthesized_context": synthesized_text,
    "confidence_score": confidence,
    "sources": sources,
    "diagram": troubleshooting_diagram  # Add this
}
```

### Option C: Both (Recommended)
Send diagram in both places for redundancy. The Android client will use whichever is available.

---

## 5. Sample Mermaid HTML Content

Here's an example of the HTML content the Android client expects for `SRVO-062`:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body { background: #1a1a2e; margin: 0; padding: 16px; display: flex; justify-content: center; }
        .mermaid { font-family: 'Segoe UI', sans-serif; }
    </style>
    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'dark',
            themeVariables: {
                primaryColor: '#ffd700',
                primaryTextColor: '#ffffff',
                primaryBorderColor: '#4a9eff',
                lineColor: '#4a9eff',
                secondaryColor: '#2d2d44',
                tertiaryColor: '#1a1a2e'
            }
        });
    </script>
</head>
<body>
    <div class="mermaid">
        flowchart TD
            A["SRVO-062<br/>Encoder Battery Depleted"] --> B{"Check Battery Voltage"}
            B -->|"< 2.8V"| C["Replace Battery<br/>A06B-6114-K504"]
            B -->|">= 2.8V"| D{"Check Encoder Cable"}
            C --> E["Power Off Robot"]
            E --> F["Replace CX5X Battery"]
            F --> G["Power On & Master"]
            D -->|"Damaged"| H["Replace Cable"]
            D -->|"OK"| I["Check JF1/JF2 Connectors"]
            H --> G
            I -->|"Loose"| J["Reseat Connectors"]
            I -->|"OK"| K["Call FANUC Support"]
            J --> G
            G --> L["Complete Mastering Procedure"]
            style A fill:#ff4444,stroke:#ff0000,color:#ffffff
            style C fill:#ffa500,stroke:#ff8c00,color:#000000
            style G fill:#4a9eff,stroke:#0066cc,color:#ffffff
            style L fill:#00ff00,stroke:#00cc00,color:#000000
    </div>
</body>
</html>
```

---

## 6. Supported Error Codes

The Android client UI is prepared for these error codes:
- SRVO-001, SRVO-002, SRVO-006, SRVO-023, SRVO-030
- SRVO-062, SRVO-063, SRVO-065
- MOTN-017, MOTN-023
- SYST-001, HOST-005

---

## 7. Verification

Once memOS includes the diagram content, verify with:

```bash
# 1. Send a FANUC error query
./automation/recoverybot-cli.sh send "Fix FANUC error SRVO-062"

# 2. Check logs for diagram capture
adb logcat | grep -i "diagram"

# Expected output:
# I AgenticSearchService: Diagram generated: type=flowchart, errorCode=SRVO-062, hasContent=true
# I ChatToolIntegration: Diagram received from agentic search: type=flowchart, errorCode=SRVO-062
# I ApiServiceManager: Diagram captured from gateway: type=flowchart, errorCode=SRVO-062
# I ApiServiceManager: Including diagram in message: type=flowchart, errorCode=SRVO-062

# 3. Take screenshot to verify rendering
./automation/recoverybot-cli.sh screenshot /tmp/diagram_test.png
```

---

## 8. Contact

**Android Client**: `/home/sparkone/sdd/Recovery_Bot/AndroidClient/RecoveryBot/`
**Diagram Components**: `app/src/main/java/com/example/recoverybot/ui/components/diagram/`
**CLI Test Script**: `automation/recoverybot-cli.sh test_diagram`

---

*Report generated: 2026-01-17 18:35 UTC*
