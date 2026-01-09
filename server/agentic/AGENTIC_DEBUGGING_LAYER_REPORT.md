# Agentic Debugging Layer - Implementation Report

> **Date**: 2026-01-06 | **Status**: Complete | **Author**: Claude Code

---

## Executive Summary

The Agentic Debugging Layer has been successfully implemented, wiring the existing observability infrastructure (P0-P3 components) into the `UniversalOrchestrator` pipeline. This enables full visibility into every decision point, LLM call, and context flow within the agentic search system.

**Key Achievement**: The unified_dashboard at `/home/sparkone/sdd/unified_dashboard/` will now receive real-time observability data from memOS agentic searches.

---

## What Was Implemented

### 1. Observability Imports Added

Added imports for all P0-P3 observability components to `orchestrator_universal.py`:

```python
from .decision_logger import DecisionLogger, get_decision_logger, DecisionType, AgentName
from .llm_logger import LLMCallLogger, get_llm_logger, LLMOperation
from .context_tracker import ContextFlowTracker, get_context_tracker
from .scratchpad_observer import ScratchpadObserver, get_scratchpad_observer
from .confidence_logger import ConfidenceLogger, get_confidence_logger
from .observability_dashboard import (
    get_observability_dashboard,
    ObservabilityAggregator,
    create_request_observability
)
```

### 2. Logger Initialization

Both `search()` and `search_with_events()` methods now initialize observability loggers:

```python
# Initialize observability loggers (P0-P3)
verbose_mode = request.options.get("verbose_logging", False) if request.options else False
decision_logger = get_decision_logger(request_id, emitter, verbose=verbose_mode)
llm_logger = get_llm_logger(request_id, emitter, verbose=verbose_mode)
context_tracker = get_context_tracker(request_id)
obs_aggregator = ObservabilityAggregator(request_id, request.query, self.preset.value)
```

### 3. Decision Points Instrumented

| Pipeline Phase | Decision Logged | Agent | Decision Type |
|----------------|-----------------|-------|---------------|
| PHASE 1 | Query analysis | ANALYZER | CLASSIFICATION |
| PHASE 1.4 | DyLAN complexity | DYLAN | CLASSIFICATION |
| PHASE 3.5 | CRAG evaluation | CRAG | EVALUATION |
| PHASE 6 | Synthesis | SYNTHESIZER | GENERATION |
| PHASE 7 | Self-RAG reflection | SELF_RAG | EVALUATION |

### 4. Context Flow Tracking

Context transfers are recorded between pipeline stages:

| Source | Target | Context Type |
|--------|--------|--------------|
| input | analyzer | query |
| analyzer | scratchpad | analysis |
| searcher | scraper | search_results |
| all_sources | synthesizer | combined_context |

### 5. Observability Aggregation

At the end of each request, all observability data is aggregated and stored:

```python
obs_aggregator.add_decisions(decision_logger.decisions)
obs_aggregator.add_context_flow(context_tracker.get_flow_summary())
obs_aggregator.add_llm_calls(llm_logger.get_call_summary())
obs_aggregator.add_confidence(confidence_breakdown)
obs_aggregator.add_feature_usage(enabled_features)

obs_record = obs_aggregator.finalize(
    success=True,
    duration_ms=execution_time_ms,
    error=None
)

dashboard = get_observability_dashboard()
dashboard.store_request(obs_record)
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          memOS Server (Port 8001)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  UniversalOrchestrator.search_with_events()                                 │
│    │                                                                         │
│    ├─► DecisionLogger                                                        │
│    │     └─► log_decision()  ─────────► SSE: DECISION_POINT                 │
│    │                                                                         │
│    ├─► LLMCallLogger                                                         │
│    │     └─► track_call()  ───────────► SSE: LLM_CALL_START/COMPLETE        │
│    │                                                                         │
│    ├─► ContextTracker                                                        │
│    │     └─► record_transfer()  ──────► SSE: CONTEXT_TRANSFER               │
│    │                                                                         │
│    └─► ObservabilityAggregator                                              │
│          └─► finalize()  ─────────────► ObservabilityDashboard.store()      │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  API Endpoints (api/observability.py)                                        │
│    │                                                                         │
│    ├─► GET /api/v1/observability/recent  ────► Recent requests summary      │
│    ├─► GET /api/v1/observability/stats   ────► Aggregate statistics         │
│    ├─► GET /api/v1/observability/request/{id}  ► Full request details       │
│    └─► GET /api/v1/observability/health  ────► Component health status      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     unified_dashboard (Port 3100/3101)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Express Backend (server/routes/agent.ts)                                    │
│    ├─► GET /api/agent/stream/global  ───► Polls memOS /observability/recent │
│    ├─► GET /api/agent/history        ───► Returns stored run history        │
│    └─► GET /api/agent/observability/:id ► Gets full request details         │
│                                                                              │
│  React Frontend (AgentConsole.tsx)                                           │
│    ├─► Live Feed Tab  ──────────────────► Real-time event stream            │
│    ├─► Run History Tab  ────────────────► Past search results               │
│    └─► Event Detail Expansion  ─────────► JSON data viewer                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Verification Instructions

### Step 1: Restart memOS Server

```bash
cd /home/sparkone/sdd/Recovery_Bot/memOS/server
source venv/bin/activate

# Stop existing server if running
pkill -f "uvicorn main:app" || true

# Start server with new code
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Step 2: Execute Test Search

```bash
# Standard search (non-verbose)
curl -X POST 'http://localhost:8001/api/v1/search/chat-gateway' \
  -H 'Content-Type: application/json' \
  -d '{"query": "SRVO-063 encoder error troubleshooting", "preset": "enhanced"}'

# Verbose search (captures full LLM prompts/responses)
curl -X POST 'http://localhost:8001/api/v1/search/chat-gateway' \
  -H 'Content-Type: application/json' \
  -d '{"query": "SRVO-063 encoder error", "preset": "enhanced", "options": {"verbose_logging": true}}'
```

### Step 3: Verify Observability Data

```bash
# Check recent requests (should now have data)
curl 'http://localhost:8001/api/v1/observability/recent?limit=5'

# Expected response structure:
# {
#   "success": true,
#   "count": 1,
#   "requests": [
#     {
#       "request_id": "uuid",
#       "query": "SRVO-063 encoder error troubleshooting",
#       "preset": "enhanced",
#       "timestamp": "2026-01-06T...",
#       "duration_ms": 45000,
#       "success": true,
#       "decision_count": 5,
#       "llm_call_count": 3,
#       "context_transfers": 4
#     }
#   ]
# }

# Get full details for a specific request
curl 'http://localhost:8001/api/v1/observability/request/{request_id}'

# Check observability health
curl 'http://localhost:8001/api/v1/observability/health'
```

### Step 4: Test unified_dashboard

```bash
cd /home/sparkone/sdd/unified_dashboard
npm run dev

# Open browser to http://localhost:3100
# Navigate to Agent Console
# Execute a test query and observe real-time events
```

---

## Files Modified

| File | Changes |
|------|---------|
| `agentic/orchestrator_universal.py` | Added observability imports, logger initialization, decision logging at 5 key phases, context tracking, aggregation at request completion |
| `agentic/AGENTIC_DEBUGGING_LAYER_PLAN.md` | Created comprehensive implementation plan |
| `agentic/AGENTIC_DEBUGGING_LAYER_REPORT.md` | This report |

---

## Observability Data Available

### Decision Log Entry

```json
{
  "agent_name": "analyzer",
  "decision_type": "classification",
  "decision_made": "query_type=troubleshooting",
  "reasoning": "Analyzed query: requires_search=True, complexity=moderate",
  "alternatives": ["factual", "research", "creative", "troubleshooting", "code"],
  "confidence": 0.85,
  "context_size_tokens": 12,
  "timestamp": "2026-01-06T12:00:00Z"
}
```

### Context Flow Entry

```json
{
  "source": "searcher",
  "target": "scraper",
  "context_type": "search_results",
  "content_preview": "Found 15 results from SearXNG...",
  "size_chars": 4523,
  "timestamp": "2026-01-06T12:00:05Z"
}
```

### LLM Call Entry (Verbose Mode)

```json
{
  "agent_name": "synthesizer",
  "operation": "synthesis",
  "model": "qwen3:8b",
  "input_tokens": 2048,
  "output_tokens": 512,
  "duration_ms": 8500,
  "prompt": "...(full prompt when verbose=true)...",
  "response": "...(full response when verbose=true)..."
}
```

---

## Verbose Mode Usage

To capture full LLM prompts and responses (for debugging):

### Via API

```bash
curl -X POST 'http://localhost:8001/api/v1/search/chat-gateway' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "your query here",
    "preset": "enhanced",
    "options": {
      "verbose_logging": true
    }
  }'
```

### Via Android Client

The Android client can pass `verbose_logging: true` in the options map when calling `AgenticSearchService.gatewaySearch()`.

---

## Success Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| `/observability/recent` returns data | After any search | Check `count > 0` in response |
| Decisions logged per request | 5+ decisions | Check `decision_count` in response |
| LLM calls tracked | 100% of calls | Check `llm_call_count` matches expected |
| Context transfers tracked | 4+ transfers | Check `context_transfers` in response |
| unified_dashboard shows data | Real-time | Visual verification in Agent Console |
| Verbose mode captures prompts | When enabled | Check `prompt` field in LLM calls |

---

## Related Documentation

- **Plan Document**: `agentic/AGENTIC_DEBUGGING_LAYER_PLAN.md`
- **Observability Components**: `agentic/observability_dashboard.py`, `decision_logger.py`, `llm_logger.py`, `context_tracker.py`
- **API Endpoints**: `api/observability.py`
- **unified_dashboard**: `/home/sparkone/sdd/unified_dashboard/`
- **Integration Report**: `/home/sparkone/sdd/PDF_Extraction_Tools/MEMOS_INTEGRATION_REPORT_2026-01-06.md`

---

## Next Steps (Optional Enhancements)

1. **Phase 2**: Add verbose logging integration to synthesizer, analyzer, and verifier components
2. **Phase 3**: Create `/api/v1/search/debug` endpoint for debugging-specific queries
3. **Dashboard**: Add decision tree visualization, context flow diagrams, confidence breakdown charts
4. **Alerts**: Configure alerts for low confidence scores or failed searches

---

*Report generated by Claude Code | 2026-01-06*
