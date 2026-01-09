# Agentic Debugging Layer Plan

> **Created**: 2026-01-06 | **Status**: In Progress | **Priority**: High

## Executive Summary

This document outlines the plan to establish full debugging visibility into the memOS agentic pipeline. The goal is to enable debugging of every text processing step, LLM input/output, NLP technique decision, search reasoning, and analytical conclusion.

## Current State Assessment

### What EXISTS (Infrastructure Built)

| Component | File | Status | Purpose |
|-----------|------|--------|---------|
| SSE Event System | `events.py` | ✅ Active | 95+ event types for real-time progress |
| DecisionLogger | `decision_logger.py` | ✅ Built | Track agent decisions with reasoning |
| LLMCallLogger | `llm_logger.py` | ✅ Built | Comprehensive LLM invocation tracking |
| ContextTracker | `context_tracker.py` | ✅ Built | Context flow between pipeline stages |
| ScratchpadObserver | `scratchpad_observer.py` | ✅ Built | Track scratchpad state changes |
| TechnicianLog | `technician_log.py` | ✅ Built | Human-readable diagnostic summaries |
| ConfidenceLogger | `confidence_logger.py` | ✅ Built | Multi-signal confidence breakdown |
| ObservabilityDashboard | `observability_dashboard.py` | ✅ Built | Aggregate observability views |
| API Endpoints | `api/observability.py` | ✅ Built | REST API for dashboard data |

### What's MISSING (The Gap)

| Issue | Impact | Effort |
|-------|--------|--------|
| **ObservabilityAggregator not wired into orchestrator** | Dashboard endpoints return empty data | Medium |
| **DecisionLogger not used in agents** | Cannot see WHY decisions were made | Medium |
| **LLMCallLogger not integrated** | Cannot see full prompts/responses | Medium |
| **ContextTracker not connected** | Cannot trace data flow between stages | Low |
| **Verbose mode not configurable via API** | Cannot enable full prompt capture on-demand | Low |

### unified_dashboard Status

The unified_dashboard project at `/home/sparkone/sdd/unified_dashboard/` is:
- ✅ React 18 + TypeScript frontend (port 3100)
- ✅ Express.js backend (port 3101)
- ✅ Has AgentConsole component for viewing agent events
- ✅ Routes correctly configured to call memOS observability endpoints
- ⚠️ **Returns empty data** because observability data is not being collected

---

## Phase 1: Wire Observability Into Orchestrator (Priority P0)

### 1.1 Import Observability Components

Add to `orchestrator_universal.py` imports:

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

### 1.2 Initialize Loggers in search_with_events()

At the start of `search_with_events()`:

```python
async def search_with_events(self, request: SearchRequest, emitter: EventEmitter) -> SearchResponse:
    request_id = emitter.request_id

    # Initialize observability loggers
    verbose_mode = request.options.get("verbose_logging", False) if request.options else False

    decision_logger = get_decision_logger(request_id, emitter, verbose=verbose_mode)
    llm_logger = get_llm_logger(request_id, emitter, verbose=verbose_mode)
    context_tracker = get_context_tracker(request_id)

    # Create aggregator for final dashboard storage
    aggregator = ObservabilityAggregator(request_id, request.query, self.preset.value)

    # ... rest of pipeline ...
```

### 1.3 Log Decisions at Key Points

Example integration points:

```python
# After query analysis
await decision_logger.log_decision(
    agent_name=AgentName.ANALYZER,
    decision_type=DecisionType.CLASSIFICATION,
    decision_made=f"query_type={query_analysis.query_type}",
    reasoning=f"Detected {query_analysis.query_type} based on keywords and structure",
    alternatives=["factual", "research", "creative", "troubleshooting"],
    confidence=query_analysis.confidence,
    context_size_tokens=len(request.query) // 4
)

# After CRAG evaluation
await decision_logger.log_decision(
    agent_name=AgentName.CRAG,
    decision_type=DecisionType.EVALUATION,
    decision_made=crag_result.action.value,
    reasoning=crag_result.reasoning,
    alternatives=["PROCEED", "REFINE_QUERY", "WEB_FALLBACK", "DECOMPOSE"],
    confidence=crag_result.relevance_score,
    metadata={"threshold": crag_threshold, "score": crag_result.relevance_score}
)
```

### 1.4 Track Context Flow

```python
# After search results
context_tracker.record_transfer(
    source="searcher",
    target="scraper",
    content=search_results,
    context_type="search_results"
)

# After scraping
context_tracker.record_transfer(
    source="scraper",
    target="verifier",
    content=scraped_content,
    context_type="scraped_content"
)
```

### 1.5 Finalize and Store Observability

At the end of `search_with_events()`:

```python
# Aggregate all observability data
aggregator.add_decisions(decision_logger.decisions)
aggregator.add_context_flow(context_tracker.get_flow_summary())
aggregator.add_llm_calls(llm_logger.get_call_summary())
aggregator.add_confidence(confidence_breakdown)

# Finalize with outcome
obs = aggregator.finalize(
    success=response.success,
    duration_ms=int((time.time() - start_time) * 1000),
    error=response.error if not response.success else None
)

# Store in dashboard
dashboard = get_observability_dashboard()
dashboard.store_request(obs)
```

---

## Phase 2: Enable Verbose LLM Logging (Priority P1)

### 2.1 Add Verbose Flag to FeatureConfig

```python
@dataclass
class FeatureConfig:
    # ... existing flags ...

    # Observability flags
    enable_verbose_llm_logging: bool = False  # Capture full prompts/responses
    enable_decision_logging: bool = True  # Log decision reasoning
    enable_context_tracking: bool = True  # Track context flow
```

### 2.2 Integrate LLMCallLogger with Synthesis

In `synthesizer.py`:

```python
async def synthesize_with_content(self, ...):
    async with self.llm_logger.track_call(
        agent_name="synthesizer",
        operation=LLMOperation.SYNTHESIS,
        model=model_name,
        prompt=full_prompt,
        prompt_template="synthesis_with_content"
    ) as call:
        response = await self._call_ollama(full_prompt, model_name)

        call.finalize(
            response=response,
            parse_success=True
        )

        # Store response for verbose logging
        if self.llm_logger.verbose:
            call.full_response = response

        return response
```

### 2.3 Add Debug Endpoint for Live Queries

```python
# In api/search.py
@router.post("/api/v1/search/debug")
async def search_with_debug(
    request: SearchRequest,
    verbose: bool = Query(default=True)
):
    """
    Execute search with full debugging output.
    Returns complete LLM prompts/responses and decision traces.
    """
    request.options = request.options or {}
    request.options["verbose_logging"] = verbose

    # ... execute search ...

    return {
        "success": True,
        "data": response,
        "debug": {
            "llm_calls": llm_logger.export_for_debugging(),
            "decisions": decision_logger.export_for_debugging(),
            "context_flow": context_tracker.get_flow_summary()
        }
    }
```

---

## Phase 3: unified_dashboard Integration (Priority P2)

### 3.1 Verify Backend Routes

The unified_dashboard backend (`server/routes/agent.ts`) already has correct endpoints:

```typescript
// These are correctly configured
GET /api/agent/stream/global -> polls memOS /api/v1/observability/recent
GET /api/agent/events/:requestId -> SSE proxy to memOS /api/v1/search/events/:requestId
GET /api/agent/observability/:requestId -> memOS /api/v1/observability/request/:requestId
```

### 3.2 Test Data Flow

After wiring observability:

```bash
# 1. Start memOS server
cd /home/sparkone/sdd/Recovery_Bot/memOS/server
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001

# 2. Execute a test search
curl -X POST "http://localhost:8001/api/v1/search/chat-gateway" \
  -H "Content-Type: application/json" \
  -d '{"query": "SRVO-063 encoder error", "preset": "enhanced"}'

# 3. Verify observability data
curl "http://localhost:8001/api/v1/observability/recent?limit=5"

# 4. Start unified_dashboard
cd /home/sparkone/sdd/unified_dashboard
npm run dev

# 5. Open http://localhost:3100 and check Agent Console
```

### 3.3 Enhanced Agent Console Features

The AgentConsole component already supports:
- Live event feed via polling/SSE
- Run history display
- Event detail expansion
- Test query execution

Additional features to enable:
- Full LLM prompt/response viewer (when verbose mode enabled)
- Decision tree visualization
- Context flow diagram
- Confidence breakdown chart

---

## Implementation Checklist

### Phase 1: Wire Observability (Estimated: 2-3 hours)

- [ ] Add imports to `orchestrator_universal.py`
- [ ] Initialize loggers in `search_with_events()`
- [ ] Add decision logging at 10+ key decision points:
  - [ ] Query classification
  - [ ] CRAG evaluation
  - [ ] Self-RAG reflection
  - [ ] DyLAN complexity
  - [ ] Feature enable/skip
  - [ ] Model selection
  - [ ] Synthesis strategy
  - [ ] Confidence calculation
  - [ ] Refinement decision
  - [ ] Halt decision
- [ ] Add context tracking between all pipeline stages
- [ ] Aggregate and store in ObservabilityDashboard
- [ ] Test with sample queries

### Phase 2: Verbose Logging (Estimated: 1-2 hours)

- [ ] Add verbose_llm_logging to FeatureConfig
- [ ] Integrate LLMCallLogger into synthesizer
- [ ] Integrate LLMCallLogger into analyzer
- [ ] Integrate LLMCallLogger into verifier
- [ ] Add /api/v1/search/debug endpoint
- [ ] Test full prompt/response capture

### Phase 3: Dashboard Testing (Estimated: 1 hour)

- [ ] Start unified_dashboard
- [ ] Execute test searches
- [ ] Verify Agent Console shows real data
- [ ] Test event streaming
- [ ] Test history retrieval
- [ ] Document any frontend fixes needed

---

## Data Flow Diagram

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
│    └─► GET /api/v1/observability/request/{id}/audit  ► Technician summary   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     unified_dashboard (Port 3100/3101)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Express Backend (server/routes/agent.ts)                                    │
│    │                                                                         │
│    ├─► GET /api/agent/stream/global  ───► Polls memOS /observability/recent │
│    ├─► GET /api/agent/history        ───► Returns stored run history        │
│    ├─► POST /api/agent/search        ───► Initiates search via memOS        │
│    └─► GET /api/agent/observability/:id ► Gets full request details         │
│                                                                              │
│  React Frontend (AgentConsole.tsx)                                           │
│    │                                                                         │
│    ├─► Live Feed Tab  ──────────────────► Real-time event stream            │
│    ├─► Run History Tab  ────────────────► Past search results               │
│    └─► Event Detail Expansion  ─────────► JSON data viewer                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User's Required Debugging Data

The user requested visibility into:

| Requirement | Solution | Component |
|-------------|----------|-----------|
| LLM inputs/outputs | LLMCallLogger with verbose mode | `llm_logger.py` |
| NLP technique decisions | DecisionLogger for each agent | `decision_logger.py` |
| Search reasoning | Decision logging in Analyzer | `decision_logger.py` |
| Analyzer conclusions | Query analysis SSE events | `events.py` |
| HyDE computations | HyDE-specific SSE events | `events.py` |
| Website searches | Web search SSE events | `events.py` |
| All decision-making data | ObservabilityDashboard aggregation | `observability_dashboard.py` |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| `/api/v1/observability/recent` returns data | After any search |
| Decision count per request | 10+ decisions logged |
| LLM calls tracked | 100% of calls |
| Context transfers tracked | All inter-stage flows |
| unified_dashboard Agent Console | Shows real-time data |
| Full prompt/response capture | When verbose=true |

---

## Files to Modify

| File | Changes |
|------|---------|
| `agentic/orchestrator_universal.py` | Add observability imports, initialize loggers, integrate throughout pipeline, aggregate at end |
| `agentic/synthesizer.py` | Integrate LLMCallLogger |
| `agentic/analyzer.py` | Integrate LLMCallLogger |
| `agentic/retrieval_evaluator.py` | Add decision logging |
| `agentic/self_reflection.py` | Add decision logging |
| `api/search.py` | Add `/search/debug` endpoint |

---

*Plan created by Claude Code | 2026-01-06*
