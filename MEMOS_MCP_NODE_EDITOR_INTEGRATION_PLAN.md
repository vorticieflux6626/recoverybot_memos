# memOS + MCP_Node_Editor Integration Plan

> **Created**: 2026-01-25 | **Version**: 1.1.0 | **Status**: Phase 0 Complete
> **Last Updated**: 2026-01-25

## Executive Summary

This document outlines the comprehensive plan for integrating the memOS agentic search pipeline with MCP_Node_Editor, enabling visual configuration, runtime path visualization, and debugging through a unified Agent Console on the unified_dashboard.

**Key Findings:**
- ~~MCP_Node_Editor has a **critical scheduler deadlock** that must be resolved first~~ ✅ FIXED
- memOS has **72 boolean + 9 non-boolean feature flags** across 4 layers
- The unified_dashboard already has an Agent Console with SSE - needs enhancement
- Both architectures are **highly compatible** (event-driven async)

## Phase 0 Completion Summary (2026-01-25)

All Phase 0 blockers have been resolved:

| Fix | File | Status |
|-----|------|--------|
| Scheduler Deadlock | `node_execution_scheduler.py` | ✅ Complete |
| Bootstrap Injection | `complete_reactive_executor.py` | ✅ Complete |
| Async Blocking Patterns | `ecosystem_health.py` | ✅ Complete |
| Cycle Detection Hardening | `cycle_detector.py` | ✅ Complete |

**Tests Passed**: All syntax checks, import tests, and functionality tests verified.

---

## Phase 0: MCP_Node_Editor Blockers (Priority: CRITICAL) ✅ COMPLETE

**All blockers resolved on 2026-01-25.**

### 0.1 Scheduler Deadlock Fix ✅

**Root Cause Identified**: Race condition in `node_execution_scheduler.py:393-404`

The deadlock occurs because:
1. Line 393 takes a snapshot: `current_executing = set(self.executing_nodes)`
2. Node's `finally` block removes from `executing_nodes` AFTER snapshot
3. Line 404 checks against STALE snapshot → node rejected for re-scheduling
4. Result: Cycles achieve 0 iterations

**Fix Applied** (`node_execution_scheduler.py`):

```python
# Replace snapshot pattern with atomic checks
async def _pre_calculate_executable_tasks(self, available_slots: int) -> List[ExecutionTask]:
    executable_tasks = []
    temp_queue = []
    checked_nodes = set()

    # Use async lock to ensure atomicity
    async with self._resource_lock:
        while self.execution_queue and len(executable_tasks) < available_slots:
            task = heapq.heappop(self.execution_queue)

            if task.node_id in checked_nodes:
                temp_queue.append(task)
                continue

            # ATOMIC check against LIVE set (not snapshot)
            not_executing = task.node_id not in self.executing_nodes

            if not_executing and other_conditions:
                self.executing_nodes.add(task.node_id)
                checked_nodes.add(task.node_id)
                executable_tasks.append(task)
            else:
                temp_queue.append(task)

        for task in temp_queue:
            heapq.heappush(self.execution_queue, task)

    return executable_tasks
```

**Additional Fixes Applied:**
- ✅ Allow re-scheduling for cycle nodes (bypass duplicate task prevention)
- ✅ Actually inject bootstrap values into buffers after generating
- ✅ Force cycle entry node state from WAITING to READY when buffer is ready

**Files Modified:**
- `/MCP_Node_Editor/mcp-servers/node_execution_scheduler.py`
- `/MCP_Node_Editor/mcp-servers/complete_reactive_executor.py`

**Actual Effort**: Completed in 1 session

---

### 0.2 Async Pattern Fixes ✅

**Blocking Operations Found and Fixed:**

| File | Lines | Issue | Impact | Status |
|------|-------|-------|--------|--------|
| `ecosystem_health.py` | 133, 139, 145 | `subprocess.run()` | 5-15s event loop block | ✅ Fixed |
| `pipeline_launcher.py` | 607, 675, 725, 857 | Blocking file I/O | Variable block time | ⚠️ Low priority |
| `test_2_banger_loop_deadlock.py` | 40, 60, 224 | `requests` library | Test blocking | ⚠️ Test file only |

**Fix Applied for ecosystem_health.py:**
- Converted `check_database_health` from blocking `subprocess.run` to async `asyncio.create_subprocess_exec`
- Updated `check_all_services` to use `asyncio.gather` for parallel database checks
- Added `await` to call in `check_single_service`

**Actual Effort**: Completed in 1 session

---

### 0.3 Cycle Detection Hardening ✅

**Issues Found and Fixed:**
- ✅ No exponential backoff → Added with base 0.1s, max 30s, 2x multiplier
- ✅ Unbounded memory growth → Added limits (MAX_NODES_TRACKED=1000, MAX_DETECTED_CYCLES=500)
- ⚠️ No jitter in coordination delays → Deferred (low priority)
- ✅ Cycle recurrence tracking → Added `cycle_recurrence` dict for backoff state

**Fixes Applied to `cycle_detector.py`:**
```python
# Memory limits constants
MAX_NODES_TRACKED = 1000
MAX_DETECTED_CYCLES = 500
MAX_EXECUTION_TIMES_PER_NODE = 200

# Exponential backoff
BACKOFF_BASE_DELAY = 0.1  # seconds
BACKOFF_MAX_DELAY = 30.0  # seconds
BACKOFF_MULTIPLIER = 2.0

# Methods added
def _enforce_memory_limits(self): ...
def get_backoff_delay(self, cycle_info): ...
def reset_backoff(self, cycle_info): ...
```

**Estimated Effort**: 1-2 days

---

## Phase 1: Core Agent Nodes (Week 1-2)

### 1.1 Node Mapping

| memOS Agent | MCP Node Type | Inputs | Outputs | Priority |
|-------------|---------------|--------|---------|----------|
| QueryAnalyzer | `query_analyzer` | query, context | QueryAnalysis | P0 |
| ResultSynthesizer | `synthesizer` | query, results, content, context | synthesis | P0 |
| SelfReflectionAgent | `self_reflection` | query, synthesis, sources | ReflectionResult, refined | P0 |
| RetrievalEvaluator | `retrieval_evaluator` | query, results, questions | Evaluation, queries | P1 |
| HyDEExpander | `hyde_expander` | query | HyDEResult, embedding | P1 |

### 1.2 Node Implementation Pattern

```python
# In mcp-servers/reactive_node_implementations.py
async def _execute_query_analyzer_node(self, node: Dict, inputs: Dict) -> Dict[int, Any]:
    """Wrap memOS QueryAnalyzer as MCP node."""
    query = inputs.get(0, "")
    context = inputs.get(1, {})
    props = node.get("properties", {})

    # Import memOS agent
    from agentic.analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer(
        model=self._get_node_property(props, "model", "qwen3:8b"),
        enable_acronym_expansion=self._get_node_property(props, "enable_acronym_expansion", True)
    )

    analysis = await analyzer.analyze(
        query=query,
        context=context,
        request_id=f"node_{node['id']}",
        use_gateway=self._get_node_property(props, "use_gateway", False)
    )

    # Yield control to event loop
    await asyncio.sleep(0)

    return {0: analysis.to_dict()}
```

### 1.3 Frontend Node Templates

```javascript
// In pipeline-editor-enhanced.html
node_templates.query_analyzer = {
    title: 'Query Analyzer',
    inputs: [
        { name: 'query', type: 'text' },
        { name: 'context', type: 'dict' }
    ],
    outputs: [
        { name: 'analysis', type: 'dict' }
    ],
    properties: {
        model: { type: 'select', value: 'qwen3:8b', options: ['qwen3:8b', 'gemma3:4b'] },
        use_gateway: { type: 'checkbox', value: false },
        enable_acronym_expansion: { type: 'checkbox', value: true }
    },
    color: '#2C5282'  // Navy
};
```

**Estimated Effort**: 5-7 days

---

## Phase 2: Feature Flag UI (Week 2-3)

### 2.1 Feature Flag Inventory

**Total: 72 boolean + 9 non-boolean flags**

| Layer | Boolean Flags | Non-Boolean | Examples |
|-------|---------------|-------------|----------|
| Layer 0: Core | 4 | 0 | query_analysis, verification, scratchpad, metrics |
| Layer 1: Quality | 7 | 2 | self_reflection, crag_evaluation, min_confidence_threshold |
| Layer 2: Performance | 19 | 1 | hyde, hybrid_reranking, cross_encoder, context_curation_preset |
| Layer 3: Reasoning | 29 | 5 | entity_tracking, technical_docs, technical_traversal_mode |
| Layer 4: Multi-Agent | 10 | 1 | dynamic_planning, dylan_agent_skipping, ib_filtering_level |

### 2.2 Preset → Flag Mapping

| Preset | Flags Enabled | Typical Latency | Use Case |
|--------|---------------|-----------------|----------|
| MINIMAL | 8 | 30-60s | Simple queries |
| BALANCED | 18 | 90-150s | General use (DEFAULT) |
| ENHANCED | 28 | 150-240s | Technical troubleshooting |
| RESEARCH | 35 | 300-420s | Academic research |
| FULL | 38+ | 450-600s | Maximum capability |

### 2.3 UI Component Design

**MCP_Node_Editor Settings Tab Extension:**

```html
<div class="settings-section">
    <h3>memOS Agentic Search Configuration</h3>

    <!-- Preset Selector -->
    <div class="preset-selector">
        <label>Preset:</label>
        <select id="agentic-preset" onchange="applyPreset(this.value)">
            <option value="MINIMAL">MINIMAL (8 features)</option>
            <option value="BALANCED" selected>BALANCED (18 features)</option>
            <option value="ENHANCED">ENHANCED (28 features)</option>
            <option value="RESEARCH">RESEARCH (35 features)</option>
            <option value="FULL">FULL (38+ features)</option>
        </select>
    </div>

    <!-- Collapsible Layer Sections -->
    <div class="layer-section" data-layer="1">
        <h4 onclick="toggleLayer(1)">Layer 1: Quality Control</h4>
        <div class="layer-content">
            <label><input type="checkbox" id="enable_self_reflection" checked> Self-Reflection</label>
            <label><input type="checkbox" id="enable_crag_evaluation" checked> CRAG Evaluation</label>
            <!-- ... -->
        </div>
    </div>
    <!-- Repeat for Layers 2-4 -->
</div>
```

### 2.4 Dependency Validation

```javascript
const FLAG_DEPENDENCIES = {
    'enable_hsea_context': ['enable_domain_corpus'],
    'enable_symptom_entry': ['enable_technical_docs'],
    'enable_structured_causal_chain': ['enable_technical_docs'],
    'enable_circuit_diagrams': ['enable_technical_docs'],
    'enable_harness_diagrams': ['enable_technical_docs'],
    'enable_pinout_diagrams': ['enable_technical_docs'],
    'enable_entity_grounding': ['enable_technical_docs'],
    'enable_cross_domain_validation': ['enable_technical_docs'],
    'enable_auto_diagram_generation': ['enable_circuit_diagrams', 'enable_harness_diagrams', 'enable_pinout_diagrams']  // OR dependency
};

function validateDependencies(enabledFlags) {
    const errors = [];
    for (const [flag, deps] of Object.entries(FLAG_DEPENDENCIES)) {
        if (enabledFlags.includes(flag)) {
            const missingDeps = deps.filter(d => !enabledFlags.includes(d));
            if (missingDeps.length > 0) {
                errors.push(`${flag} requires: ${missingDeps.join(', ')}`);
            }
        }
    }
    return errors;
}
```

**Estimated Effort**: 4-5 days

---

## Phase 3: Runtime Integration (Week 3-4)

### 3.1 memOS Observability Infrastructure

memOS already captures comprehensive observability data:

| Module | Data Captured | Access |
|--------|---------------|--------|
| DecisionLogger | Agent decisions, reasoning, alternatives, confidence | `decisions[]` |
| ContextTracker | Context transfers, token counts, bottlenecks | `context_transfers[]` |
| LLMCallLogger | LLM invocations, latency, tokens, parse success | `llm_calls[]` |
| ScratchpadObserver | Findings, questions, entities, gaps | `scratchpad_changes[]` |
| ConfidenceLogger | Multi-signal confidence breakdown | `confidence_breakdown` |

### 3.2 Data Flow Architecture

```
memOS Observability System
    ↓ SSE Events
Integration Bridge Layer (NEW)
    ↓ WebSocket
MCP_Node_Editor / unified_dashboard
    ↓ DOM Updates
Visual State Inspector UI
```

### 3.3 New API Endpoints

**WebSocket Event Stream:**
```
WebSocket /api/v1/observability/stream/{request_id}
```

**State Snapshot:**
```
GET /api/v1/observability/state/{request_id}
```

**Historical Replay:**
```
GET /api/v1/observability/replay/{request_id}?from=...&to=...
```

### 3.4 UI Components for State Inspector

| Component | Data Source | Update Trigger |
|-----------|-------------|----------------|
| Agent Execution Timeline | llm_calls timestamps | AGENT_START, AGENT_COMPLETE |
| Decision Tree Visualization | decisions[] | DECISION_POINT |
| Context Flow Diagram | context_transfers[] | CONTEXT_TRANSFER |
| LLM Call Timeline | llm_calls[] | LLM_CALL_START, LLM_CALL_COMPLETE |
| Scratchpad State Panel | scratchpad_changes[] | SCRATCHPAD_UPDATED |
| Confidence Breakdown | confidence_breakdown | CONFIDENCE_CALCULATED |

**Estimated Effort**: 6-8 days

---

## Phase 4: unified_dashboard Agent Console Enhancement (Week 4-5)

### 4.1 Current State

The unified_dashboard Agent Console already has:
- SSE streaming for live events
- Run history (database-backed)
- LLM model configuration panel
- Basic event display

### 4.2 Enhancement Plan

**New Tabs to Add:**

1. **Decision Log Tab** - Timeline with reasoning
2. **Context Flow Tab** - Sankey diagram of token flow
3. **LLM Calls Tab** - Performance metrics
4. **Scratchpad Tab** - State evolution
5. **Confidence Tab** - Signal breakdown

### 4.3 Backend API Enhancement

Add to `server/routes/agent.ts`:

```typescript
// Proxy to memOS observability endpoints
router.get('/observability/:requestId', async (req, res) => {
    const response = await fetch(
        `${MEMOS_BASE_URL}/api/v1/observability/request/${req.params.requestId}`
    );
    const data = await response.json();
    res.json(data);
});

router.get('/observability/:requestId/decisions', async (req, res) => {
    // Extract decisions[] from full observability data
});

router.get('/observability/:requestId/context-flow', async (req, res) => {
    // Extract context_transfers[] with Sankey diagram formatting
});

router.get('/observability/:requestId/llm-calls', async (req, res) => {
    // Extract llm_calls[] with sorting options
});

router.get('/observability/:requestId/scratchpad', async (req, res) => {
    // Extract scratchpad_changes[] with current state
});
```

### 4.4 Frontend Components

Create in `src/components/agent/`:

```
agent/
├── AgentConsole.tsx (enhance existing)
├── tabs/
│   ├── OverviewTab.tsx
│   ├── DecisionLogTab.tsx
│   ├── ContextFlowTab.tsx
│   ├── LLMCallsTab.tsx
│   ├── ScratchpadTab.tsx
│   └── ConfidenceTab.tsx
├── components/
│   ├── DecisionTimeline.tsx
│   ├── DecisionCard.tsx
│   ├── ContextFlowDiagram.tsx (ECharts Sankey)
│   ├── LLMCallCard.tsx
│   ├── ScratchpadStatePanel.tsx
│   └── ConfidenceBreakdown.tsx
└── hooks/
    ├── useDecisions.ts
    ├── useContextFlow.ts
    ├── useLLMCalls.ts
    └── useScratchpad.ts
```

**Estimated Effort**: 7-10 days

---

## Phase 5: Advanced Features (Week 5-6)

### 5.1 Time-Travel Debugging

**Implementation:**
- Checkpoint schema: Phase, scratchpad state, LLM calls, confidence
- Storage: PostgreSQL (metadata) + Redis (hot) + S3 (cold)
- Forking: Create new thread_id, resume from modified state

**Priority**: P1 (5-7 days)

### 5.2 A/B Testing

**Implementation:**
- Parallel execution via `asyncio.gather`
- Metrics: Quality (confidence, citations), Latency, Cost (tokens)
- Comparison endpoint: `/api/v1/search/compare`

**Priority**: P1 (4-6 days)

### 5.3 Configuration Sharing

**Implementation:**
- YAML schema for presets
- Import/export functionality
- Version control (semantic versioning)
- Validation framework

**Priority**: P2 (3-5 days)

### 5.4 Visual Debugging

**Implementation:**
- OpenTelemetry instrumentation
- AgentPrism-style trace visualization
- Real-time SSE streaming

**Priority**: P1 (6-9 days)

---

## Summary Timeline

| Phase | Description | Duration | Dependencies |
|-------|-------------|----------|--------------|
| **Phase 0** | MCP_Node_Editor Blockers | 4-7 days | None |
| **Phase 1** | Core Agent Nodes | 5-7 days | Phase 0 |
| **Phase 2** | Feature Flag UI | 4-5 days | Phase 1 |
| **Phase 3** | Runtime Integration | 6-8 days | Phase 1, 2 |
| **Phase 4** | Dashboard Enhancement | 7-10 days | Phase 3 |
| **Phase 5** | Advanced Features | 18-27 days | Phase 4 |

**Total Estimated Time**: 6-9 weeks

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scheduler deadlock persists | Medium | Critical | Isolate fix, extensive testing |
| Async pattern violations | Low | High | Code review, linting rules |
| Event volume overwhelms UI | Medium | Medium | Client-side filtering, pagination |
| Feature flag complexity | Medium | Medium | Preset-first approach, validation |
| Integration latency | Low | Medium | Caching, background processing |

---

## Success Metrics

1. **Scheduler Deadlock**: Cycle tests achieve 50+ iterations (currently 0)
2. **Feature Flag UI**: All 72 flags accessible with dependency validation
3. **Runtime Visualization**: <1s latency from event emission to UI update
4. **Agent Console**: 6 new tabs with full observability coverage
5. **A/B Testing**: Compare 2 presets in <5 minutes (parallel execution)

---

## Files to Create/Modify

### MCP_Node_Editor
- `mcp-servers/node_execution_scheduler.py` - Deadlock fix
- `mcp-servers/complete_reactive_executor.py` - Bootstrap injection
- `mcp-servers/ecosystem_health.py` - Async subprocess
- `mcp-servers/pipeline_launcher.py` - Async file I/O
- `mcp-servers/reactive_node_implementations.py` - memOS agent nodes
- `pipeline-editor-enhanced.html` - Node templates, feature flag UI

### memOS
- `server/api/observability_stream.py` - WebSocket endpoint (NEW)
- `server/agentic/event_adapter.py` - Event transformation (NEW)
- `config/feature_flags_schema.json` - JSON Schema (NEW)

### unified_dashboard
- `server/routes/agent.ts` - Observability proxy endpoints
- `src/components/agent/` - 6 new tab components
- `src/lib/api.ts` - TypeScript types for observability

---

*Last Updated: 2026-01-25*
