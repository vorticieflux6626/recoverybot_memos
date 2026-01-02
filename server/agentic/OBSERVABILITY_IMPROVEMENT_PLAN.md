# Agentic Pipeline Observability Improvement Plan

> **Created**: 2026-01-02 | **Status**: Planning | **Version**: 1.0

## Executive Summary

Based on comprehensive research into agentic AI observability best practices and a full audit of the memOS pipeline, this document outlines a structured plan to implement end-to-end observability for robotics technician debugging and system engineering.

### Key Research Findings

| Research Area | Key Insight |
|---------------|-------------|
| **Agentic AI Observability** | OpenTelemetry GenAI semantic conventions now standard; memOS has infrastructure but not connected |
| **Industrial AI Explainability** | Technicians need layered explanations with confidence visualization, not ML metrics |
| **Orchestrator Audit** | 95+ SSE events (excellent), but OpenTelemetry spans NOT used in orchestrator |
| **Agent Audit** | LLM prompts/responses not logged; decision reasoning largely invisible |

### Current State Assessment

| Component | Status | Gap |
|-----------|--------|-----|
| SSE Events | ✅ Excellent (95+ types) | Minor - add reasoning events |
| OpenTelemetry | ⚠️ Infrastructure only | Critical - not connected to orchestrator |
| Logging | ⚠️ Inconsistent | High - decision points missing |
| Metrics | ✅ Good (timing/cache) | Medium - no reasoning metrics |
| Scratchpad | ⚠️ Partial | Medium - state changes not tracked |

---

## Architecture: Three-Tier Observability

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: TECHNICIAN VIEW                               │
│  Human-readable summaries, confidence bars, source citations, safety warnings│
│  Format: Markdown/HTML for Android display                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                        TIER 2: ENGINEERING VIEW                              │
│  Full reasoning trace, agent decisions, context flows, timing               │
│  Format: Structured JSON logs + SSE events                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                        TIER 3: DEBUG VIEW                                    │
│  Token counts, LLM prompts/responses, embeddings, cache internals           │
│  Format: OpenTelemetry spans + detailed logs (opt-in)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Decision Point Logging (P0 - Immediate)

### 1.1 Agent Decision Logger

Create a centralized decision logging system that captures:

```python
# agentic/decision_logger.py

@dataclass
class AgentDecision:
    """Structured logging for any agent decision point."""
    request_id: str
    agent_name: str  # analyzer, synthesizer, verifier, crag, self_rag
    decision_type: str  # classification, action, skip, fallback, refinement
    decision_made: str  # The actual decision
    reasoning: str  # Why this decision was made
    alternatives_considered: List[str]  # What else was considered
    confidence: float  # 0-1 confidence in decision
    context_size_tokens: int  # Input context size
    timestamp: datetime

    # Decision-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class DecisionLogger:
    """Centralized decision logging with SSE emission."""

    def __init__(self, emitter: Optional[EventEmitter] = None):
        self.emitter = emitter
        self.decisions: List[AgentDecision] = []

    async def log_decision(self, decision: AgentDecision):
        # 1. Structured log
        logger.info(
            f"[{decision.request_id}] DECISION: {decision.agent_name}.{decision.decision_type} "
            f"→ {decision.decision_made} (conf={decision.confidence:.2f})",
            extra={
                "decision": asdict(decision),
                "structured": True
            }
        )

        # 2. SSE event for real-time visibility
        if self.emitter:
            await self.emitter.emit(SearchEvent(
                event_type=EventType.DECISION_POINT,
                request_id=decision.request_id,
                data={
                    "agent": decision.agent_name,
                    "decision_type": decision.decision_type,
                    "decision": decision.decision_made,
                    "reasoning": decision.reasoning[:200],
                    "confidence": decision.confidence
                }
            ))

        # 3. Store for request summary
        self.decisions.append(decision)
```

### 1.2 Integration Points

Add decision logging at these critical points:

| Agent | Decision Point | What to Log |
|-------|----------------|-------------|
| **QueryClassifier** | Pipeline routing | `direct_answer` vs `web_search` vs `agentic_search`, reasoning |
| **Analyzer** | Complexity assessment | LOW/MEDIUM/HIGH, why, feature recommendations |
| **Analyzer** | Thinking model requirement | True/False, reasoning complexity score |
| **CRAG** | Quality classification | CORRECT/AMBIGUOUS/INCORRECT, relevance score, threshold comparison |
| **CRAG** | Corrective action | PROCEED/REFINE/FALLBACK/DECOMPOSE, why |
| **Self-RAG** | Refinement decision | needs_refinement True/False, ISREL/ISSUP/ISUSE scores |
| **Synthesizer** | Model selection | Which model, why (thinking override, context size) |
| **DyLAN** | Agent skip | Which agent skipped, complexity classification |
| **Entropy Monitor** | Halt decision | CONTINUE/HALT_CONFIDENT/HALT_MAX, entropy score |

### 1.3 Feature Skip Logging

Currently features are silently skipped when disabled. Add explicit logging:

```python
# In orchestrator_universal.py

def _log_feature_status(self, feature_name: str, enabled: bool, reason: str = ""):
    """Log why a feature was enabled/disabled."""
    if enabled:
        logger.debug(f"[{self.request_id}] Feature ENABLED: {feature_name}")
    else:
        logger.info(
            f"[{self.request_id}] Feature SKIPPED: {feature_name} "
            f"(reason: {reason or 'disabled in config'})"
        )

# Usage in orchestrator:
if self.config.enable_crag_evaluation:
    self._log_feature_status("crag_evaluation", True)
    # ... execute CRAG
else:
    self._log_feature_status("crag_evaluation", False,
                            f"preset={self.preset.value}, enable_crag_evaluation=False")
```

---

## Phase 2: Context Flow Tracking (P0 - Immediate)

### 2.1 Context Transfer Logger

Track what context flows between agents:

```python
# agentic/context_tracker.py

@dataclass
class ContextTransfer:
    """Track context as it flows between pipeline stages."""
    request_id: str
    source_stage: str  # analyzer, searcher, scraper, crag, synthesizer, etc.
    target_stage: str
    context_type: str  # query, search_results, scraped_content, findings, synthesis
    token_count: int
    char_count: int
    item_count: int  # Number of sources, findings, etc.
    content_hash: str  # For deduplication detection
    timestamp: datetime

class ContextFlowTracker:
    """Track all context transfers in a request."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.transfers: List[ContextTransfer] = []

    def record_transfer(
        self,
        source: str,
        target: str,
        content: Any,
        context_type: str
    ):
        # Calculate metrics
        if isinstance(content, str):
            char_count = len(content)
            item_count = 1
        elif isinstance(content, list):
            char_count = sum(len(str(c)) for c in content)
            item_count = len(content)
        else:
            char_count = len(str(content))
            item_count = 1

        token_count = char_count // 4  # Rough estimate

        transfer = ContextTransfer(
            request_id=self.request_id,
            source_stage=source,
            target_stage=target,
            context_type=context_type,
            token_count=token_count,
            char_count=char_count,
            item_count=item_count,
            content_hash=hashlib.md5(str(content).encode()).hexdigest()[:8],
            timestamp=datetime.now()
        )

        self.transfers.append(transfer)

        logger.info(
            f"[{self.request_id}] Context: {source} → {target} | "
            f"type={context_type} | items={item_count} | ~{token_count} tokens"
        )

    def get_flow_summary(self) -> Dict:
        """Generate flow summary for debugging."""
        return {
            "total_transfers": len(self.transfers),
            "total_tokens_transferred": sum(t.token_count for t in self.transfers),
            "flow_path": [f"{t.source_stage}→{t.target_stage}" for t in self.transfers],
            "by_type": self._group_by_type()
        }
```

### 2.2 Integration in Orchestrator

```python
# In orchestrator_universal.py

async def search_with_events(self, request, emitter):
    # Initialize context tracker
    context_tracker = ContextFlowTracker(request.id)

    # After analysis
    analysis = await self.analyzer.analyze(request.query)
    context_tracker.record_transfer(
        source="input",
        target="analyzer",
        content=request.query,
        context_type="query"
    )
    context_tracker.record_transfer(
        source="analyzer",
        target="planner",
        content=analysis,
        context_type="query_analysis"
    )

    # After search
    results = await self.searcher.search(queries)
    context_tracker.record_transfer(
        source="searcher",
        target="scraper",
        content=results,
        context_type="search_results"
    )

    # ... and so on through the pipeline

    # Include in final response
    response.meta["context_flow"] = context_tracker.get_flow_summary()
```

---

## Phase 3: LLM Call Instrumentation (P1 - This Week)

### 3.1 LLM Call Logger

Create comprehensive LLM call logging:

```python
# agentic/llm_logger.py

@dataclass
class LLMCall:
    """Complete record of an LLM invocation."""
    request_id: str
    agent_name: str
    operation: str  # analysis, synthesis, verification, evaluation, reflection
    model: str

    # Input
    prompt_template: str  # Template name or identifier
    prompt_length_chars: int
    prompt_length_tokens: int  # Estimated
    input_context_items: int  # Number of sources, findings, etc.

    # Output
    response_length_chars: int
    response_length_tokens: int
    response_truncated: bool

    # Timing
    start_time: datetime
    end_time: datetime
    latency_ms: int
    time_to_first_token_ms: Optional[int]

    # Quality
    parse_success: bool  # Did JSON/structured output parse?
    fallback_used: bool  # Did we fall back to different model/provider?

    # Debug (opt-in)
    full_prompt: Optional[str] = None  # Only when debug enabled
    full_response: Optional[str] = None  # Only when debug enabled

class LLMCallLogger:
    """Log all LLM calls with optional verbose mode."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.calls: List[LLMCall] = []

    @asynccontextmanager
    async def track_call(
        self,
        request_id: str,
        agent_name: str,
        operation: str,
        model: str,
        prompt: str,
        prompt_template: str = "custom"
    ):
        """Context manager for tracking LLM calls."""
        start_time = datetime.now()

        call = LLMCall(
            request_id=request_id,
            agent_name=agent_name,
            operation=operation,
            model=model,
            prompt_template=prompt_template,
            prompt_length_chars=len(prompt),
            prompt_length_tokens=len(prompt) // 4,
            input_context_items=prompt.count("[Source"),
            start_time=start_time,
            end_time=start_time,  # Updated later
            latency_ms=0,
            response_length_chars=0,
            response_length_tokens=0,
            response_truncated=False,
            parse_success=True,
            fallback_used=False,
            full_prompt=prompt if self.verbose else None
        )

        try:
            yield call
        finally:
            call.end_time = datetime.now()
            call.latency_ms = int((call.end_time - call.start_time).total_seconds() * 1000)

            # Log summary
            logger.info(
                f"[{request_id}] LLM: {agent_name}.{operation} | "
                f"model={model} | in={call.prompt_length_tokens}tok | "
                f"out={call.response_length_tokens}tok | {call.latency_ms}ms"
            )

            if self.verbose:
                logger.debug(f"[{request_id}] LLM PROMPT:\n{prompt[:1000]}...")
                if call.full_response:
                    logger.debug(f"[{request_id}] LLM RESPONSE:\n{call.full_response[:1000]}...")

            self.calls.append(call)
```

### 3.2 Integration in Agents

```python
# In synthesizer.py

async def synthesize_with_content(self, ...):
    async with self.llm_logger.track_call(
        request_id=request_id,
        agent_name="synthesizer",
        operation="synthesis",
        model=model_name,
        prompt=full_prompt,
        prompt_template="synthesis_with_content"
    ) as call:
        response = await self._call_ollama(full_prompt, model_name)

        call.response_length_chars = len(response)
        call.response_length_tokens = len(response) // 4
        call.full_response = response if self.llm_logger.verbose else None

        return response
```

---

## Phase 4: Scratchpad State Tracking (P1 - This Week)

### 4.1 Scratchpad Observer

Track all state changes in the scratchpad:

```python
# In scratchpad.py

class ScratchpadObserver:
    """Observe and log all scratchpad state changes."""

    def __init__(self, scratchpad: AgenticScratchpad, emitter: Optional[EventEmitter] = None):
        self.scratchpad = scratchpad
        self.emitter = emitter
        self.change_log: List[Dict] = []

    async def record_change(
        self,
        operation: str,  # add_finding, update_question, add_entity, write_public, add_note
        agent: str,
        details: Dict
    ):
        change = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "agent": agent,
            "details": details,
            "scratchpad_state": self._get_summary()
        }

        self.change_log.append(change)

        logger.info(
            f"[{self.scratchpad.request_id}] Scratchpad: {agent}.{operation} | "
            f"findings={len(self.scratchpad.findings)} | "
            f"questions={self._count_answered()}/{len(self.scratchpad.questions)}"
        )

        if self.emitter:
            await self.emitter.emit(SearchEvent(
                event_type=EventType.SCRATCHPAD_UPDATED,
                request_id=self.scratchpad.request_id,
                data=change
            ))

    def _get_summary(self) -> Dict:
        return {
            "findings_count": len(self.scratchpad.findings),
            "questions_answered": self._count_answered(),
            "questions_total": len(self.scratchpad.questions),
            "entities_count": len(self.scratchpad.entities),
            "contradictions": len([f for f in self.scratchpad.findings.values() if f.conflicts_with]),
            "gaps_detected": self._count_gaps()
        }
```

### 4.2 Wrap Scratchpad Operations

```python
# Enhanced scratchpad.py methods

def add_finding(self, finding: Finding, agent: str = "unknown"):
    """Add finding with observability."""
    self.findings[finding.id] = finding

    # Log the change
    if self.observer:
        asyncio.create_task(self.observer.record_change(
            operation="add_finding",
            agent=agent,
            details={
                "finding_id": finding.id,
                "content_preview": finding.content[:100],
                "confidence": finding.confidence,
                "source_count": len(finding.sources),
                "conflicts_with": finding.conflicts_with
            }
        ))

def update_question_status(self, question_id: str, status: QuestionStatus, agent: str = "unknown"):
    """Update question status with observability."""
    if question_id in self.questions:
        old_status = self.questions[question_id].status
        self.questions[question_id].status = status

        if self.observer:
            asyncio.create_task(self.observer.record_change(
                operation="update_question",
                agent=agent,
                details={
                    "question_id": question_id,
                    "old_status": old_status.value,
                    "new_status": status.value,
                    "question": self.questions[question_id].question_text[:50]
                }
            ))
```

---

## Phase 5: Technician-Friendly Log Format (P1 - This Week)

### 5.1 TechnicianLog Schema

```python
# agentic/technician_log.py

@dataclass
class TechnicianLog:
    """Log entry optimized for technician consumption."""

    # Header
    timestamp: datetime
    request_id: str
    query_summary: str  # 1-line summary

    # Confidence visualization
    confidence_score: float
    confidence_factors: List[Tuple[str, float]]  # [("source_quality", 0.9), ...]
    confidence_bar: str  # "████████░░ 80%"

    # Reasoning chain (human-readable)
    reasoning_steps: List[str]  # ["Identified SRVO-063 as encoder alarm", ...]

    # Source attribution
    sources_consulted: List[Dict]  # [{"title": "...", "type": "...", "relevance": 0.9}]

    # Uncertainty declaration
    uncertain_about: List[str]  # ["Model-specific variations", ...]
    additional_info_needed: List[str]  # ["Robot model", "Controller version"]

    # Actionable output
    recommended_action: str
    safety_warnings: List[str]

    # Backtracking history
    refinements_made: List[Dict]  # [{"reason": "...", "action": "..."}]

    def to_markdown(self) -> str:
        """Generate technician-friendly markdown."""
        lines = [
            f"## Diagnostic Summary",
            f"**Query**: {self.query_summary}",
            f"",
            f"### Confidence: {self.confidence_bar}",
            f"",
            f"### How I Reached This Conclusion:",
        ]

        for i, step in enumerate(self.reasoning_steps, 1):
            lines.append(f"{i}. {step}")

        if self.uncertain_about:
            lines.append(f"\n### What I'm Uncertain About:")
            for item in self.uncertain_about:
                lines.append(f"- {item}")

        if self.safety_warnings:
            lines.append(f"\n### ⚠️ Safety Warnings:")
            for warning in self.safety_warnings:
                lines.append(f"- {warning}")

        lines.append(f"\n### Recommended Next Step:")
        lines.append(self.recommended_action)

        lines.append(f"\n### Sources Consulted:")
        for source in self.sources_consulted[:5]:
            lines.append(f"- [{source['title']}]({source.get('url', '#')}) (relevance: {source['relevance']:.0%})")

        return "\n".join(lines)

    @staticmethod
    def confidence_to_bar(score: float) -> str:
        """Convert 0-1 score to visual bar."""
        filled = int(score * 10)
        empty = 10 - filled
        return f"{'█' * filled}{'░' * empty} {score:.0%}"
```

### 5.2 API Endpoint

```python
# In api/search.py

@router.get("/api/v1/search/{request_id}/technician-log")
async def get_technician_log(request_id: str) -> Dict:
    """Get technician-friendly log for a search request."""
    log = technician_log_store.get(request_id)
    if not log:
        raise HTTPException(404, "Log not found")

    return {
        "success": True,
        "data": {
            "markdown": log.to_markdown(),
            "structured": asdict(log)
        }
    }
```

---

## Phase 6: Connect OpenTelemetry (P2 - Next Week)

### 6.1 Wire Tracing to Orchestrator

```python
# In orchestrator_universal.py

from agentic.tracing import AgenticTracer, get_tracer

class UniversalOrchestrator:
    def __init__(self, ...):
        self.tracer = get_tracer()

    async def search_with_events(self, request, emitter):
        with self.tracer.trace_search_pipeline(
            request.id,
            request.query,
            self.preset.value
        ) as root_span:
            # Analysis phase
            with self.tracer.trace_analysis(request.id, request.query):
                analysis = await self.analyzer.analyze(...)

            # Search phase
            with self.tracer.trace_search(request.id, iteration=1):
                results = await self.searcher.search(...)

            # CRAG phase
            with self.tracer.trace_crag_evaluation(request.id, len(results)):
                crag_result = await self.retrieval_evaluator.evaluate(...)

            # ... etc for each phase
```

### 6.2 Add GenAI Semantic Conventions

```python
# Enhanced tracing.py

def trace_llm_call(self, request_id: str, agent: str, model: str, operation: str):
    """Create span with GenAI semantic conventions."""
    with self.tracer.start_as_current_span(f"llm.{operation}") as span:
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("gen_ai.agent.role", agent)
        span.set_attribute("request_id", request_id)
        yield span
```

---

## Phase 7: Confidence Breakdown Logging (P2 - Next Week)

### 7.1 Multi-Signal Confidence Logger

```python
# In orchestrator_universal.py

def _calculate_and_log_confidence(
    self,
    request_id: str,
    verification_score: float,
    source_diversity_score: float,
    content_depth_score: float,
    synthesis_quality_score: float
) -> float:
    """Calculate confidence with full breakdown logging."""

    # Current weights (from AGENTIC_IMPROVEMENT_PLAN.md)
    weights = {
        "verification": 0.40,
        "source_diversity": 0.25,
        "content_depth": 0.20,
        "synthesis_quality": 0.15
    }

    signals = {
        "verification": verification_score,
        "source_diversity": source_diversity_score,
        "content_depth": content_depth_score,
        "synthesis_quality": synthesis_quality_score
    }

    # Calculate weighted sum
    final_confidence = sum(
        signals[k] * weights[k] for k in weights
    )

    # Log breakdown
    logger.info(
        f"[{request_id}] Confidence breakdown: "
        f"verification={verification_score:.2f}×0.40={verification_score*0.40:.2f} | "
        f"diversity={source_diversity_score:.2f}×0.25={source_diversity_score*0.25:.2f} | "
        f"depth={content_depth_score:.2f}×0.20={content_depth_score*0.20:.2f} | "
        f"synthesis={synthesis_quality_score:.2f}×0.15={synthesis_quality_score*0.15:.2f} | "
        f"FINAL={final_confidence:.2f}"
    )

    # Emit SSE event
    await self.emitter.emit(SearchEvent(
        event_type=EventType.CONFIDENCE_CALCULATED,
        request_id=request_id,
        data={
            "signals": signals,
            "weights": weights,
            "final_confidence": final_confidence
        }
    ))

    return final_confidence
```

---

## New SSE Event Types to Add

| Event Type | Purpose | Data Fields |
|------------|---------|-------------|
| `DECISION_POINT` | Log any agent decision | agent, decision_type, decision, reasoning, confidence |
| `FEATURE_STATUS` | Feature enabled/skipped | feature_name, enabled, reason |
| `CONTEXT_TRANSFER` | Context flow between stages | source, target, type, token_count |
| `LLM_CALL_START` | LLM invocation started | agent, model, prompt_tokens |
| `LLM_CALL_COMPLETE` | LLM invocation finished | agent, model, response_tokens, latency_ms |
| `CONFIDENCE_CALCULATED` | Confidence breakdown | signals, weights, final_confidence |
| `SCRATCHPAD_FINDING_ADDED` | New finding added | finding_id, content_preview, confidence |
| `SCRATCHPAD_QUESTION_UPDATED` | Question status changed | question_id, old_status, new_status |
| `TECHNICIAN_LOG_READY` | Tech log available | summary, confidence_bar |

---

## API Endpoints to Add

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/search/{request_id}/technician-log` | GET | Technician-friendly log |
| `/api/v1/search/{request_id}/decisions` | GET | All decision points |
| `/api/v1/search/{request_id}/context-flow` | GET | Context flow diagram |
| `/api/v1/search/{request_id}/llm-calls` | GET | All LLM calls with timing |
| `/api/v1/search/{request_id}/scratchpad-history` | GET | Scratchpad state changes |
| `/api/v1/observability/dashboard` | GET | Aggregate observability stats |

---

## Configuration Flags

Add to `FeatureConfig`:

```python
# Observability flags
enable_verbose_llm_logging: bool = False  # Log full prompts/responses
enable_decision_logging: bool = True  # Log all decision points
enable_context_flow_tracking: bool = True  # Track context between stages
enable_scratchpad_observer: bool = True  # Track scratchpad changes
enable_technician_log: bool = True  # Generate tech-friendly logs
enable_otel_tracing: bool = False  # OpenTelemetry spans
```

---

## Implementation Priority

| Phase | Components | Effort | Impact |
|-------|------------|--------|--------|
| **P0** | Decision Logger, Context Tracker | 2 days | High - see why decisions made |
| **P1** | LLM Call Logger, Scratchpad Observer | 3 days | High - full pipeline visibility |
| **P1** | Technician Log Format | 2 days | High - Android usability |
| **P2** | OpenTelemetry Connection | 2 days | Medium - distributed tracing |
| **P2** | Confidence Breakdown | 1 day | Medium - calibration debugging |
| **P3** | Dashboard Endpoint | 2 days | Low - aggregate views |

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `agentic/decision_logger.py` | Centralized decision logging |
| `agentic/context_tracker.py` | Context flow tracking |
| `agentic/llm_logger.py` | LLM call instrumentation |
| `agentic/technician_log.py` | Technician-friendly log format |
| `agentic/scratchpad_observer.py` | Scratchpad state tracking |

### Files to Modify

| File | Modification |
|------|--------------|
| `orchestrator_universal.py` | Add all logging integrations |
| `events.py` | Add new event types |
| `scratchpad.py` | Add observer hooks |
| `analyzer.py` | Add decision logging |
| `synthesizer.py` | Add LLM call logging |
| `verifier.py` | Add context tracking |
| `retrieval_evaluator.py` | Add decision logging |
| `self_reflection.py` | Add decision logging |
| `api/search.py` | Add new endpoints |
| `tracing.py` | Connect to orchestrator |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Decision points logged | 100% of routing/action decisions |
| Context transfers tracked | All agent-to-agent flows |
| LLM calls instrumented | All calls with timing |
| Scratchpad changes tracked | All state mutations |
| Technician log generated | For every completed search |
| OpenTelemetry spans | All pipeline phases |

---

## Research Sources

- OpenTelemetry GenAI Semantic Conventions (2025)
- Siemens Industrial Explainable AI Whitepaper
- Amazon Bedrock Agent Tracing Documentation
- LangSmith Observability Best Practices
- NVIDIA Multi-Agent RAG Logging Patterns

---

*Last Updated: 2026-01-02 by Claude Code*
