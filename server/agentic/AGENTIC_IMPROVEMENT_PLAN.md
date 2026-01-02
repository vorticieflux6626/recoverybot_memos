# Agentic Pipeline Improvement Plan

> **Created**: 2026-01-02 | **Status**: Complete | **Version**: 2.0

## Executive Summary

This document outlines a comprehensive plan to improve the memOS agentic search pipeline's ability to effectively answer user questions. The plan is based on:
- **3 Internal Audits**: Context flow, directive propagation, scratchpad usage
- **2 Research Studies**: Agentic RAG best practices, answer evaluation methods

**Key Finding**: The memOS pipeline is **remarkably well-aligned** with 2025-2026 best practices. The implementation already includes cutting-edge patterns (CRAG, Self-RAG, GoT, FLARE, blackboard/scratchpad). Primary gaps are in **citation verification** and **agent coordination utilization**.

## Current State Assessment

### Test Results (2026-01-02)

| Metric | Current | Target |
|--------|---------|--------|
| Confidence Score | 57-69% | 80%+ |
| Term Coverage | 50% | 80%+ |
| Composite Score | 37% | 70%+ |
| Response Time | 120-180s | <60s |
| Citation Accuracy | ~50% | 100% |

### Known Issues

1. **Slow Response Time**: Fresh queries take 120-180s
2. **Low Term Coverage**: Synthesis missing expected domain terms
3. **Missing Citations**: Some answers lack `[Source N]` citations
4. **Confidence Calibration**: Scores don't reflect actual quality

---

## Audit Findings

### Audit 1: Context Flow (7 Issues)

| # | Issue | Severity | Description |
|---|-------|----------|-------------|
| 1 | Information Bottleneck Filter | HIGH | Creates single-element list, may over-filter useful content |
| 2 | Results reordering loses metadata | MEDIUM | Metadata not preserved during reordering |
| 3 | FLARE source tracking | MEDIUM | Augmentation adds docs but doesn't track sources |
| 4 | Scraped content type mismatch | MEDIUM | `List[str]` vs `List[Dict]` inconsistency |
| 5 | Decomposed questions not passed | MEDIUM | Questions never reach synthesis phase |
| 6 | Confidence calculation complex | MEDIUM | Multi-signal scoring hard to debug |
| 7 | Search trace incompleteness | LOW | Missing some pipeline steps in trace |

**Verification Complete**: Synthesis correctly flows to response via `build_response()` → `SearchResultData.synthesized_context`.

### Audit 2: Directive Propagation (HEALTHY)

**Status**: All critical directives propagate correctly.

| Directive | Path Length | Status |
|-----------|-------------|--------|
| `force_thinking_model` | 9 steps | Verified |
| `preset` | 5 steps | Verified |
| `max_iterations` | N/A | Validated with range checks |
| `max_sources` | N/A | Validated with range checks |

**Minor Gap**: `force_thinking_model` not explicitly in `ChatGatewayRequest` model (documented).

### Audit 3: Scratchpad/Shared Context

**Architecture Maturity Assessment**:

| Capability | Maturity | Notes |
|------------|----------|-------|
| Question Decomposition | HIGH (85%) | Clear criteria per question |
| Finding Attribution | HIGH (85%) | Strong source tracking |
| Confidence Scoring | HIGH (85%) | Multi-signal blending |
| Progress Tracking | HIGH (85%) | Completion status accurate |
| Gap Detection | HIGH (85%) | Coverage evaluation working |
| Contradiction Detection | HIGH (85%) | Detection logic solid |
| **Entity Tracking** | MEDIUM (45%) | Extraction works, **reasoning missing** |
| **Agent Communication** | MEDIUM (45%) | Infrastructure exists, **unused** |
| **Cross-Session Memory** | MEDIUM (45%) | A-MEM persists but **never retrieves** |
| **Contradiction Resolution** | LOW (25%) | Detected but **not resolved** |
| **Multi-Hop Reasoning** | LOW (25%) | Entities stored but **not traversed** |

**Key Finding**: Foundation is strong (85%), but coordination features severely underutilized (45%).

---

## Research Findings

### Research 1: Agentic RAG Best Practices (2025-2026)

**memOS Already Implements**:

| Feature | Status | Notes |
|---------|--------|-------|
| CRAG Pre-Synthesis Evaluation | Implemented | `retrieval_evaluator.py` |
| Self-RAG Post-Synthesis Reflection | Implemented | `self_reflection.py` |
| Blackboard/Scratchpad Architecture | Implemented | `scratchpad.py` |
| Graph of Thoughts Reasoning | Implemented | `reasoning_dag.py` |
| FLARE Active Retrieval | Implemented | `flare_retriever.py` |
| Hybrid Search (Dense+Sparse+RRF) | Implemented | `bge_m3_hybrid.py` |
| Cross-Encoder Reranking | Implemented | `cross_encoder.py` |
| RAGAS Evaluation | Implemented | `ragas.py` |
| HyDE Query Expansion | Implemented | `hyde.py` |
| DyLAN Agent Importance | Implemented | `dylan_agent_network.py` |
| Sufficient Context Classification | Implemented | `sufficient_context.py` |
| Experience Distillation | Implemented | `experience_distiller.py` |
| Classifier Feedback Loop | Implemented | `classifier_feedback.py` |
| Three-Tier Memory | Implemented | `memory_tiers.py` |
| OpenTelemetry Tracing | Implemented | `tracing.py` |

**Gaps Identified**:

| Gap | Priority | Effort |
|-----|----------|--------|
| Citation accuracy verification | P1 | 2-3 days |
| Accuracy threshold alerting | P1 | 1-2 days |
| Chunk-level citation linking | P2 | 3-5 days |
| Grounding score metric | P2 | 2-3 days |
| Context sufficiency gating synthesis | P2 | 1 day |

### Research 2: Answer Evaluation Methods (2025-2026)

**Recommended Frameworks**:

| Framework | Strengths | Human Correlation |
|-----------|-----------|-------------------|
| DeepEval | 14+ metrics, pytest integration | 80% |
| RAGAS | Reference-free, component-level | High |
| ARES | PPI confidence intervals | 59.3% better than RAGAS |
| TruLens | RAG Triad, OpenTelemetry | Production-ready |
| Prometheus 2 | Open LLM judge | 0.897 Pearson |

**Quality Dimensions to Implement**:

| Dimension | Weight | Threshold |
|-----------|--------|-----------|
| Faithfulness | 0.20 | >= 0.70 |
| Answer Relevancy | 0.15 | >= 0.65 |
| Factual Correctness | 0.20 | >= 0.60 |
| Citation Accuracy | 0.15 | >= 0.60 |
| Completeness | 0.15 | >= 0.65 |
| Groundedness | 0.10 | >= 0.70 |
| Hallucination Rate | 0.05 | <= 0.30 |

---

## Improvement Roadmap

### Phase 1: Quick Wins (P0 - Immediate)

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| Fix citation formatting in synthesis prompts | High | Low | TODO |
| Add term coverage requirements to prompts | High | Low | TODO |
| Calibrate confidence weights (40/25/20/15) | Medium | Low | TODO |
| Enable agent notes communication | Medium | Low | TODO |
| Activate public space for entity context | Medium | Low | TODO |

### Phase 2: Core Improvements (P1 - This Week)

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| Add citation accuracy verification | High | Medium | Created `test_answer_effectiveness.py` |
| Implement LLM-as-judge evaluation | High | Medium | Framework in test suite |
| Add accuracy threshold alerting | High | Low | TODO |
| Implement A-MEM cross-session retrieval | Medium | Medium | TODO |
| Use task hierarchy for progress visualization | Medium | Low | TODO |

### Phase 3: Advanced Optimizations (P2 - Next Week)

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| Optimize pipeline for <60s response | High | High | TODO |
| Implement multi-hop entity reasoning | Medium | Medium | TODO |
| Implement contradiction resolution agent | Medium | Medium | TODO |
| Build example reuse from RAISE trajectory | Medium | High | TODO |

### Phase 4: Advanced Features (P3 - Future)

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| Scratchpad-based agent selection (LbMAS) | Medium | High | TODO |
| KGT integration with FANUC corpus | Medium | Medium | TODO |
| ARIES-style policy agents for GoT | Low | High | TODO |

---

## Effectiveness Test Suite

### Test File: `tests/test_answer_effectiveness.py`

**Implemented Categories**:
- Factual queries with expected entities
- Diagnostic queries with causes/symptoms/solutions
- Procedural queries with steps/warnings
- Comparative queries with criteria/tradeoffs
- Troubleshooting queries with domain terms

**Metrics Evaluated**:
```python
@dataclass
class EffectivenessMetrics:
    question_alignment: float   # Semantic similarity Q→A
    term_coverage: float        # Expected terms found
    citation_accuracy: float    # Claims with valid citations
    confidence_calibration: float  # Confidence vs actual
    completeness: float         # Question aspects covered
    effectiveness_score: float  # Weighted composite
```

### Test Query Bank

```python
EFFECTIVENESS_TESTS = {
    "factual": [
        ("What causes FANUC SRVO-063 alarm?",
         ["overcurrent", "servo", "motor"],
         ["cause", "diagnosis", "solution"]),
    ],
    "diagnostic": [
        ("Why is my robot vibrating during operation?",
         ["vibration", "servo", "gain"],
         ["symptoms", "causes", "fixes"]),
    ],
    "procedural": [
        ("How do I perform zero mastering on FANUC robot?",
         ["mastering", "calibration", "encoder"],
         ["prerequisites", "steps", "verification"]),
    ],
    "comparative": [
        ("Compare DCS vs fence safety on FANUC robots",
         ["DCS", "fence", "safety"],
         ["features", "tradeoffs", "recommendations"]),
    ],
    "troubleshooting": [
        ("FANUC SRVO-063 alarm troubleshooting",
         ["SRVO-063", "overcurrent", "motor", "encoder"],
         ["error_code", "cause", "solution"]),
    ],
}
```

---

## Success Criteria

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Effectiveness Score | ~40% | 60% | 75% | 85% |
| Response Time | 180s | 150s | 90s | 60s |
| Citation Accuracy | ~50% | 80% | 95% | 100% |
| Term Coverage | 50% | 70% | 80% | 90% |

---

## Implementation Priority Summary

1. **P0 - Immediate**:
   - Fix citation formatting
   - Add term requirements to prompts
   - Enable agent notes/public space

2. **P1 - This Week**:
   - Citation accuracy verification
   - LLM-as-judge evaluation
   - Accuracy threshold alerting

3. **P2 - Next Week**:
   - Response time optimization
   - Multi-hop entity reasoning
   - A-MEM retrieval integration

4. **P3 - Future**:
   - LbMAS agent selection
   - Advanced semantic alignment
   - KGT integration

---

## Agent Coordination Quick Wins

### 1. Enable Agent Notes Communication

```python
# Post-action: Each agent adds recommendation
verifier_notes = scratchpad.get_notes_for_agent("synthesizer")
# Synthesizer reads and acts on recommendations
```

### 2. Activate Public Space for Entity Context

```python
# Entity Tracker writes extracted entities to public space
scratchpad.write_public(
    agent_id="entity_tracker",
    key="entities:q1",
    value={"name": "SRVO-063", "type": "error_code", ...}
)
# Synthesizer reads for context enrichment
entities = scratchpad.read_public("entities:q1")
```

### 3. Use Task Hierarchy for Progress Visualization

```python
# Return task tree in SSE events
# Android app displays progress visually
```

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `AGENTIC_IMPROVEMENT_PLAN.md` | This plan document |
| `CONTEXT_FLOW_AUDIT.md` | Context flow audit findings |
| `tests/test_answer_effectiveness.py` | Effectiveness test suite |

---

*Last Updated: 2026-01-02 by Claude Code*
