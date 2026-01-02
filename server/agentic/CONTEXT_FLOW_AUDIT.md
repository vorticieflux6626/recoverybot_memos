# Agentic Pipeline Context Flow Audit

> **Audit Date**: 2026-01-02 | **Auditor**: Claude Code | **Version**: 1.0

## Executive Summary

This document analyzes the context flow through the memOS agentic search pipeline, based on code audit and research into 2025 agentic RAG best practices.

### Key Findings

1. **Synthesis Properly Flows to Response**: ✅ Verified
   - Synthesis from `_phase_synthesis()` correctly reaches `build_response()`
   - Response includes `synthesized_context` field with full synthesis

2. **Pipeline Routing Works**: ✅ Verified
   - Query analysis correctly routes to `direct_answer` vs `agentic_search`
   - `force_thinking_model` directive propagates correctly

3. **Cache Hit Issue**: ⚠️ Attention Needed
   - Similar queries hit cache, returning stale results
   - Test queries need timestamp suffix to bypass cache

4. **Context Stack Partially Implemented**: ⚠️ Room for Improvement
   - Current: system prefix + query + retrieved docs
   - Missing: explicit six-layer context stack model

## Context Flow Trace

```
Query Input
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Query Understanding                                         │
│   analyzer.analyze() → QueryAnalysis                                │
│   - key_topics, priority_domains, reasoning_complexity              │
│   - requires_thinking_model flag                                    │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Query Expansion (HyDE)                                      │
│   hyde.expand() → hypothetical document embedding                   │
│   - Bridges query-document semantic gap                             │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Search Planning                                             │
│   search_plan.decompose() → decomposed_questions[]                  │
│   - CRAG evaluation for retrieval quality                           │
│   - Query tree expansion if enabled                                 │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4-6: Web Search + Scraping                                     │
│   searcher.search() → raw_results[]                                 │
│   scraper.scrape_urls() → scraped_content[]                         │
│   - URL relevance evaluation                                        │
│   - Content extraction and cleaning                                 │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 7-8: Verification + Synthesis                                  │
│   verifier.verify_claims() → verification_results                   │
│   synthesizer.synthesize_with_content() → synthesis                 │
│   - Model override for thinking model                               │
│   - Source citations [Source N]                                     │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 9-10: Quality Evaluation                                       │
│   self_reflection.reflect() → ReflectionResult                      │
│   ragas.evaluate() → RAGASResult                                    │
│   - Refinement if confidence < threshold                            │
└─────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 11-12: Response Building                                       │
│   build_response(synthesis=...) → SearchResponse                    │
│   - synthesized_context = synthesis                                 │
│   - sources, confidence_score, search_trace                         │
└─────────────────────────────────────────────────────────────────────┘
    ↓
Response Output
```

## Research-Based Recommendations

### 1. Six-Layer Context Stack Model (Anthropic 2025)

Current implementation has implicit context layers. Research suggests explicit management:

| Layer | Status | Recommendation |
|-------|--------|----------------|
| **System Instructions** | ✅ Implemented | `CORE_SYSTEM_PREFIX` |
| **Long-term Memory** | ⚠️ Partial | Integrate A-MEM semantic memory more explicitly |
| **Retrieved Docs** | ✅ Implemented | `scraped_content[]` |
| **Tool Definitions** | ✅ Implicit | ActorFactory bundles |
| **Conversation History** | ❌ Missing | Add explicit history injection |
| **Current Task** | ✅ Implemented | Query + decomposed questions |

### 2. Context Window Optimization (12 Factor Agents)

Research identifies "dumb zone" at >40% context utilization:

```python
# Recommendation: Add context budget tracking
context_budget = model_context_window * 0.4  # 40% cap
current_utilization = len(prompt) / context_budget
if current_utilization > 0.8:
    # Summarize or truncate retrieved content
```

### 3. ACE Framework Integration

Agentic Context Engineering shows 14.8% improvement:

```python
# Recommended: Add structured lessons to context
context = {
    "system": SYSTEM_PREFIX,
    "lessons_learned": get_accumulated_lessons(query_type),  # NEW
    "retrieved_docs": scraped_content,
    "current_task": query
}
```

### 4. Multi-Agent Orchestration Patterns

Current implementation aligns with best practices:
- ✅ Specialized agents (Analyzer, Synthesizer, Verifier)
- ✅ Dynamic routing based on query analysis
- ✅ Tool bundles via ActorFactory
- ⚠️ Missing: Explicit agent-to-agent communication protocol

## Synthesis Output Verification

### Code Path Analysis

```
orchestrator_universal.py:4803
    synthesis = await self.synthesizer.synthesize_with_content(...)

orchestrator_universal.py:3886-3896
    return self.build_response(
        synthesis=synthesis,  # ← Synthesis passed here
        ...
    )

base_pipeline.py:472-473
    data=SearchResultData(
        synthesized_context=synthesis,  # ← Mapped to response field
```

### Verification: Synthesis Reaches Response ✅

The synthesis output correctly flows:
1. `_phase_synthesis()` generates synthesis via LLM
2. Synthesis passes through Self-RAG, RAGAS, adaptive refinement
3. Final synthesis passed to `build_response()`
4. `build_response()` maps to `SearchResultData.synthesized_context`
5. API returns `data.synthesized_context` to client

## Test Recommendations

### 1. Diverse Query Testing

Created `tests/test_agentic_diverse_queries.py` with:
- 40 queries across 5 categories (K/D/P/E/M)
- Random selection to avoid cache hits
- Category-specific quality scoring
- Composite score calculation

### 2. Context Flow Tracing

Add instrumentation:
```python
logger.info(f"[{request_id}] Context flow: analysis={len(str(query_analysis))} chars")
logger.info(f"[{request_id}] Context flow: scraped={len(scraped_content)} sources, {sum(len(c) for c in scraped_content)} chars")
logger.info(f"[{request_id}] Context flow: synthesis={len(synthesis)} chars")
```

### 3. Pipeline Distribution Monitoring

Track which pipeline paths are used:
- `direct_answer`: Simple queries (should be <20%)
- `web_search`: Basic search (should be ~40%)
- `agentic_search`: Complex queries (should be ~40%)

## References

- [Weaviate: What is Agentic RAG](https://weaviate.io/blog/what-is-agentic-rag)
- [Analytics Vidhya: Top 7 Agentic RAG Systems](https://www.analyticsvidhya.com/blog/2025/01/agentic-rag-system-architectures/)
- [Vellum: Multi-Agent Systems Best Practices](https://www.vellum.ai/blog/multi-agent-systems-building-with-context-engineering)
- [arXiv: Agentic Context Engineering (ACE)](https://arxiv.org/html/2510.04618v1)
- [Prompting Guide: Context Engineering](https://www.promptingguide.ai/guides/context-engineering-guide)
- [n8n: Agentic RAG Guide](https://blog.n8n.io/agentic-rag/)

## Modality Test Results (2026-01-02)

### Test Configuration

Created comprehensive modality test suite in `tests/test_agentic_modalities.py`:
- **27 modalities** defined across 11 categories
- **Preset-based testing**: Uses preset→modality mapping since API supports limited feature overrides
- **Timestamp suffix**: Added to queries to bypass semantic cache

### Modality Categories Tested

| Category | Modalities | Preset Required |
|----------|------------|-----------------|
| **Query Understanding** | query_analysis, entity_tracking | minimal, research |
| **Query Expansion** | hyde, query_tree, flare_retrieval | research |
| **Search Optimization** | hybrid_reranking, cross_encoder | research, full |
| **Pre-Synthesis Quality** | crag_evaluation, context_curation, information_bottleneck | enhanced, research |
| **Synthesis Enhancement** | thought_library, reasoning_dag, reasoning_composer, meta_buffer | research |
| **Post-Synthesis Quality** | self_reflection, ragas, verification | enhanced, balanced, full |
| **Iteration Control** | adaptive_refinement, entropy_halting, iteration_bandit | research |
| **Planning** | dynamic_planning, pre_act_planning | research, full |
| **Memory Systems** | semantic_memory, semantic_cache | research, balanced |
| **Domain Knowledge** | domain_corpus, technical_docs | full, enhanced |
| **Multi-Agent** | dylan_agent_skipping, contrastive_learning | research |

### Test Results

#### Successful Test: Balanced Preset (MOTN-063 query)

| Metric | Value |
|--------|-------|
| **Execution Time** | 175.1s |
| **Confidence** | 69% |
| **Sources Retrieved** | 10 |
| **Features Activated** | 6 |

**Features Activated:**
- `ttl_pinning` - KV cache TTL management
- `domain_corpus` - Domain-specific knowledge retrieval
- `crag` - Corrective RAG pre-synthesis evaluation
- `hybrid_reranking` - Dense+sparse fusion with RRF
- `positional_optimization` - Position-aware scoring
- `self_reflection` - Post-synthesis quality check

**Answer Quality:**
- Correctly identified MOTN-063 as collision error
- Cited related error codes (SRVO-063, SRVO-068, SRVO-069)
- Provided structured response with sections

#### Performance Observations

| Query Type | Typical Time | Notes |
|------------|--------------|-------|
| **Cached query** | <1s | Instant response from semantic cache |
| **Fresh minimal** | 120-180s | Times out at 120s threshold |
| **Fresh balanced** | 150-200s | Successful with 300s timeout |
| **Fresh research** | 200-300s | Full pipeline activation |

### Known Issues

1. **Timeout on Fresh Queries**: Fresh queries require 120-180s for full processing
   - **Cause**: Web scraping + multiple LLM calls + synthesis
   - **Mitigation**: Use longer timeouts (300s+) for testing

2. **Semantic Cache Bypass**: Similar queries hit cache
   - **Solution**: Add timestamp suffix `[test-{timestamp}]` to queries

3. **Feature Activation Verification**: Not all expected features show in `features_used`
   - Some features run but don't add to the list
   - Need to check individual phase logs

### Modality Implementation Status

| Modality | Implemented | Integrated | Tested |
|----------|-------------|------------|--------|
| query_analysis | ✅ | ✅ | ✅ |
| entity_tracking | ✅ | ✅ | ⚠️ |
| hyde | ✅ | ✅ | ⚠️ |
| query_tree | ✅ | ✅ | ⚠️ |
| flare_retrieval | ✅ | ✅ | ⚠️ |
| hybrid_reranking | ✅ | ✅ | ✅ |
| cross_encoder | ✅ | ✅ | ⚠️ |
| crag_evaluation | ✅ | ✅ | ✅ |
| context_curation | ✅ | ✅ | ⚠️ |
| information_bottleneck | ✅ | ✅ | ⚠️ |
| thought_library | ✅ | ✅ | ⚠️ |
| reasoning_dag | ✅ | ✅ | ⚠️ |
| reasoning_composer | ✅ | ✅ | ⚠️ |
| meta_buffer | ✅ | ✅ | ⚠️ |
| self_reflection | ✅ | ✅ | ✅ |
| ragas | ✅ | ✅ | ⚠️ |
| verification | ✅ | ✅ | ⚠️ |
| adaptive_refinement | ✅ | ✅ | ⚠️ |
| entropy_halting | ✅ | ✅ | ⚠️ |
| iteration_bandit | ✅ | ✅ | ⚠️ |
| dynamic_planning | ✅ | ✅ | ⚠️ |
| pre_act_planning | ✅ | ✅ | ⚠️ |
| semantic_memory | ✅ | ✅ | ⚠️ |
| semantic_cache | ✅ | ✅ | ✅ |
| domain_corpus | ✅ | ✅ | ✅ |
| technical_docs | ✅ | ✅ | ⚠️ |
| dylan_agent_skipping | ✅ | ✅ | ⚠️ |
| contrastive_learning | ✅ | ✅ | ⚠️ |

**Legend:** ✅ = Verified Working | ⚠️ = Implemented, Needs Testing

### Diverse Query Test Results

Ran `test_agentic_diverse_queries.py` with Knowledge category:

| Metric | Value |
|--------|-------|
| Success Rate | 100% |
| Confidence | 57% |
| Term Coverage | 50% (2/4 expected terms) |
| Composite Score | 37% |
| Execution Time | 213.5s |
| Pipeline | analyze |

**Quality Insights:**
- Missing source citations detected
- Term coverage at 50% suggests synthesis could better integrate key terms
- Composite score formula: `0.3*confidence + 0.4*term_coverage + 0.3*source_bonus`

### Next Steps for Modality Testing

1. **Increase timeout to 600s** for full pipeline tests
2. **Add unit tests** for each modality in isolation
3. **Add SSE event verification** to confirm feature execution
4. **Create modality benchmark suite** with expected behaviors
5. **Improve source citation formatting** in synthesis prompts

## Action Items

1. **P0**: Add context utilization tracking per agent
2. **P1**: Implement six-layer context stack explicitly
3. **P1**: Add conversation history injection for follow-up queries
4. **P2**: Integrate ACE-style lesson accumulation
5. **P2**: Add explicit agent-to-agent communication logging
6. **P2**: Create unit tests for each modality in isolation
7. **P3**: Optimize pipeline for <60s response times
