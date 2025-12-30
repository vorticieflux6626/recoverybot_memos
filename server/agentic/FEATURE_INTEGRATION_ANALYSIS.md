# Agentic Search Feature Integration Analysis

> **Updated**: 2025-12-30 | **Parent**: [AGENTIC_OVERVIEW.md](./AGENTIC_OVERVIEW.md) | **Status**: Superseded (See FEATURE_AUDIT_REPORT.md)

## Current State (December 2025)

The agentic module has **21+ implemented features**, but only **6 are actively used** in the main orchestrator pipeline.

### Features USED in Main Orchestrator

| Feature | Module | Purpose | Status |
|---------|--------|---------|--------|
| QueryClassifier | `query_classifier.py` | Route queries to appropriate pipeline | ✅ Active |
| SelfReflectionAgent | `self_reflection.py` | Self-RAG quality check post-synthesis | ✅ Active |
| RetrievalEvaluator | `retrieval_evaluator.py` | CRAG pre-synthesis quality gate | ✅ Active |
| ExperienceDistiller | `experience_distiller.py` | Learn from successful searches | ✅ Active |
| ClassifierFeedback | `classifier_feedback.py` | Adaptive classification tuning | ✅ Active |
| SufficientContextClassifier | `sufficient_context.py` | Context sufficiency checking | ✅ Active |

### Features NOT USED (Integration Opportunities)

#### Priority 1: Direct Search Quality Improvement

| Feature | Module | Purpose | Integration Point |
|---------|--------|---------|-------------------|
| **HyDE Query Expansion** | `hyde.py` | Generate hypothetical docs before search | Before web search |
| **BGE-M3 Hybrid Retrieval** | `bge_m3_hybrid.py` | Re-rank results with dense+sparse | After web search |
| **RAGAS Evaluation** | `ragas.py` | Quality scoring for RAG responses | After synthesis |
| **EntityTracker** | `entity_tracker.py` | Track entities across sources | During content analysis |

#### Priority 2: Enhanced Reasoning

| Feature | Module | Purpose | Integration Point |
|---------|--------|---------|-------------------|
| **ReasoningDAG** | `reasoning_dag.py` | Multi-path reasoning for complex queries | During planning/synthesis |
| **ThoughtLibrary** | `thought_library.py` | Reuse successful reasoning patterns | During planning |
| **EmbeddingAggregator** | `embedding_aggregator.py` | Route to domain experts | Before search |

#### Priority 3: Infrastructure/Advanced

| Feature | Module | Purpose | Integration Point |
|---------|--------|---------|-------------------|
| **ActorFactory** | `actor_factory.py` | Dynamic agent creation | Agent instantiation |
| **DynamicPlanner** | `dynamic_planner.py` | AIME-style hierarchical planning | Planning phase |
| **DomainCorpus** | `domain_corpus.py` | Domain-specific knowledge | RAG augmentation |
| **EntityEnhancedRetrieval** | `entity_enhanced_retrieval.py` | Entity-aware retrieval | After search |
| **MixedPrecisionEmbeddings** | `mixed_precision_embeddings.py` | Efficient embedding storage | Corpus indexing |
| **ProgressTools** | `progress_tools.py` | Progress reporting | UI updates |

---

## Proposed Integration Architecture

### Enhanced Agentic Pipeline

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. QUERY UNDERSTANDING                                        │
├──────────────────────────────────────────────────────────────┤
│ QueryClassifier → EntityTracker (extract entities)           │
│                 → EmbeddingAggregator (route to experts)     │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. QUERY EXPANSION (NEW)                                      │
├──────────────────────────────────────────────────────────────┤
│ HyDEExpander → Generate hypothetical document                │
│             → Create enhanced query embedding                 │
│             → Expand search terms from hypothetical          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. SEARCH EXECUTION                                           │
├──────────────────────────────────────────────────────────────┤
│ SearcherAgent → SearXNG/DDG/Brave (existing)                 │
│               → DomainCorpus (optional RAG)                  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. RESULT RE-RANKING (NEW)                                    │
├──────────────────────────────────────────────────────────────┤
│ BGEM3HybridRetriever → Embed search results                  │
│                      → Compute dense similarity              │
│                      → Apply BM25 sparse matching            │
│                      → RRF fusion for final ranking          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. CRAG EVALUATION (existing)                                 │
├──────────────────────────────────────────────────────────────┤
│ RetrievalEvaluator → Quality check re-ranked results         │
│                    → Trigger refinement if needed            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. SYNTHESIS                                                  │
├──────────────────────────────────────────────────────────────┤
│ SynthesizerAgent → Use ThoughtLibrary templates (NEW)        │
│                  → Track entities during synthesis           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. QUALITY ASSURANCE                                          │
├──────────────────────────────────────────────────────────────┤
│ SelfReflectionAgent → Self-RAG check (existing)              │
│ RAGASEvaluator → Faithfulness/relevancy scoring (NEW)        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Final Response with Quality Metrics
```

---

## Integration Implementation Plan

### Phase 1: HyDE Query Expansion

**Location**: Before `SearcherAgent.search()` in orchestrator

```python
# In search() method, before web search:
hyde_expander = await get_hyde_expander()
hyde_result = await hyde_expander.expand(
    query=request.query,
    mode=HyDEMode.SINGLE
)

# Use hypothetical for better search queries
if hyde_result.hypothetical_documents:
    enhanced_queries = self._extract_search_terms(
        hyde_result.hypothetical_documents[0]
    )
    state.pending_queries.extend(enhanced_queries)
```

**Expected Improvement**: 10-20% better recall by bridging query-document semantic gap.

### Phase 2: BGE-M3 Hybrid Re-ranking

**Location**: After `SearcherAgent.search()`, before CRAG evaluation

```python
# In search() method, after getting raw results:
hybrid_retriever = await get_hybrid_retriever()

# Index search results temporarily
for result in state.raw_results:
    await hybrid_retriever.add_document(
        doc_id=result.url,
        content=f"{result.title} {result.snippet}",
        metadata={"url": result.url, "domain": result.source_domain}
    )

# Re-rank with hybrid scoring
reranked = await hybrid_retriever.search(
    query=request.query,
    top_k=len(state.raw_results),
    mode=RetrievalMode.HYBRID
)

# Update result ordering based on hybrid scores
state.raw_results = self._apply_hybrid_ranking(state.raw_results, reranked)
```

**Expected Improvement**: Better precision by combining semantic (dense) and lexical (sparse) matching.

### Phase 3: RAGAS Quality Scoring

**Location**: After synthesis, alongside SelfReflection

```python
# In search() method, after synthesis:
ragas_evaluator = await get_ragas_evaluator()

ragas_result = await ragas_evaluator.evaluate(
    question=request.query,
    answer=synthesis,
    contexts=[r.snippet for r in state.verified_results[:5]]
)

# Blend RAGAS score with existing confidence
confidence_score = (
    0.4 * verifier_confidence +
    0.3 * self_reflection_score +
    0.3 * ragas_result.overall_score
)
```

**Expected Improvement**: More robust quality scoring with faithfulness/relevancy metrics.

### Phase 4: Entity Tracking

**Location**: Throughout pipeline for cross-source consistency

```python
# Initialize entity tracker at start
entity_tracker = await create_entity_tracker()

# Track entities from each source
for result in scraped_content:
    entities = await entity_tracker.extract_entities(result.content)
    state.scratchpad.add_entities(entities)

# Generate entity-aware context for synthesis
entity_context = entity_tracker.generate_context(request.query)
```

**Expected Improvement**: Better handling of entities mentioned across sources, coreference resolution.

---

## Files to Modify

| File | Changes |
|------|---------|
| `orchestrator.py` | Add HyDE, BGE-M3, RAGAS integration points |
| `api/search.py` | Expose new quality metrics in response |
| `events.py` | Add events for new processing stages |
| `models.py` | Add fields for hybrid scores, RAGAS metrics |

---

## Testing Plan

1. **Baseline**: Run current orchestrator on test queries, record confidence/quality
2. **Phase 1 Test**: Add HyDE, measure recall improvement
3. **Phase 2 Test**: Add hybrid re-ranking, measure precision improvement
4. **Phase 3 Test**: Add RAGAS, compare quality scoring accuracy
5. **Full Integration**: Run complete pipeline, measure end-to-end improvement

---

## Estimated Impact

| Metric | Current | With Integration | Improvement |
|--------|---------|------------------|-------------|
| Confidence Score | 0.71 | 0.82+ | +15% |
| Result Relevance | ~70% | ~85% | +21% |
| Query Coverage | Good | Excellent | +quality |
| Processing Time | 128s | 150s | +17% (tradeoff) |

The additional processing time is a reasonable tradeoff for significantly improved quality.
